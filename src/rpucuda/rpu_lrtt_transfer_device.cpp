/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "rpu_lrtt_transfer_device.h"
#include "rpu_pulsed_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <sstream>
#include <random>

namespace RPU {

/**************************************************************************************/
/* LRTTTransferRPUDeviceMetaParameter */

template <typename T>
void LRTTTransferRPUDeviceMetaParameter<T>::initializeWithSize(int x_size, int d_size) {
  
  // --- Validate vector device count upfront
  if (this->vec_par.size() != 3) {
    RPU_FATAL("LRTTTransferRPUDevice requires exactly 3 devices: [fastA, fastB, visible]");
  }
  
  // --- Validate visible must be the last device
  const int ndev = (int)this->vec_par.size();
  if (idx_visible != ndev - 1) {
    RPU_FATAL("LR-TT requires 'visible' to be the last device (idx_visible == n_devices - 1).");
  }
  
  // --- LR-TT disables parent read-based transfer altogether
  this->n_reads_per_transfer = 0;
  
  // Disable parent chain scheduling explicitly (no-op transfers)
  this->transfer_every_vec.assign(this->vec_par.size(), (T)0.0);
  
  // --- Force fully_hidden invariants
  // Visible must be the last device so that fullyHidden() (which checks back()) matches visible
  this->gamma_vec.assign(ndev, (T)0.0);
  this->gamma_vec[idx_visible] = (T)1.0;  // Only visible contributes to network weight
  
  // Critical: make sure scalar 'gamma' becomes 0 so fullyHidden() => true
  // (Parent will also set gamma = sum(gamma_vec[0..last-1]) == 0.)
  this->gamma = (T)0.0;
  
  // --- Update policy for LR-TT (CUDA handles A/B sequencing directly)
  this->update_policy = VectorDeviceUpdatePolicy::SingleFixed;
  this->first_update_idx = idx_fastA;
  this->same_context = true;
  
  // --- Now call the parent. It will validate sizes, confirm last gamma != 0,
  //     set 'gamma' consistently (0), and keep transfer_every_vec as all zeros.
  TransferRPUDeviceMetaParameter<T>::initializeWithSize(x_size, d_size);
  
  // Legacy field mapping is handled in loadExtra, not here
  
  // Validate rank if specified
  if (rank > 0) {
    if (rank > std::min(x_size, d_size)) {
      RPU_FATAL("LRTT rank " << rank << " exceeds min(d_size=" << d_size 
                << ", x_size=" << x_size << ")");
    }
  }
  
  // Validate update rule
  if (update_rule != LRUpdateRule::LR_TT) {
    RPU_FATAL("Invalid update_rule in LRTT device (only LR_TT is supported)");
  }
  
  // Validate transfer cadence for LR-TT
  if (this->transfer_every <= 0) {
    RPU_FATAL("LR-TT requires transfer_every > 0 to guarantee periodic transfer A@B -> visible");
  }
  
  // DEBUG: Log device indices and configuration
#ifdef AIHWKIT_DEBUG_LRTT
  std::cout << "[LR-TT DEBUG] initializeWithSize:" << std::endl;
  std::cout << "  idx_fastA=" << idx_fastA << ", idx_fastB=" << idx_fastB 
            << ", idx_visible=" << idx_visible << std::endl;
  std::cout << "  vec_par.size()=" << this->vec_par.size() << std::endl;
  std::cout << "  Device order: [fastA=" << idx_fastA << ", fastB=" << idx_fastB 
            << ", visible=" << idx_visible << "]" << std::endl;
  std::cout << "  update_policy=SingleFixed (CUDA handles A/B sequencing)" << std::endl;
  std::cout << "  fully_hidden mode: gamma=0, gamma_vec=[0,0,1]" << std::endl;
  std::cout << "  n_reads_per_transfer=0, transfer_every_vec=[0,0,0]" << std::endl;
#endif
}

template <typename T>
void LRTTTransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  ss << "\t LR-TT Transfer Device Parameters:" << std::endl;
  ss << "\t Device indices: visible=" << idx_visible 
     << ", fastA=" << idx_fastA 
     << ", fastB=" << idx_fastB << std::endl;
  ss << "\t Rank: " << rank << std::endl;
  ss << "\t Update rule: LR_TT" << std::endl;
  ss << "\t Transfer LR: " << transfer_lr << std::endl;
  ss << "\t LoRA alpha: " << lora_alpha << std::endl;
  ss << "\t Forward inject: " << forward_inject << std::endl;
  ss << "\t Correct gradient magnitudes: " << correct_gradient_magnitudes << std::endl;
  ss << "\t Swap X/D: " << swap_xd << std::endl;
  ss << "\t A/B Update BL Management: " << ab_use_bl_management 
     << ", desired_bl=" << ab_desired_bl << std::endl;
  ss << "\t Transfer BL Management: " << transfer_use_bl_management 
     << ", desired_bl=" << transfer_desired_bl << std::endl;
  // Digital transfer removed - only pulsed stochastic updates are used
  ss << "\t Step-1 Convention: First " << rank << " columns of A and first " 
     << rank << " rows of B used as LR factors" << std::endl;
  
  // Print parent parameters
  TransferRPUDeviceMetaParameter<T>::printToStream(ss);
}

/**************************************************************************************/
/* LRTTTransferRPUDevice */

template <typename T>
LRTTTransferRPUDevice<T>::LRTTTransferRPUDevice(
    int x_size, int d_size,
    const MetaPar &par,
    RealWorldRNG<T> *rng)
    : TransferRPUDevice<T>(x_size, d_size, par, rng) {
  
  // Validate topology
  validateTopology();
  
  // For Step-1: All devices have equal shape [d_size, x_size]
  // Use the rank parameter from config (default to min(d_size, x_size) / 4 if not specified)
  if (par.rank > 0) {
    rank_ = par.rank;
  } else {
    // Default rank heuristic
    rank_ = std::min(d_size, x_size) / 4;
    if (rank_ < 1) rank_ = 1;
  }
  
  // Validate rank is reasonable
  if (rank_ > std::min(d_size, x_size)) {
    RPU_FATAL("Rank " << rank_ << " exceeds matrix dimensions [" 
              << d_size << ", " << x_size << "]");
  }
}

template <typename T>
LRTTTransferRPUDevice<T>::LRTTTransferRPUDevice(const LRTTTransferRPUDevice<T> &other)
    : TransferRPUDevice<T>(other), rank_(other.rank_) {}

template <typename T>
LRTTTransferRPUDevice<T> &
LRTTTransferRPUDevice<T>::operator=(const LRTTTransferRPUDevice<T> &other) {
  if (this != &other) {
    TransferRPUDevice<T>::operator=(other);
    rank_ = other.rank_;
  }
  return *this;
}

template <typename T>
LRTTTransferRPUDevice<T>::LRTTTransferRPUDevice(LRTTTransferRPUDevice<T> &&other) noexcept
    : TransferRPUDevice<T>(std::move(other)), 
      rank_(other.rank_) {}

template <typename T>
LRTTTransferRPUDevice<T> &
LRTTTransferRPUDevice<T>::operator=(LRTTTransferRPUDevice<T> &&other) noexcept {
  if (this != &other) {
    TransferRPUDevice<T>::operator=(std::move(other));
    rank_ = other.rank_;
  }
  return *this;
}

template <typename T>
void LRTTTransferRPUDevice<T>::validateTopology() const {
  if (this->rpu_device_vec_.size() != 3) {
    RPU_FATAL("LRTTTransferRPUDevice requires exactly 3 devices");
  }
  
  const auto& par = getPar();
  if (par.idx_visible < 0 || par.idx_visible >= 3 ||
      par.idx_fastA < 0 || par.idx_fastA >= 3 ||
      par.idx_fastB < 0 || par.idx_fastB >= 3) {
    RPU_FATAL("Invalid device indices");
  }
  
  if (par.idx_visible == par.idx_fastA || 
      par.idx_visible == par.idx_fastB || 
      par.idx_fastA == par.idx_fastB) {
    RPU_FATAL("Device indices must be unique");
  }
  
  // CRITICAL: Visible must be the last device for fully_hidden mode to work correctly
  if (par.idx_visible != (int)this->rpu_device_vec_.size() - 1) {
    RPU_FATAL("LR-TT expects 'visible' to be the last device (idx_visible == n_devices - 1).");
  }
  
  // For Step-1: Enforce equal shapes for simplicity
  // All devices must be [d_size, x_size]
  // Low-rank content will be stored in auxiliary buffers in Step-2
  for (const auto& device : this->rpu_device_vec_) {
    if (device->getDSize() != this->d_size_ || device->getXSize() != this->x_size_) {
      RPU_FATAL("For Step-1, all devices must have shape [" << this->d_size_ 
                << ", " << this->x_size_ << "]. Got [" << device->getDSize() 
                << ", " << device->getXSize() << "]");
    }
  }
}

template <typename T>
AbstractRPUDevice<T>* LRTTTransferRPUDevice<T>::visibleDev() const {
  const auto& par = getPar();
  return this->rpu_device_vec_[par.idx_visible].get();
}

template <typename T>
AbstractRPUDevice<T>* LRTTTransferRPUDevice<T>::fastADev() const {
  const auto& par = getPar();
  return this->rpu_device_vec_[par.idx_fastA].get();
}

template <typename T>
AbstractRPUDevice<T>* LRTTTransferRPUDevice<T>::fastBDev() const {
  const auto& par = getPar();
  return this->rpu_device_vec_[par.idx_fastB].get();
}

template <typename T>
void LRTTTransferRPUDevice<T>::doTransferCPU() {
  // CPU fallback implementation for LR-TT transfer
  // Note: Full CPU implementation would require access to internal weight storage
  // which is not exposed in the abstract device interface.
  // The CUDA implementation handles the actual transfer logic.
  
  // For Step-2, we provide a placeholder that documents the intended behavior:
  // 1. Extract weight matrices from visible, fastA, and fastB devices
  // 2. Perform low-rank update: W_visible += transfer_lr * (A[:,:rank] @ B[:rank,:])
  // 3. Apply chunking if rank_chunk > 0
  // 4. Reinitialize fast tiles (A: Kaiming normal, B: zeros)
  
  // The actual implementation happens in the CUDA version via doTransfer()
  // which is called from the runUpdateKernel when transfer_every condition is met
}

template <typename T>
void LRTTTransferRPUDevice<T>::reinitFastTilesCPU() {
  // CPU placeholder for reinitializing fast tiles
  // Note: This would require access to internal weight storage
  // The actual implementation is handled in the CUDA version
  
  // Reinitialize A (entire matrix): Kaiming(He) normal initialization
  // Reinitialize B (entire matrix): all zeros
}

template <typename T>
void LRTTTransferRPUDeviceMetaParameter<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  // 1) TransferRPUDeviceMetaParameter doesn't have loadExtra, so we handle everything here
  auto state = RPU::selectWithPrefix(extra, prefix);

  // 2) update_rule: map old TT_Pure (0) -> LR_TT
  int ur = static_cast<int>(this->update_rule);
  RPU::load(state, "update_rule", ur, /*strict=*/false);
  if (ur == 0) { // legacy TT_Pure
    this->update_rule = LRUpdateRule::LR_TT;
    RPU_WARNING("LRTT: deprecated 'TT_Pure' mapped to 'LR_TT'.");
  } else {
    this->update_rule = LRUpdateRule::LR_TT; // only LR_TT is supported
  }

  // 3) forward_inject default changed to true; honor persisted value if present
  RPU::load(state, "forward_inject", this->forward_inject, /*strict=*/false);

  // 4) reinit_gain kept
  RPU::load(state, "reinit_gain", this->reinit_gain, /*strict=*/false);

  // 5) Load new BL management parameters
  // Check if new fields exist first
  bool has_new_fields = state.count("ab_use_bl_management") || state.count("transfer_use_bl_management");
  
  if (has_new_fields) {
    // Load new fields if present
    RPU::load(state, "ab_use_bl_management", this->ab_use_bl_management, /*strict=*/false);
    RPU::load(state, "ab_use_update_management", this->ab_use_update_management, /*strict=*/false);
    RPU::load(state, "ab_desired_bl", this->ab_desired_bl, /*strict=*/false);
    RPU::load(state, "transfer_use_bl_management", this->transfer_use_bl_management, /*strict=*/false);
    RPU::load(state, "transfer_use_update_management", this->transfer_use_update_management, /*strict=*/false);
    RPU::load(state, "transfer_desired_bl", this->transfer_desired_bl, /*strict=*/false);
    // Digital transfer removed - loading transfer_digital_bypass for backward compatibility only
    bool dummy_digital_bypass = false;
    RPU::load(state, "transfer_digital_bypass", dummy_digital_bypass, /*strict=*/false);
  } else {
    // Legacy checkpoint: map from old fields to new
    if (state.count("use_bl_management")) {
      RPU::load(state, "use_bl_management", this->ab_use_bl_management, /*strict=*/false);
      // Transfer defaults to false for BL management (new behavior)
      this->transfer_use_bl_management = false;
    }
    
    // Load legacy desired_BL/desired_bl
    T legacy_bl = (T)-1.0;
    if (state.count("desired_bl")) {
      RPU::load(state, "desired_bl", legacy_bl, /*strict=*/false);
    } else if (state.count("desired_BL")) {
      RPU::load(state, "desired_BL", legacy_bl, /*strict=*/false);
    }
    
    if (legacy_bl > 0) {
      this->ab_desired_bl = legacy_bl;
      // Transfer gets no override by default (sentinel)
      this->transfer_desired_bl = (T)-1.0;
    }
  }
  
  // Always load legacy fields for Python compatibility if they exist
  if (state.count("use_bl_management")) {
    RPU::load(state, "use_bl_management", this->use_bl_management, /*strict=*/false);
  }
  if (state.count("desired_BL")) {
    RPU::load(state, "desired_BL", this->desired_BL, /*strict=*/false);
  }
  if (state.count("desired_bl")) {
    RPU::load(state, "desired_bl", this->desired_bl, /*strict=*/false);
  }

  // 6) Soft-ignore removed keys unless strict=true
  auto ignore_removed = [&](const char* k){
    if (state.count(k)) {
      if (strict) {
        RPU_FATAL(std::string("LRTT: key '") + k + "' was removed in Step-1.");
      } else {
        RPU_WARNING(std::string("LRTT: ignoring removed key '") + k + "'.");
      }
    }
  };
  ignore_removed("gamma");
  ignore_removed("reset_policy");
  ignore_removed("reinit_only_lr_subspace");
  ignore_removed("reinit_randomize_A");
  ignore_removed("reinit_zero_B");
}

// Explicit instantiations
template struct LRTTTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct LRTTTransferRPUDeviceMetaParameter<double>;
#endif

template class LRTTTransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class LRTTTransferRPUDevice<double>;
#endif
// NO FP16 instantiation for LR-TT

} // namespace RPU