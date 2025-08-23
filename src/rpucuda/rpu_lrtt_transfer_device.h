/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "rpu_transfer_device.h"
#include <string>

namespace RPU {

/**
 * Update rule for LRTT devices
 */
enum class LRUpdateRule : int {
  LR_TT = 1     // LoRA-style updates (mathematically identical to LoRA)
};

template <typename T> class LRTTTransferRPUDevice;

/**
 * Low-Rank Tiki-Taka Transfer Device Meta Parameter
 * 
 * Implements low-rank decomposition for transfer learning with three devices:
 * - Visible/slow device (W_to): Main weight matrix [d_size, x_size]
 * - Fast device A: Low-rank factor stored in first K columns [d_size, x_size]
 * - Fast device B: Low-rank factor stored in first K rows [d_size, x_size]
 * 
 * Step-1 Convention: All devices have equal shape [d_size, x_size]
 * - A's first rank columns (A[:, :rank]) represent the d→rank mapping
 * - B's first rank rows (B[:rank, :]) represent the rank→x mapping
 * - Transfer performs: W_to += lr_scale * (A[:,:rank] @ B[:rank,:])
 * 
 * Step-2 will use dedicated buffers or heterogeneous shapes [d,r] and [r,x]
 */
template <typename T> 
struct LRTTTransferRPUDeviceMetaParameter : public TransferRPUDeviceMetaParameter<T> {
  
  // Device indices in the vector (visible must be last for parent compatibility)
  int idx_fastA = 0;    // Fast device A (rank->out)
  int idx_fastB = 1;    // Fast device B (in->rank)
  int idx_visible = 2;  // Slow/visible device (W_to) - MUST BE LAST
  
  // Low-rank parameters
  int rank = -1;  // Rank of the decomposition (-1 = auto-detect from device dims)
  int rank_chunk = -1;  // Process rank in chunks of this size (-1 = use full rank)
  int rank_offset = 0;  // Starting offset for rank chunking
  
  // Update rule configuration
  LRUpdateRule update_rule = LRUpdateRule::LR_TT;  // Only LR_TT is supported
  T lora_alpha = (T)1.0;   // LoRA alpha scaling factor (used when forward_inject = true)
  bool forward_inject = true;  // DEFAULT: true (LoRA-style analog compose in forward)
  
  // Transfer and update parameters
  bool correct_gradient_magnitudes = false;
  bool swap_xd = false;     // Swap X/D for update contract
  T transfer_lr = (T)1.0;   // Learning rate for transfer
  
  // BL (bound level) controls for PWU
  bool use_bl_management = false;  // Enable BL management in PWU
  T desired_BL = (T)0.0;  // Desired bound level (0 = use device default)
  T desired_bl = (T)0.0;  // deprecated alias for backward compatibility
  
  // Reinit-after-transfer (only option that remains)
  T reinit_gain = (T)1.0;  // Gain for Kaiming(He) normal on A; B is zero-initialized.
  
  LRTTTransferRPUDeviceMetaParameter() {};
  
  LRTTTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp_slow,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fastA,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fastB)
      : TransferRPUDeviceMetaParameter<T>() {
    // New canonical order: fastA, fastB, visible (visible MUST be last)
    this->vec_par.clear();
    this->appendVecPar(dp_fastA);  // idx 0: fast A
    this->appendVecPar(dp_fastB);  // idx 1: fast B  
    this->appendVecPar(dp_slow);   // idx 2: visible/slow (LAST)
  }
  
  void initializeWithSize(int x_size, int d_size) override;
  void printToStream(std::stringstream &ss) const override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);
  
  std::string getName() const override {
    std::ostringstream ss;
    ss << "LRTTTransfer(rank=" << rank << ")";
    return ss.str();
  }
  
  LRTTTransferRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new LRTTTransferRPUDevice<T>(x_size, d_size, *this, rng);
  }
  
  LRTTTransferRPUDeviceMetaParameter<T> *clone() const override {
    return new LRTTTransferRPUDeviceMetaParameter<T>(*this);
  }
  
  DeviceUpdateType implements() const override { 
    return DeviceUpdateType::Transfer;  // Reuse Transfer type, factory will check dynamic type
  }
};

/**
 * Low-Rank Tiki-Taka Transfer Device (CPU)
 * 
 * Implements low-rank transfer learning with three analog devices:
 * W_to += scale * (W_A @ W_B)
 */
template <typename T> 
class LRTTTransferRPUDevice : public TransferRPUDevice<T> {
  
public:
  using Base = TransferRPUDevice<T>;
  using MetaPar = LRTTTransferRPUDeviceMetaParameter<T>;
  
  // Constructors
  LRTTTransferRPUDevice() = default;
  
  explicit LRTTTransferRPUDevice(
      int x_size, int d_size, 
      const MetaPar &par, 
      RealWorldRNG<T> *rng);
  
  ~LRTTTransferRPUDevice() = default;
  
  // Copy/move constructors
  LRTTTransferRPUDevice(const LRTTTransferRPUDevice<T> &other);
  LRTTTransferRPUDevice<T> &operator=(const LRTTTransferRPUDevice<T> &other);
  LRTTTransferRPUDevice(LRTTTransferRPUDevice<T> &&other) noexcept;
  LRTTTransferRPUDevice<T> &operator=(LRTTTransferRPUDevice<T> &&other) noexcept;
  
  friend void swap(LRTTTransferRPUDevice<T> &a, LRTTTransferRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<TransferRPUDevice<T> &>(a), static_cast<TransferRPUDevice<T> &>(b));
    swap(a.rank_, b.rank_);
  }
  
  // Accessors - match parent signature exactly
  LRTTTransferRPUDeviceMetaParameter<T> &getPar() const override { 
    return static_cast<LRTTTransferRPUDeviceMetaParameter<T>&>(
        TransferRPUDevice<T>::getPar());
  }
  
  // Typed helper for internal use
  inline const LRTTTransferRPUDeviceMetaParameter<T>& getParTyped() const {
    return static_cast<const LRTTTransferRPUDeviceMetaParameter<T>&>(
        TransferRPUDevice<T>::getPar());
  }
  
  // Device accessors
  AbstractRPUDevice<T>* visibleDev() const;
  AbstractRPUDevice<T>* fastADev() const;
  AbstractRPUDevice<T>* fastBDev() const;
  
  // Get rank
  int getRank() const { return rank_; }
  
  // Override clone
  LRTTTransferRPUDevice<T> *clone() const override {
    return new LRTTTransferRPUDevice<T>(*this);
  }
  
  // CPU transfer implementation (mostly for debugging/fallback)
  virtual void doTransferCPU();
  
protected:
  int rank_ = -1;  // Actual rank after validation
  
  // Validation helper
  void validateTopology() const;
  
  // Reinitialize fast tiles (CPU placeholder)
  void reinitFastTilesCPU();
};

} // namespace RPU