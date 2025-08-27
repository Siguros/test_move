/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */


#pragma once

#include "rpucuda_transfer_device.h"
#include "rpu_lrtt_transfer_device.h"
#include "cuda_util.h"
#include "maximizer.h"
#include "pulsed_weight_updater.h"
#include <memory>
#include <utility>          // for std::swap
#include <curand_kernel.h>  // for curandState_t
#include <cstdio>           // for printf in inline functions
#include <cstdlib>          // for std::getenv

namespace RPU {

/**
 * Low-Rank Tiki-Taka Transfer Device (CUDA)
 * 
 * GPU implementation of low-rank transfer learning.
 * 
 * Step-1 Convention (equal shapes):
 * - All devices are [d_size, x_size] for simplicity
 * - A's first rank columns store the low-rank d→rank mapping
 * - B's first rank rows store the low-rank rank→x mapping
 * - Transfer: W_to += scale * (A[:,:rank] @ B[:rank,:])
 * 
 * Both A and B are updated every training step with the same gradient.
 * Transfer occurs according to transfer_every schedule.
 * 
 * Step-2 will implement dedicated buffers/PWU routing for full analog semantics.
 */
template <typename T>
class LRTTTransferRPUDeviceCuda : public TransferRPUDeviceCuda<T> {
  
public:
  using Base = TransferRPUDeviceCuda<T>;
  using CUDAMetaPar = LRTTTransferRPUDeviceMetaParameter<T>;
  
  // Constructors
  explicit LRTTTransferRPUDeviceCuda() {};
  explicit LRTTTransferRPUDeviceCuda(CudaContextPtr c, const LRTTTransferRPUDevice<T> &cpu_device);
  
  ~LRTTTransferRPUDeviceCuda();
  
  // Copy/move constructors
  LRTTTransferRPUDeviceCuda(const LRTTTransferRPUDeviceCuda<T> &other);
  LRTTTransferRPUDeviceCuda<T> &operator=(const LRTTTransferRPUDeviceCuda<T> &other);
  LRTTTransferRPUDeviceCuda(LRTTTransferRPUDeviceCuda<T> &&other) noexcept;
  LRTTTransferRPUDeviceCuda<T> &operator=(LRTTTransferRPUDeviceCuda<T> &&other) noexcept;
  
  friend void swap(LRTTTransferRPUDeviceCuda<T> &a, LRTTTransferRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<TransferRPUDeviceCuda<T> &>(a), static_cast<TransferRPUDeviceCuda<T> &>(b));
    swap(a.rank_, b.rank_);
    swap(a.transfer_lr_, b.transfer_lr_);
    swap(a.transfer_every_, b.transfer_every_);
    swap(a.units_in_mbatch_, b.units_in_mbatch_);
    swap(a.n_reads_per_transfer_, b.n_reads_per_transfer_);
    swap(a.ab_use_bl_management_, b.ab_use_bl_management_);
    swap(a.ab_use_update_management_, b.ab_use_update_management_);
    swap(a.ab_desired_bl_, b.ab_desired_bl_);
    swap(a.transfer_use_bl_management_, b.transfer_use_bl_management_);
    swap(a.transfer_use_update_management_, b.transfer_use_update_management_);
    swap(a.transfer_desired_bl_, b.transfer_desired_bl_);
    swap(a.dev_w_visible_, b.dev_w_visible_);
    swap(a.dev_w_a_, b.dev_w_a_);
    swap(a.dev_w_b_, b.dev_w_b_);
    swap(a.dev_temp_x_, b.dev_temp_x_);
    swap(a.dev_temp_d_, b.dev_temp_d_);
    swap(a.dev_temp_x_T_, b.dev_temp_x_T_);
    swap(a.dev_xa_mb_, b.dev_xa_mb_);
    swap(a.dev_xb_mb_, b.dev_xb_mb_);
    swap(a.dev_db_mb_, b.dev_db_mb_);
    swap(a.dev_da_mb_, b.dev_da_mb_);
    swap(a.scratch_mb_capacity_, b.scratch_mb_capacity_);
    swap(a.fastA_pwu_, b.fastA_pwu_);
    swap(a.fastB_pwu_, b.fastB_pwu_);
    swap(a.visible_pwu_, b.visible_pwu_);
    swap(a.dev_x_pad_, b.dev_x_pad_);
    swap(a.dev_d_pad_, b.dev_d_pad_);
    swap(a.transfer_counter_, b.transfer_counter_);
    swap(a.num_a_updates_, b.num_a_updates_);
    swap(a.num_b_updates_, b.num_b_updates_);
    swap(a.num_transfers_, b.num_transfers_);
    swap(a.debug_no_host_copies_, b.debug_no_host_copies_);
    swap(a.reinit_counter_, b.reinit_counter_);
    swap(a.dev_fb_out_, b.dev_fb_out_);
    swap(a.dev_y_ab_, b.dev_y_ab_);
    swap(a.visible_synced_, b.visible_synced_);
    swap(a.last_agg_ptr_, b.last_agg_ptr_);
    swap(a.visible_sync_ev_, b.visible_sync_ev_);
    swap(a.rank_chunk_, b.rank_chunk_);
    swap(a.correct_gradient_magnitudes_, b.correct_gradient_magnitudes_);
    swap(a.forward_inject_, b.forward_inject_);
    swap(a.lora_alpha_, b.lora_alpha_);
    swap(a.reinit_gain_, b.reinit_gain_);
    swap(a.idx_fastA_, b.idx_fastA_);
    swap(a.idx_fastB_, b.idx_fastB_);
    swap(a.idx_visible_, b.idx_visible_);
    swap(a.update_rule_, b.update_rule_);
  }
  
  // Get parameter
  CUDAMetaPar &getPar() const override {
    return static_cast<CUDAMetaPar &>(TransferRPUDeviceCuda<T>::getPar());
  }
  
  // Clone
  LRTTTransferRPUDeviceCuda<T> *clone() const override {
    return new LRTTTransferRPUDeviceCuda<T>(*this);
  }
  
  // Override update kernel to trigger transfer
  void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContextPtr up_context,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) override;
  
  // Step-2: LR-TT uses only pulsed updates, no direct/FP path
  bool hasDirectUpdate() const override { return false; }
  
  void doDirectUpdate(
      const T *x_in,
      const T *d_in,
      T *dev_weights,
      const T lr,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const T beta,
      const PulsedUpdateMetaParameter<T> &up,
      T *x_buffer = nullptr,
      T *d_buffer = nullptr) override;
  
  // Override kernel selection to use fast A device
  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
  
  // Main transfer entry point (GPU-only)
  void doTransfer(cudaStream_t stream = nullptr);
  
  // Override populate to setup device pointers
  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  
  // Debug counter
  int getHostCopyCount() const { return debug_no_host_copies_; }
  
  // Debug stats
  void printDebugStats() const;
  
  // Forward-inject support: compose effective weights on device
  // W_eff = W_visible + alpha * (A[:, :rank] @ B[:rank, :])
  // Writes to dev_out [d_size_ * x_size_], device memory
  void composeForwardInject(T alpha, T* dev_out, cudaStream_t stream = nullptr);
  
  // Analog forward-inject entry point
  template <typename OutputIteratorT>
  bool forwardWithAnalogInject(
      OutputIteratorT out_values,
      InputOutputManager<T> &iom,
      ForwardBackwardPassIOManagedCuda<T> &fb,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool transposed);
  
  // Check if analog inject should be used
  bool shouldUseAnalogInject() const {
    const auto &par = getPar();
    return (par.update_rule == LRUpdateRule::LR_TT && par.forward_inject && rank_ > 0);
  }
  
  // Sync visible weights with aggregated weights (for setWeights)
  void syncVisibleWithAggregated(T* aggregated_weights, cudaStream_t stream = nullptr);
  
  // Bind the current aggregated weight buffer and mark visible as dirty
  inline void bindAggregatedPointer(T* agg) {
    last_agg_ptr_ = agg;        // remember which aggregated buffer we last saw
    visible_synced_ = false;    // force next sync (or immediate sync by the caller)
  }
  
  // Sub-tile access methods for debugging/inspection
  int getRank() const { return rank_; }
  
  void copyVisibleWeightsTo(T* dst, cudaStream_t stream = nullptr) const;
  void copyALRTo(T* dst, cudaStream_t stream = nullptr) const;
  void copyBLRTo(T* dst, cudaStream_t stream = nullptr) const;
  
  void copyVisibleWeightsFrom(const T* src, cudaStream_t stream = nullptr);
  void copyALRFrom(const T* src, cudaStream_t stream = nullptr);
  void copyBLRFrom(const T* src, cudaStream_t stream = nullptr);
  
  // Serialization support
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;
  
protected:
  // Initialize device pointers
  void initializeDevicePointers();
  inline void checkWeightsReadyOrDie();
  void ensureLazyInit(cudaStream_t stream = nullptr);
  
  // Override parent's transfer to use AB outer product
  void transfer(
      int to_device_idx,
      int from_device_idx,
      const PulsedUpdateMetaParameter<T> &current_up,
      const T current_lr,
      const T current_count_lr) override;
  
  // Transfer helpers
  void applyABOuterAsPulsedUpdate(T lr_scale, cudaStream_t stream);
  // Digital transfer removed - only pulsed stochastic updates are used
  void reinitFastTiles(cudaStream_t stream);
  
  // Get device weight pointers from child devices
  T* getVisibleWeightsDevice();
  T* getFastAWeightsDevice();
  T* getFastBWeightsDevice();
  
  // LoRA-specific scratch buffer management
  void ensureScratchBuffers(int m_batch);
  
  // Scratch for packed A_lr / B_lr and their gradients
  void ensureABScratch_();
  
  // Helper to ensure padded buffers for pulsed path
  void ensurePaddedBuffers(int m_batch);
  
  // LoRA update helpers
  void doLoRAUpdate(
      const T *x_in,
      const T *d_in,
      T *dev_weights,
      const T lr,
      const int m_batch,
      cudaStream_t stream,
      const T beta = (T)1.0);
  
  // Pulsed LoRA update helper
  void doLoRAPulsedUpdate_(
      const T *x_in,
      const T *d_in,
      const T lr,
      const int m_batch,
      const PulsedUpdateMetaParameter<T> &up,
      CudaContextPtr up_context);
  
private:
  int rank_ = -1;
  T transfer_lr_ = (T)1.0;  // Store transfer_lr locally
  int transfer_every_ = 1;  // Store transfer_every locally (integer cadence)
  bool units_in_mbatch_ = false;  // Store units_in_mbatch locally
  int n_reads_per_transfer_ = 0;  // Store n_reads_per_transfer locally (must be 0 for LRTT)
  
  // BL management parameters (stored locally to avoid cast issues)
  bool ab_use_bl_management_ = true;
  bool ab_use_update_management_ = true;
  T ab_desired_bl_ = (T)-1.0;
  bool transfer_use_bl_management_ = false;
  bool transfer_use_update_management_ = false;
  T transfer_desired_bl_ = (T)-1.0;
  
  // Additional LRTT parameters (stored locally to avoid cast issues)
  int rank_chunk_ = -1;  // Process rank in chunks of this size (-1 = use full rank)
  bool correct_gradient_magnitudes_ = false;  // Whether to correct gradient magnitudes
  bool forward_inject_ = true;  // Enable forward injection (default: true)
  T lora_alpha_ = (T)1.0;  // LoRA alpha scaling factor
  T reinit_gain_ = (T)1.0;  // Gain for Kaiming(He) normal on A
  
  // Device indices (stored locally to avoid cast issues)
  int idx_fastA_ = 0;  // Index of fast device A in the vector
  int idx_fastB_ = 1;  // Index of fast device B in the vector
  int idx_visible_ = 2;  // Index of visible device in the vector
  
  // Update rule (stored locally to avoid cast issues)
  LRUpdateRule update_rule_ = LRUpdateRule::LR_TT;  // Fixed to LR_TT
  
  // Lazy initialization flag for reinit
  bool need_reinit_ = true;  // Set to true to trigger reinit on first use
  
  // Device weight pointers (no ownership, just references)
  T *dev_w_visible_ = nullptr;
  T *dev_w_a_ = nullptr;
  T *dev_w_b_ = nullptr;
  
  // Temporary buffers for X and D
  std::unique_ptr<CudaArray<T>> dev_temp_x_;
  std::unique_ptr<CudaArray<T>> dev_temp_d_;
  std::unique_ptr<CudaArray<T>> dev_temp_x_T_; // Transpose buffer for AB-outer transfer [x × chunk]
  
  // LoRA scratch buffers for projections
  std::unique_ptr<CudaArray<T>> dev_xa_mb_;  // X_A = B_lr @ X, shape [rank, m_batch]
  std::unique_ptr<CudaArray<T>> dev_xb_mb_;  // X_B = B_lr @ X, shape [rank, m_batch] (for pulsed path)
  std::unique_ptr<CudaArray<T>> dev_db_mb_;  // D_B = A_lr^T @ D, shape [rank, m_batch]
  std::unique_ptr<CudaArray<T>> dev_da_mb_;  // D_A = A_lr^T @ D, shape [rank, m_batch] (alias for pulsed)
  int scratch_mb_capacity_ = 0;              // Track allocated batch size
  
  // Pulsed LoRA path support (Step-2: all updates via pulsed)
  std::unique_ptr<PulsedWeightUpdater<T>> fastA_pwu_;
  std::unique_ptr<PulsedWeightUpdater<T>> fastB_pwu_;
  std::unique_ptr<PulsedWeightUpdater<T>> visible_pwu_; // Step-2: pulsed transfer to visible tile
  std::unique_ptr<CudaArray<T>> dev_x_pad_; // [x_size * mb] - zero-padded X with X_B in first rank rows
  std::unique_ptr<CudaArray<T>> dev_d_pad_; // [d_size * mb] - zero-padded D with D_A in first rank rows
  
  // Transfer counter
  int transfer_counter_ = 0;
  
  // Debug counters for update tracking
  int num_a_updates_ = 0;
  int num_b_updates_ = 0;
  int num_transfers_ = 0;
  
  // Debug counter for host copies
  mutable int debug_no_host_copies_ = 0;
  
  // Reinit counter for thread-safe random seed generation
  unsigned long long reinit_counter_ = 0;
  
  // Temporary buffers for analog forward-inject
  std::unique_ptr<CudaArray<T>> dev_fb_out_; // B*x full result [d_size * m_batch]
  std::unique_ptr<CudaArray<T>> dev_y_ab_;   // A*(B_lr*x)    [d_size * m_batch]
  
  // Fix A: Track visible weight sync state
  bool visible_synced_ = false;
  T* last_agg_ptr_ = nullptr;  // Last seen aggregated dev_weights buffer
  
  // Event for cross-stream synchronization of visible weight copies
  cudaEvent_t visible_sync_ev_ = nullptr;
  
public:
  // Helper to wait for visible sync completion on a consumer stream
  inline void waitVisibleSyncOn(cudaStream_t consumer) {
    if (visible_sync_ev_) {
      if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
        printf("[DEBUG] waiting on visible-sync event from stream %p\n", consumer);
      }
      CUDA_CALL(cudaStreamWaitEvent(consumer, visible_sync_ev_, 0));
    }
  }
};

} // namespace RPU