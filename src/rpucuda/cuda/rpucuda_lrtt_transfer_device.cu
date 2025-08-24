/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "rpucuda_lrtt_transfer_device.h"
#include "rpu_lrtt_transfer_device.h"  // For LRUpdateRule enum
#include "rpucuda_pulsed_device.h"     // For PulsedRPUDeviceBaseCuda
#include "forward_backward_pass.h"     // For ForwardBackwardPassIOManagedCuda
#include "io_manager.h"                 // For InputOutputManager
#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"
#include "pwu_kernel_parameter.h"
#include "pulsed_weight_updater.h"
#include <memory>
#include <algorithm>        // std::min, std::max
#include <cmath>            // std::round, std::sqrt
#include <cstdlib>          // std::getenv
#include <sstream>          // std::ostringstream
#include <cstdio>           // printf
#include <curand_kernel.h>  // curandState_t, curand_normal(_double)

namespace RPU {

// Sync visible weights with aggregated weights implementation
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::syncVisibleWithAggregated(T* aggregated, cudaStream_t stream) {
  // Early exit if no aggregated buffer provided
  if (!aggregated) {
    return;
  }
  
  // Initialize device pointers if needed
  if (!dev_w_visible_) {
    if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[DEBUG] syncVisibleWithAggregated: initializing device pointers\n");
    }
    initializeDevicePointers();
    if (!dev_w_visible_) {
      return;  // Still no visible pointer after init
    }
  }
  
  cudaStream_t s = stream ? stream : this->context_->getStream();
  size_t bytes = sizeof(T) * this->d_size_ * this->x_size_;
  bool same_ptr = (dev_w_visible_ == aggregated);
  
  // Perform copy if needed
  if (!same_ptr) {
    if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[DEBUG] syncVisibleWithAggregated: memcpy %zu bytes from %p to %p on stream %p\n",
             bytes, aggregated, dev_w_visible_, s);
    }
    CUDA_CALL(cudaMemcpyAsync(dev_w_visible_, aggregated, bytes, cudaMemcpyDeviceToDevice, s));
  } else {
    if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[DEBUG] syncVisibleWithAggregated: same pointer, no copy, stream %p\n", s);
    }
  }
  
  // Create event lazily if needed
  if (!visible_sync_ev_) {
    CUDA_CALL(cudaEventCreateWithFlags(&visible_sync_ev_, cudaEventDisableTiming));
  }
  
  // Record completion event for consumers to wait on
  CUDA_CALL(cudaEventRecord(visible_sync_ev_, s));
  
  if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[DEBUG] visible-sync event recorded on stream %p (bytes=%zu, same_ptr=%d)\n",
           s, bytes, same_ptr);
  }
  
  // Mark as synced (now means "event recorded", not necessarily finished on other streams)
  last_agg_ptr_ = aggregated;
  visible_synced_ = true;
}

// CUDA kernel for resetting weights to zero
template <typename T>
__global__ void kernelResetWeights(T *weights, int size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    weights[tid] = (T)0.0;
  }
}

// Pack kernels for extracting submatrices for PWU inputs
template <typename T>
__global__ void kernelPackFirstKCols(T* out, const T* in, int d, int x, int K) {
  // Pack first K columns of [d, x] col-major matrix into contiguous [d, K] col-major buffer
  // in is [d, x] col-major: element (r,c) at in[r + c*d]
  // out is [d, K] col-major: element (r,c) at out[r + c*d]
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..(d*K-1)
  if (i >= d * K) return;
  int row = i % d;  // col-major: row cycles faster
  int col = i / d;  // col-major: column cycles slower
  out[row + col*d] = in[row + col*d]; // read from full [d,x] at first K cols
}

template <typename T>
__global__ void kernelPackFirstKRows(T* out, const T* in, int d, int x, int K) {
  // Pack first K rows of [d, x] col-major matrix into contiguous [K, x] col-major buffer
  // in is [d, x] col-major: element (r,c) at in[r + c*d]
  // out is [K, x] col-major: element (r,c) at out[r + c*K]
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..(K*x-1)
  if (i >= K * x) return;
  int row = i % K;  // col-major: row cycles faster
  int col = i / K;  // col-major: column cycles slower
  out[row + col*K] = in[row + col*d]; // read from full [d,x] at first K rows
}

// Transpose kernel for col-major [K×x] → [x×K] (used in AB-outer transfer)
template <typename T>
__global__ void kernelTransposeKxXtoXxK(T* __restrict__ out,
                                        const T* __restrict__ in,
                                        int K, int x) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 0..K*x-1
  if (i >= K * x) return;
  int r_out = i % x;    // 0..x-1
  int c_out = i / x;    // 0..K-1
  // out is [x × K], in is [K × x], both col-major
  out[r_out + c_out * x] = in[c_out + r_out * K];
}

// Pack kernels with offset support for rank-chunking
template <typename T>
__global__ void kernelPackColsWithOffset(T* out, const T* in, int d, int x, int K, int offset) {
  // Pack K columns starting at offset from [d, x] col-major matrix into contiguous [d, K] col-major buffer
  // in is [d, x] col-major: element (r,c) at in[r + c*d]
  // out is [d, K] col-major: element (r,c) at out[r + c*d]
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..(d*K-1)
  if (i >= d * K) return;
  int row = i % d;  // col-major: row cycles faster
  int col = i / d;  // col-major: column cycles slower
  out[row + col*d] = in[row + (col + offset)*d]; // read from [d,x] at columns [offset:offset+K]
}

// Simple 1-D AXPY: y[i] += alpha * x[i]
template <typename T>
__global__ void kernelAxpy1D(T* __restrict__ y, const T* __restrict__ x, int n, T alpha) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) y[i] += alpha * x[i];
}

template <typename T>
__global__ void kernelPackRowsWithOffset(T* out, const T* in, int d, int x, int K, int offset) {
  // Pack K rows starting at offset from [d, x] col-major matrix into contiguous [K, x] col-major buffer
  // in is [d, x] col-major: element (r,c) at in[r + c*d]
  // out is [K, x] col-major: element (r,c) at out[r + c*K]
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..(K*x-1)
  if (i >= K * x) return;
  int row = i % K;  // col-major: row cycles faster
  int col = i / K;  // col-major: column cycles slower
  out[row + col*K] = in[(row + offset) + col*d]; // read from [d,x] at rows [offset:offset+K]
}

// ---- LoRA-style reinit kernels ----
template <typename T>
__device__ inline T _curand_gauss(curandState_t* states, int idx);

template <>
__device__ inline float _curand_gauss<float>(curandState_t* states, int idx) {
  return curand_normal(states + idx);
}

#ifdef RPU_USE_DOUBLE
template <>
__device__ inline double _curand_gauss<double>(curandState_t* states, int idx) {
  return curand_normal_double(states + idx);
}
#endif

// Stateless Kaiming Normal initialization for first K columns
// Uses Philox RNG with per-thread initialization - no external state needed
template <typename T>
__global__ void kernelKaimingInitFirstKColsStateless(
    T* W, int d, int x_size, int K, unsigned long long seed, T std) {
  
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 0..(d*K-1)
  int n = d * K;
  if (i >= n) return;
  
  int row = i % d;     // col-major: row cycles faster
  int col = i / d;     // 0..K-1 (column index in the K columns)
  int idx = row + col * d;  // col-major indexing
  
  // Initialize Philox RNG state for this thread
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, /*subsequence=*/(unsigned long long)i, /*offset=*/0ULL, &rng);
  
  // Generate normal deviate and scale by std
  T z = (T)curand_normal(&rng) * std;
  W[idx] = z;
}

// B[:K, :] -> zero
// Matrix shape is col-major [d, x_size]; we touch the first K rows.
template <typename T>
__global__ void kernelZeroFirstKRows(T* W, int d, int x_size, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 0..(K*x_size-1)
  if (i >= K * x_size) return;
  int row = i % K;       // 0..K-1 (row index in the K rows)
  int col = i / K;       // 0..x_size-1
  int idx = row + col * d;  // col-major indexing
  W[idx] = (T)0.0;
}

// Stateless Kaiming Normal for first K rows (fill B_lr = B[:K, :])
template <typename T>
__global__ void kernelKaimingInitFirstKRowsStateless(
    T* W, int d, int x_size, int K, unsigned long long seed, T std) {
  // W is [d, x_size] col-major. We fill rows 0..K-1 across all columns.
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 0..(K*x_size-1)
  int n = K * x_size;
  if (i >= n) return;
  int row = i % K;         // 0..K-1
  int col = i / K;         // 0..x_size-1
  int idx = row + col * d; // col-major index

  curandStatePhilox4_32_10_t rng;
  curand_init(seed, static_cast<unsigned long long>(i), 0ULL, &rng);

  T z = (T)curand_normal(&rng) * std;
  W[idx] = z;
}

// ---- LoRA update kernels ----
// Scale and update the first K columns of a matrix
template <typename T>
__global__ void kernelScaleAndAxpyCols(
    T* __restrict__ W, const T* __restrict__ delta,
    int d, int ncols, T beta, T neg_lr) {
  // W[:, :ncols] = beta * W[:, :ncols] + neg_lr * delta
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int n = d * ncols;
  if (i >= n) return;
  T w = W[i];
  T g = delta[i];
  W[i] = beta * w + neg_lr * g;
}

// Scale and update the first K rows of a matrix
template <typename T>
__global__ void kernelScaleAndAxpyRows(
    T* __restrict__ W, const T* __restrict__ delta,
    int d, int x, int nrows, T beta, T neg_lr) {
  // W[:nrows, :] = beta * W[:nrows, :] + neg_lr * delta
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int n = nrows * x;
  if (i >= n) return;
  int row = i % nrows;
  int col = i / nrows;
  int w_idx = row + col * d;      // in W[d, x]
  int delta_idx = row + col * nrows;  // in delta[nrows, x]
  T w = W[w_idx];
  T g = delta[delta_idx];
  W[w_idx] = beta * w + neg_lr * g;
}

// Scatter kernel for pulsed LoRA path
template <typename T>
__global__ void kernelScatterRankRowsToPadded(
    T* dst, int ld_dst, const T* src, int ld_src, int rank, int full_rows, int cols) {
  // dst: [full_rows, cols] col-major, fill rows [0..rank-1] from src: [rank, cols]
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 0..(rank*cols-1)
  if (i >= rank * cols) return;
  int r = i % rank;      // 0..rank-1
  int c = i / rank;      // 0..cols-1
  dst[r + c * ld_dst] = src[r + c * ld_src];
}

// Unpack kernels for sub-tile setters
template <typename T>
__global__ void kernelUnpackToFirstKCols(T* dst, const T* src, int d, int x, int K) {
  // Unpack compact [d, K] src into first K columns of [d, x] dst
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d * K) return;
  int row = i % d;
  int col = i / d;
  dst[row + col * d] = src[row + col * d];
}

template <typename T>
__global__ void kernelUnpackToFirstKRows(T* dst, const T* src, int d, int x, int K) {
  // Unpack compact [K, x] src into first K rows of [d, x] dst
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= K * x) return;
  int row = i % K;
  int col = i / K;
  dst[row + col * d] = src[row + col * K];
}

// Step-2: Removed ad-hoc kernels, using existing kernelPack* functions instead

// Step-2: Helper to enforce StochasticCompressed for LR-TT
template <typename T>
static inline void enforceStochasticCompressed(PulsedUpdateMetaParameter<T>& up) {
  static bool warned_once = false;
  if (up.pulse_type != PulseType::StochasticCompressed) {
    if (!warned_once) {
      RPU_WARNING("LR-TT: forcing PulseType::StochasticCompressed (was " << (int)up.pulse_type << ")");
      warned_once = true;
    }
    up.pulse_type = PulseType::StochasticCompressed;
    up.update_management = true;
    up.update_bl_management = true;
    // Set reasonable default BL if not specified
    if (up.desired_BL <= 0) {
      up.desired_BL = 31;
    }
  }
}

#ifdef AIHWKIT_DEBUG_LRTT
  #ifndef RPU_USE_FP16
// DEBUG: Frobenius norm calculation for first K columns or rows
template <typename T>
__global__ void kernelFrobNormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // cols==true: norm of W[:, :K], else norm of W[:K, :]
  // W is col-major [d, x]: element (r,c) at W[r + c*d]
  __shared__ double acc[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  
  if (cols) {
    // sum over rows 0..d-1, cols 0..K-1
    for (int idx = i; idx < d*K; idx += blockDim.x * gridDim.x) {
      int r = idx % d;  // col-major: row cycles faster
      int c = idx / d;  // col-major: column cycles slower
      double v = (double)W[r + c*d];
      sum += v*v;
    }
  } else {
    // sum over rows 0..K-1, cols 0..x-1
    for (int idx = i; idx < K*x; idx += blockDim.x * gridDim.x) {
      int r = idx % K;  // for first K rows
      int c = idx / K;  // column index
      double v = (double)W[r + c*d];
      sum += v*v;
    }
  }
  
  acc[tid] = sum;
  __syncthreads();
  
  // Reduce
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) acc[tid] += acc[tid + s];
    __syncthreads();
  }
  
  if (tid == 0) {
    // FIX: Just accumulate sum of squares, don't take sqrt here
    atomicAdd(out_norm, (T)acc[0]);
  }
}
  #else
// FP16 build: provide inline no-op stub with identical signature
template <typename T>
__global__ void kernelFrobNormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // No-op stub for FP16 build
}
  #endif
#else
// Debug OFF: always provide inline no-op stub (identical signature)
template <typename T>
__global__ void kernelFrobNormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // No-op stub when debug is off
}
#endif

#ifdef AIHWKIT_DEBUG_LRTT
  #ifndef RPU_USE_FP16
// DEBUG: L1 norm calculation for first K columns or rows
template <typename T>
__global__ void kernelL1NormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // cols==true: norm of W[:, :K], else norm of W[:K, :]
  // W is col-major [d, x]: element (r,c) at W[r + c*d]
  __shared__ double acc[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  
  if (cols) {
    // sum over rows 0..d-1, cols 0..K-1
    for (int idx = i; idx < d*K; idx += blockDim.x * gridDim.x) {
      int r = idx % d;  // col-major: row cycles faster
      int c = idx / d;  // col-major: column cycles slower
      double v = (double)W[r + c*d];
      sum += fabs(v);
    }
  } else {
    // sum over rows 0..K-1, cols 0..x-1
    for (int idx = i; idx < K*x; idx += blockDim.x * gridDim.x) {
      int r = idx % K;  // for first K rows
      int c = idx / K;  // column index
      double v = (double)W[r + c*d];
      sum += fabs(v);
    }
  }
  
  acc[tid] = sum;
  __syncthreads();
  
  // Reduce
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) acc[tid] += acc[tid + s];
    __syncthreads();
  }
  
  if (tid == 0) {
    atomicAdd(out_norm, (T)acc[0]);
  }
}
  #else
// FP16 build: provide inline no-op stub with identical signature
template <typename T>
__global__ void kernelL1NormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // No-op stub for FP16 build
}
  #endif
#else
// Debug OFF: always provide inline no-op stub (identical signature)
template <typename T>
__global__ void kernelL1NormFirstK(const T* W, int d, int x, int K, T* out_norm, bool cols) {
  // No-op stub when debug is off
}
#endif

#ifdef AIHWKIT_DEBUG_LRTT
  #ifndef RPU_USE_FP16
// DEBUG: Helper to compute and print Frobenius norms
template <typename T>
void printFrobNorms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                   int d, int x, int K, const char* label) {
  T norm_A = 0, norm_B = 0, norm_W = 0;
  
  // Allocate temporary scalars on device
  CudaArray<T> dev_norm_A(context, 1);
  CudaArray<T> dev_norm_B(context, 1);
  CudaArray<T> dev_norm_W(context, 1);
  
  dev_norm_A.setConst(0.0);
  dev_norm_B.setConst(0.0);
  dev_norm_W.setConst(0.0);
  
  dim3 blocks((d*K + 255) / 256);
  dim3 threads(256);
  cudaStream_t s = context->getStream();
  
  // Compute sum of squares (not sqrt yet) on context stream
  kernelFrobNormFirstK<<<blocks, threads, 0, s>>>(dev_A, d, x, K, dev_norm_A.getData(), true);  // A[:,:K]
  kernelFrobNormFirstK<<<blocks, threads, 0, s>>>(dev_B, d, x, K, dev_norm_B.getData(), false); // B[:K,:]
  
  // Full W norm (note: only covers W[:,:min(d,x)])
  blocks = dim3((d*x + 255) / 256);
  kernelFrobNormFirstK<<<blocks, threads, 0, s>>>(dev_W, d, x, std::min(d,x), dev_norm_W.getData(), true);
  
  // Copy to host
  dev_norm_A.copyTo(&norm_A);
  dev_norm_B.copyTo(&norm_B);
  dev_norm_W.copyTo(&norm_W);
  
  context->synchronize();
  
  // Take sqrt on host after accumulation
  norm_A = std::sqrt(norm_A);
  norm_B = std::sqrt(norm_B);
  norm_W = std::sqrt(norm_W);
  
  printf("[LR-TT DEBUG %s] ||A[:,:K]||_F = %.6e, ||B[:K,:]||_F = %.6e, ||W[:,:min(d,x)]||_F = %.6e\n",
         label, (float)norm_A, (float)norm_B, (float)norm_W);
}
  #else
// FP16 build: provide inline no-op stub with identical signature
template <typename T>
inline void printFrobNorms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                          int d, int x, int K, const char* label) {
  (void)context; (void)dev_A; (void)dev_B; (void)dev_W;
  (void)d; (void)x; (void)K; (void)label;
}
  #endif
#else
// Debug OFF: always provide inline no-op stub (identical signature)
template <typename T>
inline void printFrobNorms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                          int d, int x, int K, const char* label) {
  (void)context; (void)dev_A; (void)dev_B; (void)dev_W;
  (void)d; (void)x; (void)K; (void)label;
}
#endif

#ifdef AIHWKIT_DEBUG_LRTT
  #ifndef RPU_USE_FP16
// DEBUG: Helper to compute and print L1 norms (more sensitive to small changes)
template <typename T>
void printL1Norms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                  int d, int x, int K, const char* label) {
  T l1_A = 0, l1_B = 0;
  
  // Allocate temporary scalars on device
  CudaArray<T> dev_l1_A(context, 1);
  CudaArray<T> dev_l1_B(context, 1);
  
  dev_l1_A.setConst(0.0);
  dev_l1_B.setConst(0.0);
  
  dim3 blocks((d*K + 255) / 256);
  dim3 threads(256);
  cudaStream_t s = context->getStream();
  
  // Compute L1 norms on context stream
  kernelL1NormFirstK<<<blocks, threads, 0, s>>>(dev_A, d, x, K, dev_l1_A.getData(), true);  // A[:,:K]
  kernelL1NormFirstK<<<blocks, threads, 0, s>>>(dev_B, d, x, K, dev_l1_B.getData(), false); // B[:K,:]
  
  // Copy to host
  dev_l1_A.copyTo(&l1_A);
  dev_l1_B.copyTo(&l1_B);
  
  context->synchronize();
  
  printf("[LR-TT L1 %s] |A[:,:K]|_1 = %.6e, |B[:K,:]|_1 = %.6e\n",
         label, (float)l1_A, (float)l1_B);
}
  #else
// FP16 build: provide inline no-op stub with identical signature
template <typename T>
inline void printL1Norms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                         int d, int x, int K, const char* label) {
  (void)context; (void)dev_A; (void)dev_B; (void)dev_W;
  (void)d; (void)x; (void)K; (void)label;
}
  #endif
#else
// Debug OFF: always provide inline no-op stub (identical signature)
template <typename T>
inline void printL1Norms(CudaContextPtr context, const T* dev_A, const T* dev_B, const T* dev_W,
                         int d, int x, int K, const char* label) {
  (void)context; (void)dev_A; (void)dev_B; (void)dev_W;
  (void)d; (void)x; (void)K; (void)label;
}
#endif

/**************************************************************************************/
/* LRTTTransferRPUDeviceCuda */

template <typename T>
LRTTTransferRPUDeviceCuda<T>::LRTTTransferRPUDeviceCuda(
    CudaContextPtr c, 
    const LRTTTransferRPUDevice<T> &cpu_device)
    : TransferRPUDeviceCuda<T>(c, static_cast<const TransferRPUDevice<T>&>(cpu_device)) {
  
  // Parameters are stored in base class
  
  // Extract rank from CPU device (Step-1: using config value, not device dimensions)
  rank_ = cpu_device.getRank();
  
  // Initialize counters to zero
  transfer_counter_ = 0;
  num_a_updates_ = 0;
  num_b_updates_ = 0;
  num_transfers_ = 0;
  debug_no_host_copies_ = 0;
  
  // Fix A: Reset sync state
  visible_synced_ = false;
  last_agg_ptr_ = nullptr;
  
  // Initialize after devices are created
  initializeDevicePointers();
  
  // Note: Learning rate validation is handled in pulsed update path
  
  // Allocate temporary buffers for PWU inputs
  int d_size = this->d_size_;
  int x_size = this->x_size_;
  
  // X buffer: [rank, x_size] for packed B rows
  // D buffer: [d_size, rank] for packed A columns
  if (rank_ > 0) {
    dev_temp_x_ = RPU::make_unique<CudaArray<T>>(c, rank_ * x_size);
    dev_temp_d_ = RPU::make_unique<CudaArray<T>>(c, d_size * rank_);
  }
  
  // Initialize PWU for analog transfer
  // Note: TransferRPUDeviceCuda already has transfer_pwu_ from parent class
  // We'll use that instead of creating our own
  
  // Initialize PWUs for pulsed path if we have valid devices
  if (rank_ > 0 && this->rpucuda_device_vec_.size() >= 3) {
    const auto& par = getPar();
    
    // Check if fastA device supports pulsed updates
    if (par.idx_fastA < this->rpucuda_device_vec_.size() && 
        this->rpucuda_device_vec_[par.idx_fastA]) {
      auto* devA = dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(
          this->rpucuda_device_vec_[par.idx_fastA].get());
      if (devA) {
        // Create PWU for fastA (d_size x x_size for full matrix)
        fastA_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);
        if (getenv("AIHWKIT_DEBUG_LRTT")) {
          printf("[LR-TT CUDA] Created PWU for fastA: %dx%d\n", d_size, x_size);
        }
      }
    }
    
    // Check if fastB device supports pulsed updates
    if (par.idx_fastB < this->rpucuda_device_vec_.size() && 
        this->rpucuda_device_vec_[par.idx_fastB]) {
      auto* devB = dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(
          this->rpucuda_device_vec_[par.idx_fastB].get());
      if (devB) {
        // Create PWU for fastB (d_size x x_size for full matrix)
        fastB_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);
        if (getenv("AIHWKIT_DEBUG_LRTT")) {
          printf("[LR-TT CUDA] Created PWU for fastB: %dx%d\n", d_size, x_size);
        }
      }
    }
  }
  
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT CUDA] Initialized with rank=%d, PWUs: A=%s, B=%s\n", 
           rank_, 
           fastA_pwu_ ? "YES" : "NO",
           fastB_pwu_ ? "YES" : "NO");
  }
  
  c->synchronize();
}

// Destructor
template <typename T>
LRTTTransferRPUDeviceCuda<T>::~LRTTTransferRPUDeviceCuda() {
  if (visible_sync_ev_) {
    cudaEventDestroy(visible_sync_ev_);
    visible_sync_ev_ = nullptr;
  }
}

// Copy constructor
template <typename T>
LRTTTransferRPUDeviceCuda<T>::LRTTTransferRPUDeviceCuda(
    const LRTTTransferRPUDeviceCuda<T> &other)
    : TransferRPUDeviceCuda<T>(other),
      rank_(other.rank_),
      transfer_counter_(0),
      num_a_updates_(0),
      num_b_updates_(0),
      num_transfers_(0),
      debug_no_host_copies_(0),
      visible_synced_(false),  // Force re-sync on first use after copy
      last_agg_ptr_(nullptr),
      visible_sync_ev_(nullptr) {  // Never copy event handles
  
  // Re-allocate temporary buffers
  if (other.dev_temp_x_) {
    dev_temp_x_ = RPU::make_unique<CudaArray<T>>(*other.dev_temp_x_);
  }
  if (other.dev_temp_d_) {
    dev_temp_d_ = RPU::make_unique<CudaArray<T>>(*other.dev_temp_d_);
  }
  
  // Copy scratch buffers if they exist
  if (other.dev_xa_mb_) {
    dev_xa_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_xa_mb_);
  }
  if (other.dev_xb_mb_) {
    dev_xb_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_xb_mb_);
  }
  if (other.dev_db_mb_) {
    dev_db_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_db_mb_);
  }
  if (other.dev_da_mb_) {
    dev_da_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_da_mb_);
  }
  scratch_mb_capacity_ = other.scratch_mb_capacity_;
  
  // PWUs use lazy-init - do not copy
  fastA_pwu_.reset();
  fastB_pwu_.reset();
  visible_pwu_.reset();
  
  // Copy padded buffers if they exist
  if (other.dev_x_pad_) {
    dev_x_pad_ = RPU::make_unique<CudaArray<T>>(*other.dev_x_pad_);
  }
  if (other.dev_d_pad_) {
    dev_d_pad_ = RPU::make_unique<CudaArray<T>>(*other.dev_d_pad_);
  }
  
  // Re-initialize device pointers after copy
  initializeDevicePointers();
}

template <typename T>
LRTTTransferRPUDeviceCuda<T> &
LRTTTransferRPUDeviceCuda<T>::operator=(const LRTTTransferRPUDeviceCuda<T> &other) {
  if (this != &other) {
    TransferRPUDeviceCuda<T>::operator=(other);
    rank_ = other.rank_;
    transfer_counter_ = 0;
    num_a_updates_ = 0;
    num_b_updates_ = 0;
    num_transfers_ = 0;
    debug_no_host_copies_ = 0;
    
    // Destroy existing event if any
    if (visible_sync_ev_) {
      cudaEventDestroy(visible_sync_ev_);
    }
    visible_sync_ev_ = nullptr;  // Never copy event handles
    visible_synced_ = false;     // Force re-sync on first use
    last_agg_ptr_ = nullptr;
    
    if (other.dev_temp_x_) {
      dev_temp_x_ = RPU::make_unique<CudaArray<T>>(*other.dev_temp_x_);
    }
    if (other.dev_temp_d_) {
      dev_temp_d_ = RPU::make_unique<CudaArray<T>>(*other.dev_temp_d_);
    }
    
    // Copy scratch buffers if they exist
    if (other.dev_xa_mb_) {
      dev_xa_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_xa_mb_);
    }
    if (other.dev_xb_mb_) {
      dev_xb_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_xb_mb_);
    }
    if (other.dev_db_mb_) {
      dev_db_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_db_mb_);
    }
    if (other.dev_da_mb_) {
      dev_da_mb_ = RPU::make_unique<CudaArray<T>>(*other.dev_da_mb_);
    }
    scratch_mb_capacity_ = other.scratch_mb_capacity_;
    
    // PWUs use lazy-init - do not copy
    fastA_pwu_.reset();
    fastB_pwu_.reset();
    visible_pwu_.reset();
    
    // Copy padded buffers if they exist
    if (other.dev_x_pad_) {
      dev_x_pad_ = RPU::make_unique<CudaArray<T>>(*other.dev_x_pad_);
    }
    if (other.dev_d_pad_) {
      dev_d_pad_ = RPU::make_unique<CudaArray<T>>(*other.dev_d_pad_);
    }
    
    initializeDevicePointers();
  }
  return *this;
}

template <typename T>
LRTTTransferRPUDeviceCuda<T>::LRTTTransferRPUDeviceCuda(
    LRTTTransferRPUDeviceCuda<T> &&other) noexcept
    : TransferRPUDeviceCuda<T>(std::move(other)),
      rank_(other.rank_),
      dev_w_visible_(other.dev_w_visible_),
      dev_w_a_(other.dev_w_a_),
      dev_w_b_(other.dev_w_b_),
      dev_temp_x_(std::move(other.dev_temp_x_)),
      dev_temp_d_(std::move(other.dev_temp_d_)),
      dev_xa_mb_(std::move(other.dev_xa_mb_)),
      dev_xb_mb_(std::move(other.dev_xb_mb_)),
      dev_db_mb_(std::move(other.dev_db_mb_)),
      dev_da_mb_(std::move(other.dev_da_mb_)),
      scratch_mb_capacity_(other.scratch_mb_capacity_),
      fastA_pwu_(std::move(other.fastA_pwu_)),
      fastB_pwu_(std::move(other.fastB_pwu_)),
      visible_pwu_(std::move(other.visible_pwu_)),
      dev_x_pad_(std::move(other.dev_x_pad_)),
      dev_d_pad_(std::move(other.dev_d_pad_)),
      transfer_counter_(other.transfer_counter_),
      num_a_updates_(other.num_a_updates_),
      num_b_updates_(other.num_b_updates_),
      num_transfers_(other.num_transfers_),
      debug_no_host_copies_(other.debug_no_host_copies_),
      dev_fb_out_(std::move(other.dev_fb_out_)),
      dev_y_ab_(std::move(other.dev_y_ab_)),
      visible_synced_(other.visible_synced_),
      last_agg_ptr_(other.last_agg_ptr_),
      visible_sync_ev_(other.visible_sync_ev_) {  // Transfer event handle
  
  other.dev_w_visible_ = nullptr;
  other.dev_w_a_ = nullptr;
  other.dev_w_b_ = nullptr;
  other.scratch_mb_capacity_ = 0;
  other.visible_sync_ev_ = nullptr;  // Clear moved-from event
}

template <typename T>
LRTTTransferRPUDeviceCuda<T> &
LRTTTransferRPUDeviceCuda<T>::operator=(LRTTTransferRPUDeviceCuda<T> &&other) noexcept {
  if (this != &other) {
    TransferRPUDeviceCuda<T>::operator=(std::move(other));
    rank_ = other.rank_;
    dev_w_visible_ = other.dev_w_visible_;
    dev_w_a_ = other.dev_w_a_;
    dev_w_b_ = other.dev_w_b_;
    dev_temp_x_ = std::move(other.dev_temp_x_);
    dev_temp_d_ = std::move(other.dev_temp_d_);
    dev_xa_mb_ = std::move(other.dev_xa_mb_);
    dev_xb_mb_ = std::move(other.dev_xb_mb_);
    dev_db_mb_ = std::move(other.dev_db_mb_);
    dev_da_mb_ = std::move(other.dev_da_mb_);
    scratch_mb_capacity_ = other.scratch_mb_capacity_;
    fastA_pwu_ = std::move(other.fastA_pwu_);
    fastB_pwu_ = std::move(other.fastB_pwu_);
    visible_pwu_ = std::move(other.visible_pwu_);
    dev_x_pad_ = std::move(other.dev_x_pad_);
    dev_d_pad_ = std::move(other.dev_d_pad_);
    transfer_counter_ = other.transfer_counter_;
    num_a_updates_ = other.num_a_updates_;
    num_b_updates_ = other.num_b_updates_;
    num_transfers_ = other.num_transfers_;
    debug_no_host_copies_ = other.debug_no_host_copies_;
    dev_fb_out_ = std::move(other.dev_fb_out_);
    dev_y_ab_ = std::move(other.dev_y_ab_);
    
    // Destroy existing event if any, then transfer the handle
    if (visible_sync_ev_) {
      cudaEventDestroy(visible_sync_ev_);
    }
    visible_sync_ev_ = other.visible_sync_ev_;
    visible_synced_ = other.visible_synced_;
    last_agg_ptr_ = other.last_agg_ptr_;
    
    other.dev_w_visible_ = nullptr;
    other.dev_w_a_ = nullptr;
    other.dev_w_b_ = nullptr;
    other.scratch_mb_capacity_ = 0;
    other.visible_sync_ev_ = nullptr;  // Clear moved-from event
  }
  return *this;
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device) {
  // Call parent populate
  TransferRPUDeviceCuda<T>::populateFrom(rpu_device);
  
  // Cast to our type
  const auto &cpu_device = static_cast<const LRTTTransferRPUDevice<T> &>(rpu_device);
  
  // Parameters are stored in base class
  
  // Extract rank from CPU device
  rank_ = cpu_device.getRank();
  
  // Ensure parent read-based transfer is disabled for LRTT
  const auto &lrtt_par = getPar();
  if (lrtt_par.n_reads_per_transfer > 0) {
    RPU_FATAL("LRTT: n_reads_per_transfer must be 0 for LRTT devices. Parent read-based transfer causes double accumulation.");
  }
  
  // Initialize device pointers
  initializeDevicePointers();
  
  // Fix A: Reset sync state
  visible_synced_ = false;
  last_agg_ptr_ = nullptr;
  
  // Re-initialize PWUs after population if needed
  if (rank_ > 0 && this->rpucuda_device_vec_.size() >= 3) {
    const auto& par = getPar();
    
    // Check if we need to create PWUs
    if (!fastA_pwu_ && par.idx_fastA < this->rpucuda_device_vec_.size() && 
        this->rpucuda_device_vec_[par.idx_fastA]) {
      auto* devA = dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(
          this->rpucuda_device_vec_[par.idx_fastA].get());
      if (devA) {
        fastA_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
        if (getenv("AIHWKIT_DEBUG_LRTT")) {
          printf("[LR-TT CUDA] populateFrom: Created PWU for fastA\n");
        }
      }
    }
    
    if (!fastB_pwu_ && par.idx_fastB < this->rpucuda_device_vec_.size() && 
        this->rpucuda_device_vec_[par.idx_fastB]) {
      auto* devB = dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(
          this->rpucuda_device_vec_[par.idx_fastB].get());
      if (devB) {
        fastB_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
        if (getenv("AIHWKIT_DEBUG_LRTT")) {
          printf("[LR-TT CUDA] populateFrom: Created PWU for fastB\n");
        }
      }
    }
  }
  
  // Validate transfer cadence for LR-TT
  const auto& par = getPar();
  if (par.update_rule == LRUpdateRule::LR_TT) {
    if (par.transfer_every <= 0) {
      RPU_WARNING("LR-TT: transfer_every <= 0 disables periodic transfers. "
                  "This is acceptable for inference-only scenarios but breaks the "
                  "'always pulsed' guarantee during training. Current value: " << par.transfer_every);
    }
    if (par.n_reads_per_transfer > 0) {
      RPU_FATAL("LR-TT requires n_reads_per_transfer == 0 (parent read-based transfer is disabled). "
                "Current value: " << par.n_reads_per_transfer);
    }
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT CUDA] Transfer validation: transfer_every=%d, n_reads_per_transfer=%d\n",
             par.transfer_every, par.n_reads_per_transfer);
    }
  }
  
  // Reset all counters
  transfer_counter_ = 0;
  num_a_updates_ = 0;
  num_b_updates_ = 0;
  num_transfers_ = 0;
  debug_no_host_copies_ = 0;
  
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT CUDA] populateFrom complete: PWUs: A=%s, B=%s\n",
           fastA_pwu_ ? "YES" : "NO", fastB_pwu_ ? "YES" : "NO");
  }
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::initializeDevicePointers() {
  // Get device weight pointers from the weight storage
  // The parent VectorRPUDeviceCuda stores all weights in dev_weights_vec_
  // Each device's weights are stored contiguously with size this->size_
  
  const auto& par = getPar();
  
  // Each device has this->size_ elements in the weight vector
  // int device_size = this->size_;  // Unused, commented out to avoid warnings
  
  // Use dev_weights_ptrs_ array for robustness
  if (!this->dev_weights_ptrs_.empty()) {
    // Get pointers from the pre-computed array
    if (par.idx_visible >= 0 && par.idx_visible < (int)this->dev_weights_ptrs_.size()) {
      dev_w_visible_ = this->dev_weights_ptrs_[par.idx_visible];
      // CRITICAL: Ensure the pointer array always reflects our current visible pointer
      // This is needed for reduceToWeights to work correctly after transfer
      this->dev_weights_ptrs_[par.idx_visible] = dev_w_visible_;
    }
    
    if (par.idx_fastA >= 0 && par.idx_fastA < (int)this->dev_weights_ptrs_.size()) {
      dev_w_a_ = this->dev_weights_ptrs_[par.idx_fastA];
    }
    
    if (par.idx_fastB >= 0 && par.idx_fastB < (int)this->dev_weights_ptrs_.size()) {
      dev_w_b_ = this->dev_weights_ptrs_[par.idx_fastB];
    }
  } else {
    // Weights not yet allocated - will be set later
    dev_w_visible_ = nullptr;
    dev_w_a_ = nullptr;
    dev_w_b_ = nullptr;
  }
}

template <typename T>
T* LRTTTransferRPUDeviceCuda<T>::getVisibleWeightsDevice() {
  return dev_w_visible_;
}

template <typename T>
T* LRTTTransferRPUDeviceCuda<T>::getFastAWeightsDevice() {
  return dev_w_a_;
}

template <typename T>
T* LRTTTransferRPUDeviceCuda<T>::getFastBWeightsDevice() {
  return dev_w_b_;
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::runUpdateKernel(
    pwukp_t<T> kpars,
    CudaContextPtr up_context,
    T *dev_weights,
    int m_batch,
    const BitLineMaker<T> *blm,
    const PulsedUpdateMetaParameter<T> &up,
    const T lr,
    curandState_t *dev_states,
    int one_sided,
    uint32_t *x_counts_chunk,
    uint32_t *d_counts_chunk,
    const ChoppedWeightOutput<T> *cwo) {
  
  // Validate preconditions for LR-TT operation
  const auto &par = this->getPar();
  if (par.update_rule != LRUpdateRule::LR_TT) {
    RPU_FATAL("LRTTTransferRPUDeviceCuda requires update_rule == LR_TT");
  }
  if (rank_ <= 0) {
    RPU_FATAL("LRTTTransferRPUDeviceCuda requires rank > 0");
  }
  
  // Handle fully hidden case FIRST (before sync)
  if (this->fully_hidden_) {
    this->dev_weights_ptrs_[par.idx_visible] = dev_weights;
    dev_w_visible_ = dev_weights;  // Ensure consistent buffer usage in transfer path
  }
  
  // Fix A: One-time sync of visible tile from aggregated layer weights
  // Must be done AFTER fully_hidden pointer update
  if (!visible_synced_ || last_agg_ptr_ != dev_weights) {
    cudaStream_t stream = up_context ? up_context->getStream() : this->context_->getStream();
    syncVisibleWithAggregated(dev_weights, stream);
    last_agg_ptr_ = dev_weights;
    visible_synced_ = true;
  }
  
  // Ensure device pointers are initialized (good safety check)
  if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
    initializeDevicePointers();
    if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
      RPU_FATAL("LRTT: device weight pointers not initialized in runUpdateKernel");
    }
  }
  
  // Ensure scratch buffers are allocated before use
  ensureABScratch_();
  ensurePaddedBuffers(m_batch);
  
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT DEBUG] runUpdateKernel: pulsed path, rank=%d, m_batch=%d\n", rank_, m_batch);
  }
  
  // Step-2: LR-TT always uses pulsed LoRA-style updates  
  {
    // LoRA pulsed path with projected signals
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT DEBUG] LoRA pulsed path (Step-2: always StochasticCompressed)\n");
    }
    
    // Get input pointers from BitLineMaker - LR-TT requires these for pulsed updates
    const T* x_in = blm ? blm->getXData() : nullptr;
    const T* d_in = blm ? blm->getDData() : nullptr;
    
    if (!x_in || !d_in) {
      // For StochasticCompressed mode, getXData/getDData may return null
      // Try the raw/analog accessors which should be available
      if (blm) {
        x_in = blm->getAnalogX();
        d_in = blm->getAnalogD();
      }
    }
    
    if (!x_in || !d_in) {
      RPU_FATAL("LR-TT: BitLineMaker must expose X/D inputs for StochasticCompressed pulsed updates. "
                "Ensure PulsedWeightUpdater calls setExposeRawInputs(true) on BitLineMaker.");
    }
    
    // Step-2: Enforce StochasticCompressed mode for all LR-TT updates
    auto up_local = up; // local copy for modification  
    enforceStochasticCompressed(up_local);
    
    // Apply gradient magnitude correction if enabled (for pulsed path)
    T lr_scaled = lr;
    if (par.correct_gradient_magnitudes && rank_ > 0) {
      // Scale learning rate by std::sqrt(rank) to correct for low-rank gradient magnitudes
      T correction = std::sqrt((T)rank_);
      lr_scaled = lr * correction;
      if (getenv("AIHWKIT_DEBUG_LRTT")) {
        printf("[LR-TT Pulsed] Scaling LR by sqrt(%d) = %.3f: %.6e -> %.6e\n", 
               rank_, (float)correction, (float)lr, (float)lr_scaled);
      }
    }
    
    // Use the pulsed LoRA update with scaled LR
    doLoRAPulsedUpdate_(x_in, d_in, lr_scaled, m_batch, up_local, up_context);
    num_a_updates_++;
    num_b_updates_++;
    
    // Handle transfer scheduling (same as in direct path)
    int transfer_every = par.transfer_every;
    if (par.units_in_mbatch) {
      transfer_every *= m_batch;
    }
    
    if (transfer_every > 0 && (++transfer_counter_ >= transfer_every)) {
      transfer_counter_ = 0;  // Reset counter
      if (getenv("AIHWKIT_DEBUG_LRTT")) {
        printf("[LR-TT Pulsed] Transfer triggered (counter=%d, every=%d)\n",
               transfer_counter_, transfer_every);
      }
      doTransfer(up_context->getStream());
      num_transfers_++;  // Increment transfer counter
    }
    
    // Wait for any pending visible sync before reducing to aggregated weights
    cudaStream_t reduce_stream = up_context ? up_context->getStream() : this->context_->getStream();
    waitVisibleSyncOn(reduce_stream);
    
    // Reduce to aggregated weights
    this->reduceToWeights(up_context, dev_weights);
  }  // End of LR-TT pulsed path block
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::doTransfer(cudaStream_t stream) {
  // Re-validate pointers in case of late allocation/resize
  if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
    initializeDevicePointers();
    if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
      RPU_FATAL("LRTT: device weight pointers not initialized in doTransfer");
    }
  }
  
  // Debug output
  DEBUG_OUT("LRTT doTransfer called! transfer_lr=" << getPar().transfer_lr << ", rank=" << rank_);
  
  // Apply the outer product update
  const auto& par = getPar();
  applyABOuterAsPulsedUpdate(par.transfer_lr, stream);
  
  // Note: reinitFastTiles is now called inside applyABOuterAsPulsedUpdate
  // with proper stream ordering to avoid races
}


template <typename T>
void LRTTTransferRPUDeviceCuda<T>::applyABOuterAsPulsedUpdate(T lr_scale, cudaStream_t stream) {
  const int K = rank_;
  if (K <= 0 || lr_scale == (T)0) return;
  if (K > std::min(this->d_size_, this->x_size_)) {
    std::ostringstream ss;
    ss << "LR-TT transfer: rank (" << K << ") exceeds dims d=" << this->d_size_ << ", x=" << this->x_size_;
    RPU_FATAL(ss.str().c_str());
  }
  
  initializeDevicePointers(); // if needed
  ensureABScratch_(); // Ensure dev_temp_d_ and dev_temp_x_ are allocated

  cudaStream_t s = stream ? stream : this->context_->getStream();
  cudaStream_t ctx_s = this->context_->getStream();
  const bool cross_stream = (ctx_s != s);
  
  // Fix B: Create events for stream synchronization
  cudaEvent_t ev_pack_done = nullptr;
  cudaEvent_t ev_update_done = nullptr;
  if (cross_stream) {
    CUDA_CALL(cudaEventCreateWithFlags(&ev_pack_done, cudaEventDisableTiming));
    CUDA_CALL(cudaEventCreateWithFlags(&ev_update_done, cudaEventDisableTiming));
  }
  
  if (!visible_pwu_) visible_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);

  PulsedUpdateMetaParameter<T> up;
  const auto& par = getPar();
  up.desired_BL = (par.desired_BL > 0) ? par.desired_BL : 31;
  enforceStochasticCompressed(up);

  const int chunk = (par.rank_chunk > 0) ? par.rank_chunk : std::min(K, 256);
  const int threads = 256;

  auto* devVis = (par.idx_visible < this->rpucuda_device_vec_.size())
    ? dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(this->rpucuda_device_vec_[par.idx_visible].get()) : nullptr;
  if (!devVis) {
    std::ostringstream ss;
    ss << "LR-TT transfer requires pulsed visible tile (idx_visible=" << par.idx_visible << ").";
    RPU_FATAL(ss.str().c_str());
  }
  
  // CRITICAL: The visible device might have restrictive dw_min that prevents updates
  // We need to ensure the device allows updates with the transfer learning rate
  // Check if devVis has appropriate parameters for transfer

  // PWU computes: W += -lr * D @ X^T
  // We want: W += transfer_lr * A @ B
  // We pass D = A_chunk, X = B_chunk^T
  // So PWU computes: W += -lr * A_chunk @ B_chunk
  // To get W += transfer_lr * A_chunk @ B_chunk, we need lr = -transfer_lr
  const T lr_eff = -lr_scale;

  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT] applyABOuterAsPulsedUpdate: K=%d, chunk=%d, lr_scale=%.6e, lr_eff=%.6e\n", 
           K, chunk, (float)lr_scale, (float)lr_eff);
  }

  for (int off = 0; off < K; off += chunk) {
    const int cur = std::min(chunk, K - off);

    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT]   chunk [%d:%d) (cur=%d) A[:,%d:%d] @ B[%d:%d,:]\n", 
             off, off+cur, cur, off, off+cur, off, off+cur);
    }

    // D_chunk = A[:, off:off+cur]  -> dev_temp_d_ [d × cur]
    const int nD = this->d_size_ * cur;
    kernelPackColsWithOffset<<<(nD + threads - 1)/threads, threads, 0, s>>>(
        dev_temp_d_->getData(), dev_w_a_, this->d_size_, this->x_size_, cur, off);

    // Pack B_chunk = B[off:off+cur, :] -> dev_temp_x_ [cur × x]
    const int nX = cur * this->x_size_;
    kernelPackRowsWithOffset<<<(nX + threads - 1)/threads, threads, 0, s>>>(
        dev_temp_x_->getData(), dev_w_b_, this->d_size_, this->x_size_, cur, off);

    // Ensure transpose buffer is allocated with sufficient capacity
    if (!dev_temp_x_T_ || dev_temp_x_T_->getSize() < this->x_size_ * chunk) {
      dev_temp_x_T_ = std::make_unique<CudaArray<T>>(this->context_, this->x_size_ * chunk);
    }

    // Transpose B_chunk from [cur × x] to B_chunk^T [x × cur] for PWU
    kernelTransposeKxXtoXxK<<<(nX + threads - 1)/threads, threads, 0, s>>>(
        dev_temp_x_T_->getData(),
        dev_temp_x_->getData(),
        cur, this->x_size_);

    // Fix B: Ensure PWU (on ctx_s) only starts after pack/transpose on 's' has completed
    if (cross_stream) {
      CUDA_CALL(cudaEventRecord(ev_pack_done, s));
      CUDA_CALL(cudaStreamWaitEvent(ctx_s, ev_pack_done, 0));
    }
    
    // Wait for any pending visible sync before updating visible weights
    waitVisibleSyncOn(ctx_s);

    // Call updater safely with x_trans=false, d_trans=false
    visible_pwu_->update(
        /*X=*/dev_temp_x_T_->getData(), // [x × cur] col-major (correct layout)
        /*D=*/dev_temp_d_->getData(),   // [d × cur] col-major (correct layout)
        dev_w_visible_, devVis, up, lr_eff, /*m_batch=*/cur,
        /*x_trans=*/false, /*d_trans=*/false);
    
    // Fix B: Reinit (on 's') must wait until the PWU update on ctx_s has finished
    if (cross_stream) {
      CUDA_CALL(cudaEventRecord(ev_update_done, ctx_s));
      CUDA_CALL(cudaStreamWaitEvent(s, ev_update_done, 0));
    }
  }

  reinitFastTiles(s);
  
  // Fix B: Clean up events
  if (cross_stream) {
    CUDA_CALL(cudaEventDestroy(ev_pack_done));
    CUDA_CALL(cudaEventDestroy(ev_update_done));
  }
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::reinitFastTiles(cudaStream_t stream) {
  const int K = rank_;
  if (K <= 0) return;

  cudaStream_t s = stream ? stream : this->context_->getStream();
  const int threads = 256;
  const int d = this->d_size_;
  const int x = this->x_size_;

  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT] reinitFastTiles (LoRA-convention mapping): K=%d\n", K);
    printf("       -> A_lr = 0 (first K cols of A),  B_lr ~ Kaiming (first K rows of B)\n");
  }

  // --- Zero A entirely (safe & simple) -> ensures A_lr == 0
  const int nA_full = d * x;
  const int blocks_A_full = (nA_full + threads - 1) / threads;
  kernelResetWeights<<<blocks_A_full, threads, 0, s>>>(dev_w_a_, nA_full);

  // --- Zero B entirely first (so rows beyond K stay clean)
  const int nB_full = d * x;
  const int blocks_B_full = (nB_full + threads - 1) / threads;
  kernelResetWeights<<<blocks_B_full, threads, 0, s>>>(dev_w_b_, nB_full);

  // --- Kaiming(He) init for B_lr = first K rows of B
  // Fan-in for a row-wise linear map B_lr (K×x) applied to X (x×mb) is 'x'.
  static unsigned long long reinit_counter = 0;
  reinit_counter++;
  unsigned long long seed =
      ((unsigned long long)(uintptr_t)dev_w_b_ ^ reinit_counter) +
      (unsigned long long)std::rand();

  const T std_dev_B = getPar().reinit_gain * std::sqrt((T)2.0 / (T)x);

  const int nB_lr = K * x;
  const int blocks_B_lr = (nB_lr + threads - 1) / threads;
  kernelKaimingInitFirstKRowsStateless<<<blocks_B_lr, threads, 0, s>>>(
      dev_w_b_, d, x, K, seed, std_dev_B);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::printDebugStats() const {
  printf("[LR-TT STATS] A_updates=%d, B_updates=%d, transfers=%d, rank=%d\n", 
         num_a_updates_, num_b_updates_, num_transfers_, rank_);
  
  // Guard against calling before first update pair
  if (num_a_updates_ > 0 || num_b_updates_ > 0) {
    if (num_a_updates_ != num_b_updates_) {
      printf("[LR-TT WARNING] A and B update counts differ! A=%d, B=%d. This indicates an issue.\n",
             num_a_updates_, num_b_updates_);
#ifdef AIHWKIT_DEBUG_LRTT
      RPU_FATAL("LR-TT: A and B update counts must be equal for proper gradient accumulation");
#endif
    }
  }
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::doDirectUpdate(
    const T *x_in,
    const T *d_in,
    T *dev_weights,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const T beta,
    const PulsedUpdateMetaParameter<T> &up,
    T *x_buffer,
    T *d_buffer) {
  
  // Handle fully hidden case FIRST (before sync)
  if (this->fully_hidden_) {
    const auto& par = this->getPar();
    this->dev_weights_ptrs_[par.idx_visible] = dev_weights;
    dev_w_visible_ = dev_weights;  // Ensure consistent buffer usage in transfer path
  }
  
  // Fix A: One-time sync of visible tile from aggregated layer weights
  // Must be done AFTER fully_hidden pointer update
  if (!visible_synced_ || last_agg_ptr_ != dev_weights) {
    syncVisibleWithAggregated(dev_weights, this->context_->getStream());
    last_agg_ptr_ = dev_weights;
    visible_synced_ = true;
  }
  
  // Ensure device pointers are initialized
  if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
    initializeDevicePointers();
    if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
      RPU_FATAL("LRTT: device weight pointers not initialized in doDirectUpdate");
    }
  }
  
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT DEBUG] update path: NONE->direct\n");
  }
  
  const auto& par = getPar();
  
  // Check if we should use LoRA-style updates
  if (par.update_rule == LRUpdateRule::LR_TT) {
    // LR-TT MUST use pulsed path to maintain "always pulsed" guarantee
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT DEBUG] Using pulsed LoRA updates (LR_TT mode - always pulsed)\n");
    }
    
    // Enforce StochasticCompressed for pulsed path
    auto up_local = up;
    enforceStochasticCompressed(up_local);
    
    // Use pulsed LoRA update - NOT the GEMM-based doLoRAUpdate
    doLoRAPulsedUpdate_(x_in, d_in, lr, m_batch, up_local, this->context_);
    num_a_updates_++;
    num_b_updates_++;
  } else {
    // Legacy behavior path (deprecated)
    // 1. Update fast device A directly
    if (this->rpucuda_device_vec_[par.idx_fastA]) {
      // Call the sub-device's doDirectUpdate directly
      this->rpucuda_device_vec_[par.idx_fastA]->doDirectUpdate(
          x_in, d_in, this->dev_weights_ptrs_[par.idx_fastA], lr, m_batch, 
          x_trans, d_trans, beta, up, x_buffer, d_buffer);
      num_a_updates_++;
      
      if (getenv("AIHWKIT_DEBUG_LRTT")) {
#ifndef RPU_USE_FP16
        cudaStream_t s = this->context_->getStream();
        T norm_a = 0;
        CudaArray<T> temp_norm(this->context_, 1);
        temp_norm.setConst(0.0);
        kernelFrobNormFirstK<<<(this->d_size_*rank_ + 255)/256, 256, 0, s>>>(
            dev_w_a_, this->d_size_, this->x_size_, rank_, temp_norm.getData(), true);
        temp_norm.copyTo(&norm_a);
        this->context_->synchronize();
        printf("[LR-TT DIRECT] A updated: ||A[:,:K]||_F = %.6e\n", (float)norm_a);
#else
        printf("[LR-TT DIRECT] A updated (FP16 - norm not computed)\n");
#endif
      }
    }
    
    // 2. Update fast device B directly (with transposed semantics - swap_x_d=true)
    if (this->rpucuda_device_vec_[par.idx_fastB]) {
      // For B, we swap the input vectors to get transpose semantics (equivalent to swap_x_d=true)
      this->rpucuda_device_vec_[par.idx_fastB]->doDirectUpdate(
          d_in, x_in, this->dev_weights_ptrs_[par.idx_fastB], lr, m_batch,
          d_trans, x_trans, beta, up, d_buffer, x_buffer);
      num_b_updates_++;
      
      if (getenv("AIHWKIT_DEBUG_LRTT")) {
#ifndef RPU_USE_FP16
        cudaStream_t s = this->context_->getStream();
        T norm_b = 0;
        CudaArray<T> temp_norm(this->context_, 1);
        temp_norm.setConst(0.0);
        kernelFrobNormFirstK<<<(rank_*this->x_size_ + 255)/256, 256, 0, s>>>(
            dev_w_b_, this->d_size_, this->x_size_, rank_, temp_norm.getData(), false);
        temp_norm.copyTo(&norm_b);
        this->context_->synchronize();
        printf("[LR-TT DIRECT] B updated: ||B[:K,:]||_F = %.6e\n", (float)norm_b);
#else
        printf("[LR-TT DIRECT] B updated (FP16 - norm not computed)\n");
#endif
      }
    }
  }
  
  // 3. Transfer scheduling using increment-then-compare
  int transfer_every = par.transfer_every;
  if (par.units_in_mbatch) {
    transfer_every *= m_batch;
  }
  
  if (transfer_every > 0 && (++transfer_counter_ >= transfer_every)) {
    transfer_counter_ = 0;  // Reset counter
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT DIRECT] Transfer triggered (counter=%d, every=%d)\n",
             transfer_counter_, transfer_every);
    }
    
    // For LR-TT, use pulsed transfer; for legacy path use GEMM fallback
    if (rank_ > 0 && par.transfer_lr != 0) {
      if (par.update_rule == LRUpdateRule::LR_TT) {
        // LR-TT: Use pulsed transfer to maintain "always pulsed" guarantee
        doTransfer(this->context_->getStream());
        num_transfers_++;
      } else {
        // Legacy path: use GEMM fallback for transfer
        const int threads = 256;
        cudaStream_t ctx_stream = this->context_->getStream();
        
        // Pack A columns and B rows on context stream (same as GEMM)
        const int d_k_size = this->d_size_ * rank_;
        const int blocks_a = (d_k_size + threads - 1) / threads;
        kernelPackColsWithOffset<<<blocks_a, threads, 0, ctx_stream>>>(
            dev_temp_d_->getData(), dev_w_a_,
            this->d_size_, this->x_size_, rank_, 0);
        
        const int k_x_size = rank_ * this->x_size_;
        const int blocks_b = (k_x_size + threads - 1) / threads;
        kernelPackRowsWithOffset<<<blocks_b, threads, 0, ctx_stream>>>(
            dev_temp_x_->getData(), dev_w_b_,
            this->d_size_, this->x_size_, rank_, 0);
        
        // GEMM fallback: W += transfer_lr * (A[:,:K] @ B[:K,:])
        RPU::math::gemm<T>(
            this->context_,
            /*trans_A=*/false,  // A_lr is [d_size, K] col-major
            /*trans_B=*/false,  // B_lr is [K, x_size] col-major
            this->d_size_,      // M
            this->x_size_,      // N
            rank_,              // K
            par.transfer_lr,    // alpha
            dev_temp_d_->getData(), // A
            this->d_size_,      // lda (col-major: A_lr is [d_size, K], so ld = d_size)
            dev_temp_x_->getData(), // B
            rank_,              // ldb (col-major: B_lr is [K, x_size], so ld = K)
            (T)1.0,             // beta
            dev_w_visible_,     // C
            this->d_size_);     // ldc (col-major: W is [d_size, x_size], so ld = d_size)
      
        // Reinitialize fast tiles on context stream (same as GEMM)
        reinitFastTiles(ctx_stream);
        num_transfers_++;
      
        if (getenv("AIHWKIT_DEBUG_LRTT")) {
          T norm_w = 0;
          CudaArray<T> temp_norm(this->context_, 1);
          temp_norm.setConst(0.0);
          dim3 blocks((this->d_size_*this->x_size_ + 255) / 256);
          kernelFrobNormFirstK<<<blocks, 256, 0, ctx_stream>>>(
              dev_w_visible_, this->d_size_, this->x_size_, 
              std::min(this->d_size_, this->x_size_), temp_norm.getData(), true);
          temp_norm.copyTo(&norm_w);
          this->context_->synchronize();
          printf("[LR-TT DIRECT] After transfer: ||W||_F = %.6e\n", (float)norm_w);
        }
      }  // end else (legacy path)
    }  // end if (rank_ > 0 && par.transfer_lr != 0)
  }
  
  // 4. Wait for any pending visible sync before reducing to aggregated weights
  waitVisibleSyncOn(this->context_->getStream());
  
  // 5. Reduce to aggregated weights
  this->reduceToWeights(this->context_, dev_weights);
}

template <typename T>
pwukpvec_t<T> LRTTTransferRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch,
    int nK32,
    int use_bo64,
    bool out_trans,
    const PulsedUpdateMetaParameter<T> &up) {
  
  // Use fast A device for kernel selection (not device 0)
  const auto& par = getPar();
  pwukpvec_t<T> v;
  
  // Calculate chunking for transfer scheduling
  int transfer_every = par.transfer_every;
  if (par.units_in_mbatch) {
    transfer_every *= m_batch;
  }
  
  int nchunks = (transfer_every > 0 && transfer_every < m_batch) ? 
                ((m_batch + transfer_every - 1) / transfer_every) : 1;
  int chunk_size = (m_batch + nchunks - 1) / nchunks;
  
  // Get kernels from fast A device
  v = this->rpucuda_device_vec_[par.idx_fastA]->getUpdateKernels(
      chunk_size, nK32, use_bo64, out_trans, up);
  
  if (nchunks > 1) {
    for (auto &kpars : v) {
      kpars->ensureChunk();
    }
  }
  
  // CWO not supported
  for (auto &kpars : v) {
    kpars->disableCWO();
  }
  
  return v;
}

// Explicit instantiations
// Scratch buffer management for LoRA projections

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::ensureScratchBuffers(int m_batch) {
  if (scratch_mb_capacity_ < m_batch || !dev_xa_mb_ || !dev_xb_mb_ || !dev_db_mb_ || !dev_da_mb_) {
    // Allocate or reallocate scratch buffers
    dev_xa_mb_ = std::make_unique<CudaArray<T>>(this->context_, rank_ * m_batch);
    dev_xb_mb_ = std::make_unique<CudaArray<T>>(this->context_, rank_ * m_batch);
    dev_db_mb_ = std::make_unique<CudaArray<T>>(this->context_, rank_ * m_batch);
    dev_da_mb_ = std::make_unique<CudaArray<T>>(this->context_, rank_ * m_batch);
    scratch_mb_capacity_ = m_batch;
    
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT] Allocated scratch buffers: rank=%d, m_batch=%d\n", rank_, m_batch);
    }
  }
}

// LoRA update implementation
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::doLoRAUpdate(
    const T *x_in,
    const T *d_in,
    T *dev_weights,
    const T lr,
    const int m_batch,
    cudaStream_t stream,
    const T beta) {
  
  const auto& par = getPar();
  
  // Edge case: skip if rank is 0 or lr is 0
  if (rank_ <= 0 || lr == (T)0.0) {
    if (getenv("AIHWKIT_DEBUG_LRTT")) {
      printf("[LR-TT LoRA] Skipping update: rank=%d, lr=%.6e\n", rank_, (float)lr);
    }
    return;
  }
  
  // Validate rank
  if (rank_ > std::min(this->d_size_, this->x_size_)) {
    RPU_FATAL("LoRA rank " << rank_ << " exceeds matrix dimensions");
  }
  
  if (!stream) {
    stream = this->context_->getStream();
  }
  
  // Ensure scratch buffers are allocated
  ensureScratchBuffers(m_batch);
  ensureABScratch_();
  
  if (rank_ <= 0) { 
    RPU_FATAL("LR-TT: rank must be > 0 before LoRA update."); 
  }
  
  // 1. Pack A_lr and B_lr submatrices
  // A_lr = A[:, :rank] -> [d_size, rank]
  const int threads = 256;
  const int d_k_size = this->d_size_ * rank_;
  const int blocks_a = (d_k_size + threads - 1) / threads;
  kernelPackColsWithOffset<<<blocks_a, threads, 0, stream>>>(
      dev_temp_d_->getData(), dev_w_a_,
      this->d_size_, this->x_size_, rank_, 0);
  
  // B_lr = B[:rank, :] -> [rank, x_size]
  const int k_x_size = rank_ * this->x_size_;
  const int blocks_b = (k_x_size + threads - 1) / threads;
  kernelPackRowsWithOffset<<<blocks_b, threads, 0, stream>>>(
      dev_temp_x_->getData(), dev_w_b_,
      this->d_size_, this->x_size_, rank_, 0);
  
  // 2. Compute LoRA projections
  // X_A = B_lr @ X -> [rank, m_batch]
  RPU::math::gemm<T>(
      this->context_,
      /*transA=*/false,  // B_lr is [rank, x_size] col-major
      /*transB=*/false,  // X is [x_size, m_batch] col-major
      rank_,             // M
      m_batch,           // N
      this->x_size_,     // K
      (T)1.0,            // alpha
      dev_temp_x_->getData(),  // A (B_lr)
      rank_,             // lda
      x_in,              // B (X)
      this->x_size_,     // ldb
      (T)0.0,            // beta
      dev_xa_mb_->getData(),   // C (X_A)
      rank_);            // ldc
  
  // D_B = A_lr^T @ D -> [rank, m_batch]
  RPU::math::gemm<T>(
      this->context_,
      /*transA=*/true,   // A_lr is [d_size, rank] col-major, need A_lr^T
      /*transB=*/false,  // D is [d_size, m_batch] col-major
      rank_,             // M
      m_batch,           // N
      this->d_size_,     // K
      (T)1.0,            // alpha
      dev_temp_d_->getData(),  // A (A_lr)
      this->d_size_,     // lda
      d_in,              // B (D)
      this->d_size_,     // ldb
      (T)0.0,            // beta
      dev_db_mb_->getData(),   // C (D_B)
      rank_);            // ldc
  
  // 3. Apply LoRA updates to A and B
  // ΔA_lr = D @ X_A^T -> [d_size, rank]
  RPU::math::gemm<T>(
      this->context_,
      /*transA=*/false,  // D is [d_size, m_batch] col-major
      /*transB=*/true,   // X_A is [rank, m_batch] col-major, need X_A^T
      this->d_size_,     // M
      rank_,             // N
      m_batch,           // K
      (T)1.0,            // alpha
      d_in,              // A (D)
      this->d_size_,     // lda
      dev_xa_mb_->getData(),   // B (X_A)
      rank_,             // ldb
      (T)0.0,            // beta
      dev_temp_d_->getData(),  // C (ΔA_lr)
      this->d_size_);    // ldc
  
  // A[:, :rank] = beta * A[:, :rank] - lr * ΔA_lr
  const T neg_lr = (T)(-lr);
  const int blocks_add_a = (d_k_size + threads - 1) / threads;
  kernelScaleAndAxpyCols<<<blocks_add_a, threads, 0, stream>>>(
      dev_w_a_, dev_temp_d_->getData(), 
      this->d_size_, rank_, /*beta=*/beta, /*neg_lr=*/neg_lr);
  
  // ΔB_lr = D_B @ X^T -> [rank, x_size]
  RPU::math::gemm<T>(
      this->context_,
      /*transA=*/false,  // D_B is [rank, m_batch] col-major
      /*transB=*/true,   // X is [x_size, m_batch] col-major, need X^T
      rank_,             // M
      this->x_size_,     // N
      m_batch,           // K
      (T)1.0,            // alpha
      dev_db_mb_->getData(),  // A (D_B)
      rank_,             // ldb
      x_in,              // B (X)
      this->x_size_,     // ldb
      (T)0.0,            // beta
      dev_temp_x_->getData(),  // C (ΔB_lr)
      rank_);            // ldc
  
  // B[:rank, :] = beta * B[:rank, :] - lr * ΔB_lr
  const int blocks_add_b = (k_x_size + threads - 1) / threads;
  kernelScaleAndAxpyRows<<<blocks_add_b, threads, 0, stream>>>(
      dev_w_b_, dev_temp_x_->getData(),
      this->d_size_, this->x_size_, rank_,
      /*beta=*/beta, /*neg_lr=*/neg_lr);
  
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    T norm_a = 0, norm_b = 0;
    CudaArray<T> temp_norm(this->context_, 1);
    
    temp_norm.setConst(0.0);
    kernelFrobNormFirstK<<<(d_k_size + 255)/256, 256, 0, this->context_->getStream()>>>(
        dev_w_a_, this->d_size_, this->x_size_, rank_, temp_norm.getData(), true);
    temp_norm.copyTo(&norm_a);
    
    temp_norm.setConst(0.0);
    kernelFrobNormFirstK<<<(k_x_size + 255)/256, 256, 0, this->context_->getStream()>>>(
        dev_w_b_, this->d_size_, this->x_size_, rank_, temp_norm.getData(), false);
    temp_norm.copyTo(&norm_b);
    
    this->context_->synchronize();
    printf("[LR-TT LoRA] After update: ||A[:,:K]||_F = %.6e, ||B[:K,:]||_F = %.6e\n", 
           (float)norm_a, (float)norm_b);
  }
}

// Step-2: Pulsed LoRA update implementation (always enabled, no more compile guards)
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::doLoRAPulsedUpdate_(
    const T *x_in,
    const T *d_in,
    const T lr,
    const int m_batch,
    const PulsedUpdateMetaParameter<T> &up_in,
    CudaContextPtr up_context) {

  if (rank_ <= 0) return;
  if (rank_ > std::min(this->d_size_, this->x_size_)) {
    std::ostringstream ss;
    ss << "LR-TT: rank (" << rank_ << ") exceeds matrix dims d=" << this->d_size_ << ", x=" << this->x_size_;
    RPU_FATAL(ss.str().c_str());
  }
  auto up = up_in;
  enforceStochasticCompressed(up);

  cudaStream_t s = up_context ? up_context->getStream() : this->context_->getStream();
  
  // Make sure all math/pwu ops run on the same stream as the pack/route kernels
  CudaStreamGuard guard(this->context_, s);
  
  const int K = rank_;
  const int threads = 256;

  // 1) Pack LR subspace once per step
  ensureABScratch_(); // dev_temp_d_ [d×K], dev_temp_x_ [K×x]
  kernelPackFirstKCols<<<(this->d_size_*K + threads-1)/threads, threads, 0, s>>>(
      dev_temp_d_->getData(), dev_w_a_, this->d_size_, this->x_size_, K);
  kernelPackFirstKRows<<<(K*this->x_size_ + threads-1)/threads, threads, 0, s>>>(
      dev_temp_x_->getData(), dev_w_b_, this->d_size_, this->x_size_, K);

  // 2) Compute projections: X_B = B_lr * X, D_A = A_lr^T * D
  ensureScratchBuffers(m_batch); // alloc rank×mb buffers
  RPU::math::gemm<T>(this->context_, false, false,
      K, m_batch, this->x_size_, (T)1.0,
      dev_temp_x_->getData(), K, x_in, this->x_size_,
      (T)0.0, dev_xb_mb_->getData(), K); // X_B

  RPU::math::gemm<T>(this->context_, true, false,
      K, m_batch, this->d_size_, (T)1.0,
      dev_temp_d_->getData(), this->d_size_, d_in, this->d_size_,
      (T)0.0, dev_da_mb_->getData(), K); // D_A

  // 3) Route to LR rows via padded buffers (top-K rows)
  ensurePaddedBuffers(m_batch);
  dev_x_pad_->setConst((T)0.0);
  dev_d_pad_->setConst((T)0.0);

  // Put X_B into first K rows of x_pad, and D_A into first K rows of d_pad
  const int blocks_scatter_x = (K*m_batch + threads-1)/threads;
  kernelScatterRankRowsToPadded<<<blocks_scatter_x, threads, 0, s>>>(
      dev_x_pad_->getData(), this->x_size_,
      dev_xb_mb_->getData(), K,
      K, this->x_size_, m_batch);

  const int blocks_scatter_d = (K*m_batch + threads-1)/threads;
  kernelScatterRankRowsToPadded<<<blocks_scatter_d, threads, 0, s>>>(
      dev_d_pad_->getData(), this->d_size_,
      dev_da_mb_->getData(), K,
      K, this->d_size_, m_batch);

  // 4) Pulsed updates on fast tiles (A uses d_in vs X_B; B uses D_A vs x_in)
  if (!fastA_pwu_) fastA_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
  if (!fastB_pwu_) fastB_pwu_ = std::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);

  const auto& par = getPar();
  auto* devA = (par.idx_fastA < this->rpucuda_device_vec_.size())
    ? dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(this->rpucuda_device_vec_[par.idx_fastA].get()) : nullptr;
  auto* devB = (par.idx_fastB < this->rpucuda_device_vec_.size())
    ? dynamic_cast<PulsedRPUDeviceCudaBase<T>*>(this->rpucuda_device_vec_[par.idx_fastB].get()) : nullptr;

  if (devA) {
    fastA_pwu_->update(
        dev_x_pad_->getData(),  // X = padded X_B (rank rows)
        d_in,                   // D = original D
        dev_w_a_, devA, up, lr, m_batch, /*x_trans=*/false, /*d_trans=*/false);
  }
  if (devB) {
    fastB_pwu_->update(
        x_in,                   // X = original X
        dev_d_pad_->getData(),  // D = padded D_A (rank rows)
        dev_w_b_, devB, up, lr, m_batch, /*x_trans=*/false, /*d_trans=*/false);
  }
}

// Serialization support
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  // Call base class first
  TransferRPUDeviceCuda<T>::dumpExtra(extra, prefix);
  
  // Create state for LRTT-specific data
  RPU::state_t state;
  
  // Save LRTT-specific parameters
  RPU::insert(state, "rank", rank_);
  RPU::insert(state, "transfer_counter", transfer_counter_);
  RPU::insert(state, "num_a_updates", num_a_updates_);
  RPU::insert(state, "num_b_updates", num_b_updates_);
  RPU::insert(state, "num_transfers", num_transfers_);
  
  // Save meta parameters that affect forward pass
  const auto &par = getPar();
  RPU::insert(state, "forward_inject", par.forward_inject);
  RPU::insert(state, "lora_alpha", par.lora_alpha);
  RPU::insert(state, "idx_fastA", par.idx_fastA);
  RPU::insert(state, "idx_fastB", par.idx_fastB);
  RPU::insert(state, "idx_visible", par.idx_visible);
  RPU::insert(state, "rank_chunk", par.rank_chunk);
  RPU::insert(state, "rank_offset", par.rank_offset);
  
  // Insert with prefix
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {
  // Call base class first
  TransferRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);
  
  // Extract state with prefix
  auto state = RPU::selectWithPrefix(extra, prefix);
  
  // Load LRTT-specific parameters
  RPU::load(state, "rank", rank_, strict);
  RPU::load(state, "transfer_counter", transfer_counter_, strict);
  RPU::load(state, "num_a_updates", num_a_updates_, strict);
  RPU::load(state, "num_b_updates", num_b_updates_, strict);
  RPU::load(state, "num_transfers", num_transfers_, strict);
  
  // Load meta parameters (non-strict for backward compatibility)
  auto &par = getPar();
  RPU::load(state, "forward_inject", par.forward_inject, false);
  RPU::load(state, "lora_alpha", par.lora_alpha, false);
  RPU::load(state, "idx_fastA", par.idx_fastA, false);
  RPU::load(state, "idx_fastB", par.idx_fastB, false);
  RPU::load(state, "idx_visible", par.idx_visible, false);
  RPU::load(state, "rank_chunk", par.rank_chunk, false);
  RPU::load(state, "rank_offset", par.rank_offset, false);
  
  // Re-initialize device pointers after load to ensure consistency
  initializeDevicePointers();
}

// Template instantiations moved to end of file

// Forward-inject implementation: compose W_eff on device
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::composeForwardInject(
    T alpha, T* dev_out, cudaStream_t stream) {

  if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
    initializeDevicePointers();
    if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
      RPU_FATAL("LRTT: device weight pointers not initialized in composeForwardInject");
    }
  }

  // Force single stream to avoid races between pack kernels and GEMM
  cudaStream_t s = this->context_->getStream();
  
  // Wait for any pending visible sync from another stream
  waitVisibleSyncOn(s);
  
  // Use stream guard to ensure all operations (including GEMM) use the same stream
  CudaStreamGuard guard(this->context_, s);
  
  // Fix A: Ensure visible weights are synced on first forward
  if (!visible_synced_ && last_agg_ptr_) {
    syncVisibleWithAggregated(last_agg_ptr_, s);
    visible_synced_ = true;
  }
  const int d = this->d_size_;
  const int x = this->x_size_;
  const int K = rank_;
  
  // Apply LoRA alpha scaling (caller typically passes base alpha, we multiply by lora_alpha)
  const auto& par = getPar();
  T effective_alpha = alpha * par.lora_alpha;
  
  if (K <= 0) {
    // Just copy visible weights to out if rank==0
    CUDA_CALL(
      cudaMemcpyAsync(dev_out, dev_w_visible_, sizeof(T) * d * x,
                      cudaMemcpyDeviceToDevice, s));
    return;
  }

  // 1) Pack A_lr and B_lr into existing LR scratch buffers
  ensureABScratch_();
  const int threads = 256;

  // A_lr: [d, K]
  const int nA = d * K;
  kernelPackColsWithOffset<<<(nA + threads - 1) / threads, threads, 0, s>>>(
      dev_temp_d_->getData(), dev_w_a_, d, x, K, /*offset=*/0);

  // B_lr: [K, x]
  const int nB = K * x;
  kernelPackRowsWithOffset<<<(nB + threads - 1) / threads, threads, 0, s>>>(
      dev_temp_x_->getData(), dev_w_b_, d, x, K, /*offset=*/0);

  // 2) dev_out = W_visible
  CUDA_CALL(
    cudaMemcpyAsync(dev_out, dev_w_visible_, sizeof(T) * d * x,
                    cudaMemcpyDeviceToDevice, s));

  // 3) dev_out += alpha * (A_lr @ B_lr)
  // All matrices are column-major [d, x] as in the rest of the codebase.
  RPU::math::gemm<T>(
      this->context_,
      /*trans_A=*/false, /*trans_B=*/false, // [d,K] @ [K,x] -> [d,x]
      d, x, K,
      effective_alpha,
      dev_temp_d_->getData(), /*lda=*/d,
      dev_temp_x_->getData(), /*ldb=*/K,
      (T)1.0,
      dev_out, /*ldc=*/d);

  // Note: This does not modify W_visible; dev_out is the effective weight.
  
#ifdef AIHWKIT_DEBUG_LRTT
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    this->context_->synchronize();
    printf("[LR-TT] composeForwardInject: W_eff = W_visible + %.4f * (A[:,%d] @ B[%d,:]) (base_alpha=%.4f, lora_alpha=%.4f)\n", 
           (float)effective_alpha, K, K, (float)alpha, (float)par.lora_alpha);
  }
#endif
}

// Analog forward-inject implementation
template <typename T>
template <typename OutputIteratorT>
bool LRTTTransferRPUDeviceCuda<T>::forwardWithAnalogInject(
    OutputIteratorT out_values,
    InputOutputManager<T> &iom,
    ForwardBackwardPassIOManagedCuda<T> &fb,
    const MVParameterCuda<T> &mv_pars,
    const bool out_trans,
    const bool transposed) {

  const auto &par = getPar();

  // Fallback: if not enabled or invalid rank, use existing visible-only path
  if (!par.forward_inject || rank_ <= 0) {
    printf("[LR-TT DEBUG] Fallback path: forward_inject=%d, rank=%d, dev_w_visible_=%p\n",
           (int)par.forward_inject, rank_, dev_w_visible_);
    fb.computeAnalogMVSinglePassPublic(dev_w_visible_, iom, mv_pars, out_trans, transposed);
    return fb.finalizeOutputPublic(out_values, iom, mv_pars, out_trans, transposed);
  }

  // Current implementation requires non-transposed layouts; fall back if not satisfied
  if (out_trans || transposed) {
    RPU_WARNING(
        "LR-TT analog forward-inject currently requires "
        "out_trans==false and transposed==false; "
        "falling back to visible-only path.");
    fb.computeAnalogMVSinglePassPublic(dev_w_visible_, iom, mv_pars, out_trans, transposed);
    return fb.finalizeOutputPublic(out_values, iom, mv_pars, out_trans, transposed);
  }

  // Ensure device pointers are initialized
  if (!dev_w_visible_ || !dev_w_a_ || !dev_w_b_) {
    initializeDevicePointers();
  }
  
  // Fix A: Ensure visible weights are synced on first forward
  cudaStream_t s = this->context_->getStream();
  
  // Wait for any pending visible sync from another stream
  waitVisibleSyncOn(s);
  
  if (!visible_synced_ && last_agg_ptr_) {
    syncVisibleWithAggregated(last_agg_ptr_, s);
    visible_synced_ = true;
  }
  
  const int mb = iom.getMBatch();
  ensurePaddedBuffers(mb);   // provides dev_x_pad_
  ensureScratchBuffers(mb);  // provides dev_xb_mb_ (rank×mb)
  
  if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT] forwardWithAnalogInject: mb=%d, dev_w_visible_=%p, dev_w_a_=%p, dev_w_b_=%p\n",
           mb, dev_w_visible_, dev_w_a_, dev_w_b_);
  }

  const int out_elems = this->d_size_ * mb;
  if (!dev_fb_out_ || dev_fb_out_->getSize() < out_elems) {
    dev_fb_out_ = std::make_unique<CudaArray<T>>(this->context_, out_elems);
  }
  if (!dev_y_ab_ || dev_y_ab_->getSize() < out_elems) {
    dev_y_ab_ = std::make_unique<CudaArray<T>>(this->context_, out_elems);
  }

  // (1) Visible analog forward: y_vis into iom.getOutBuffer()
  // NOTE: We intentionally use the same mv_pars for W, B, and A so all three
  // tiles share the same IO calibration (noise, offsets, NL) in this design.
  // If per-tile MV params are introduced later, plumb them here accordingly.
  if (std::getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT] About to compute visible forward with dev_w_visible_=%p\n", dev_w_visible_);
    if (dev_w_visible_) {
      // Print first few weights to check if they're zero
      T weights[4];
      cudaMemcpy(weights, dev_w_visible_, sizeof(T) * 4, cudaMemcpyDeviceToHost);
      printf("[LR-TT] First 4 visible weights: %f, %f, %f, %f\n", 
             (float)weights[0], (float)weights[1], (float)weights[2], (float)weights[3]);
    }
  }
  fb.computeAnalogMVSinglePassPublic(dev_w_visible_, iom, mv_pars, out_trans, transposed);

  // (2) B analog forward: g_full = B*x → pack top-K rows to g_rank
  T *saved_out = iom.getOutBuffer();
  iom.setOutBuffer(dev_fb_out_->getData());
  fb.computeAnalogMVSinglePassPublic(dev_w_b_, iom, mv_pars, out_trans, transposed);
  iom.setOutBuffer(saved_out);

  // g_rank = first K rows of g_full → dev_xb_mb_ [rank, mb]
  {
    const int threads = 256;
    const int blocks = (rank_ * mb + threads - 1) / threads;
    kernelPackFirstKRows<<<blocks, threads, 0, this->context_->getStream()>>>(
        dev_xb_mb_->getData(),        // out  [rank, mb]
        dev_fb_out_->getData(),       // in   [d_size, mb]
        this->d_size_, mb, rank_);
  }

  // Zero-pad: x_pad [x_size, mb], put g_rank into first K rows
  dev_x_pad_->setConst((T)0.0);
  {
    const int threads = 256;
    const int blocks = (rank_ * mb + threads - 1) / threads;
    kernelScatterRankRowsToPadded<<<blocks, threads, 0, this->context_->getStream()>>>(
        dev_x_pad_->getData(), this->x_size_,  // dst ld
        dev_xb_mb_->getData(), rank_,          // src ld
        rank_, this->x_size_, mb);
  }

  // (3) A analog forward: y_ab = A * x_pad
  T *saved_in = iom.getInBuffer();
  saved_out   = iom.getOutBuffer();

  iom.setInBuffer(dev_x_pad_->getData());
  iom.setOutBuffer(dev_y_ab_->getData());
  fb.computeAnalogMVSinglePassPublic(dev_w_a_, iom, mv_pars, out_trans, transposed);

  // Restore IOM buffers
  iom.setInBuffer(saved_in);
  iom.setOutBuffer(saved_out);

  // Accumulate: y_total += α * y_ab  (on current out buffer)
  {
    int threads = 256;
    int blocks  = (out_elems + threads - 1) / threads;
    kernelAxpy1D<<<blocks, threads, 0, this->context_->getStream()>>>(
        iom.getOutBuffer(),            // y
        dev_y_ab_->getData(),          // x
        out_elems,
        (T)par.lora_alpha);
  }

#ifdef AIHWKIT_DEBUG_LRTT
  if (getenv("AIHWKIT_DEBUG_LRTT")) {
    printf("[LR-TT FWD-INJECT] rank=%d, alpha=%.4f\n", rank_, (float)par.lora_alpha);
  }
#endif

  // (4) Finalize ONCE after sum (OtoO output noise, nonlinearities, scaling, etc.)
  return fb.finalizeOutputPublic(out_values, iom, mv_pars, out_trans, transposed);
}

// Template instantiations for forwardWithAnalogInject
#define OARG(NUM_T) , InputOutputManager<NUM_T>&, ForwardBackwardPassIOManagedCuda<NUM_T>&, const MVParameterCuda<NUM_T>&, const bool, const bool

RPU_GEN_OITER_TEMPLATES(
    float, bool, LRTTTransferRPUDeviceCuda<float>::forwardWithAnalogInject, OARG(float));
#ifdef RPU_USE_DOUBLE
RPU_GEN_OITER_TEMPLATES(
    double, bool, LRTTTransferRPUDeviceCuda<double>::forwardWithAnalogInject, OARG(double));
#endif
// NO FP16 forward instantiation for LR-TT (as specified in Step-2)

#undef OARG

// Sub-tile access method implementations
template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyVisibleWeightsTo(T* dst, cudaStream_t stream) const {
  if (!dev_w_visible_) {
    const_cast<LRTTTransferRPUDeviceCuda<T>*>(this)->initializeDevicePointers();
  }
  const int d = this->d_size_;
  const int x = this->x_size_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  RPU::math::copy<T>(this->context_, d * x, dev_w_visible_, 1, dst, 1);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyALRTo(T* dst, cudaStream_t stream) const {
  if (!dev_w_a_) {
    const_cast<LRTTTransferRPUDeviceCuda<T>*>(this)->initializeDevicePointers();
  }
  const int d = this->d_size_;
  const int r = rank_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  
  // Extract first r columns from dev_w_a_ (which is [d, x_size])
  // We need to pack the first r columns into a compact [d, r] matrix
  const int threads = 256;
  const int blocks = (d * r + threads - 1) / threads;
  kernelPackFirstKCols<<<blocks, threads, 0, s>>>(dst, dev_w_a_, d, this->x_size_, r);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyBLRTo(T* dst, cudaStream_t stream) const {
  if (!dev_w_b_) {
    const_cast<LRTTTransferRPUDeviceCuda<T>*>(this)->initializeDevicePointers();
  }
  const int r = rank_;
  const int x = this->x_size_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  
  // Extract first r rows from dev_w_b_ (which is [d_size, x])
  // We need to pack the first r rows into a compact [r, x] matrix
  const int threads = 256;
  const int blocks = (r * x + threads - 1) / threads;
  kernelPackFirstKRows<<<blocks, threads, 0, s>>>(dst, dev_w_b_, this->d_size_, x, r);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyVisibleWeightsFrom(const T* src, cudaStream_t stream) {
  if (!dev_w_visible_) {
    initializeDevicePointers();
  }
  const int d = this->d_size_;
  const int x = this->x_size_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  RPU::math::copy<T>(this->context_, d * x, src, 1, dev_w_visible_, 1);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyALRFrom(const T* src, cudaStream_t stream) {
  if (!dev_w_a_) {
    initializeDevicePointers();
  }
  const int d = this->d_size_;
  const int r = rank_;
  const int x = this->x_size_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  
  // Zero the full matrix first
  const int threads = 256;
  const int blocks_zero = (d * x + threads - 1) / threads;
  kernelResetWeights<<<blocks_zero, threads, 0, s>>>(dev_w_a_, d * x);
  
  // Unpack compact [d, r] into first r columns of [d, x_size]
  const int blocks = (d * r + threads - 1) / threads;
  kernelUnpackToFirstKCols<<<blocks, threads, 0, s>>>(dev_w_a_, src, d, x, r);
}

template <typename T>
void LRTTTransferRPUDeviceCuda<T>::copyBLRFrom(const T* src, cudaStream_t stream) {
  if (!dev_w_b_) {
    initializeDevicePointers();
  }
  const int d = this->d_size_;
  const int r = rank_;
  const int x = this->x_size_;
  cudaStream_t s = stream ? stream : this->context_->getStream();
  
  // Zero the full matrix first
  const int threads = 256;
  const int blocks_zero = (d * x + threads - 1) / threads;
  kernelResetWeights<<<blocks_zero, threads, 0, s>>>(dev_w_b_, d * x);
  
  // Unpack compact [r, x] into first r rows of [d_size, x]
  const int blocks = (r * x + threads - 1) / threads;
  kernelUnpackToFirstKRows<<<blocks, threads, 0, s>>>(dev_w_b_, src, d, x, r);
}

// Explicit template instantiations - MUST be at the end after all member definitions
template class LRTTTransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class LRTTTransferRPUDeviceCuda<double>;
#endif
// NO FP16 instantiation for LR-TT

} // namespace RPU