# LoRA Pulsed to AIHWKit Function Mapping Report

## Executive Summary
After analyzing the AIHWKit CUDA codebase, here's the mapping of lora_pulsed.cu functions to existing AIHWKit equivalents.

## 1. Matrix Operations (cuda_math_util.h)

### Available AIHWKit Functions:
- ✅ **`RPU::math::gemm()`** - General matrix multiply
- ✅ **`RPU::math::gemv()`** - Matrix-vector multiply  
- ✅ **`RPU::math::ger()`** - Rank-1 update (outer product)
- ✅ **`RPU::math::scal()`** - Scale vector/matrix
- ✅ **`RPU::math::copy()`** - Copy with stride
- ✅ **`RPU::math::elemcopy()`** - Element-wise copy with type conversion

### Required LoRA Operations → AIHWKit Mapping:

| LoRA Function | AIHWKit Equivalent | Notes |
|--------------|-------------------|-------|
| `computeXA()` (X @ A^T) | `RPU::math::gemm()` with TransB=true | Column-major compatible |
| `computeDB()` (D^T @ B) | `RPU::math::gemm()` with TransA=true | Column-major compatible |
| `applyOuterProduct()` | `RPU::math::ger()` | Perfect for rank-1 updates |
| Matrix scaling | `RPU::math::scal()` | For gradient correction |

## 2. Initialization Kernels

### Found in AIHWKit:
- ❌ **Kaiming initialization**: NOT FOUND - Need custom kernel
- ❌ **Zero initialization**: NOT FOUND - Can use `cudaMemset`
- ✅ **Random noise**: Found in `noise_manager.cu` but not suitable

### Current Implementation in lora_pulsed.cu:
```cpp
// Need to keep these custom kernels:
__global__ void kKaimingInitTopKRows() // Line 122
__global__ void kZeroInitTopKCols()    // Need to add
```

## 3. Packing/Transpose Operations

### AIHWKit Status:
- ❌ **Pack columns**: NOT FOUND - Need custom kernel
- ❌ **Extract rank region**: NOT FOUND - Need custom kernel  
- ❌ **Transpose**: No general transpose kernel found

### Required Custom Kernels:
```cpp
__global__ void kPackAColumns()    // Extract K columns from A
__global__ void kPackBRows()       // Extract K rows from B
__global__ void kTransposeMatrix() // For PWU preparation
```

## 4. BitLineMaker Integration

### Available in bit_line_maker.cu:
- ✅ **`getXData()`** - Line 1437 - Get raw X input
- ✅ **`getDData()`** - Line 1441 - Get raw D input
- ❌ **`setExposeRawInputs()`** - NOT FOUND (needs implementation)

### Integration Requirements:
```cpp
// Need to add to BitLineMaker:
void setExposeRawInputs(bool expose) {
  expose_raw_inputs_ = expose;
}
```

## 5. PulsedWeightUpdater (PWU) Integration

### Available in pulsed_weight_updater.cu:
- ✅ **PWU base class** - Fully compatible
- ✅ **Update kernels** - Via `PWUKernelParameter` system
- ✅ **StochasticCompressed** - Available pulse type

### Usage Pattern:
```cpp
// Already compatible:
auto pwu_a = std::make_unique<PulsedWeightUpdater<T>>(
  context, x_size, rank, par);
pwu_a->update(x_data, d_data, weights, ...);
```

## 6. Forward/Backward Pass Integration

### forward_backward_pass.cu Analysis:
- Uses `RPU::math::gemm()` extensively (lines 315, 379, 538, 658)
- No LoRA-specific injection found
- Position filtering via GEMM (line 547)

### Required Addition:
```cpp
// Need custom LoRA injection:
void forwardWithLoRAInject(y, x, w_full, w_a, w_b, alpha)
```

## 7. IO Manager Integration

### io_manager.cu Status:
- Input/output bound management kernels available
- No matrix packing/unpacking operations
- Focus on boundary conditions and scaling

## 8. Parameter System

### pwu_kernel_parameter.h:
- ✅ Macro system for defining kernels: `DEFINE_PWU_KERNEL_PARAMETER`
- ✅ Compatible with LoRA update pattern

## Integration Strategy

### Phase 1: Use AIHWKit Functions Where Possible
```cpp
// Replace custom GEMM with AIHWKit:
RPU::math::gemm<T>(context, false, true, m, k, n, 
                   1.0, X, m, A, k, 0.0, XA, m);

// Replace outer product:
RPU::math::ger<T>(context, d_size, x_size, lr,
                  d_vec, 1, x_vec, 1, W, d_size);
```

### Phase 2: Keep Custom Kernels for:
1. **Kaiming initialization** - Critical for LoRA
2. **Matrix packing/unpacking** - K-rank extraction
3. **Transposition** - For PWU column-major requirement
4. **LoRA forward injection** - Custom path

### Phase 3: BitLineMaker Modifications
```cpp
// Add to BitLineMaker class:
bool expose_raw_inputs_ = false;
void setExposeRawInputs(bool expose) { 
  expose_raw_inputs_ = expose; 
}
```

## Critical Compatibility Points

### ✅ CONFIRMED COMPATIBLE:
1. **GEMM operations** via cuda_math_util
2. **PWU integration** via PulsedWeightUpdater class
3. **BLM raw data access** via getXData()/getDData()
4. **Column-major convention** throughout

### ⚠️ REQUIRES CUSTOM IMPLEMENTATION:
1. **Kaiming initialization** with correct fan-in
2. **Rank-region extraction** (pack K cols/rows)
3. **Matrix transposition** for PWU
4. **BLM expose raw inputs flag**

## Recommended Approach

1. **Use AIHWKit math functions** for all GEMM/GER operations
2. **Keep custom kernels** for initialization and packing
3. **Extend BitLineMaker** with expose_raw_inputs flag  
4. **Use standard PWU** for weight updates
5. **Add LoRA injection** to forward pass when needed

## Files to Modify

1. **bit_line_maker.h/cu**: Add `setExposeRawInputs()` method
2. **rpucuda_lrtt_transfer_device_refactored.cu**: 
   - Replace GEMM calls with `RPU::math::gemm()`
   - Keep custom init/pack kernels
   - Use standard PWU interface

## Next Steps

1. Implement `setExposeRawInputs()` in BitLineMaker
2. Replace matrix operations with AIHWKit equivalents
3. Test gradient flow with standard PWU
4. Validate column-major indexing throughout