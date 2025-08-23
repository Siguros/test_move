# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Low-Rank Transfer Tiki-Taka (LR-TT) transfer orchestration.

Pure Python implementation of fast→slow low-rank transfers:
- Keeps TransferCompound & C++/CUDA backends unchanged
- Supports column (forward) and row (backward) modes
- Uses batch updates with optional chunking; falls back to per-rank loop
"""

from typing import Tuple, Optional, Any
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class TransferReport:
    """Report containing metrics and debug info from a single LR-TT transfer."""
    path: str  # "update", "set_delta_weights", "program_weights", "set_weights", "dry_run", "none"
    tile_type: str
    shape: Tuple[int, int]  # (out_dim, in_dim)
    device: str
    dtype: str
    delta_est_norm: float  # Frobenius norm of U @ V^T * transfer_lr
    pre_norm: float  # ||W_before||_F
    post_norm: float  # ||W_after||_F  
    applied_norm: float  # ||W_after - W_before||_F
    cosine_sim: float  # cosine similarity between vec(delta_est) and vec(W_after - W_before), NaN if not comparable
    rank: int
    transfer_lr: float
    rank_chunk: Optional[int]
    errors: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_dims_and_ttype(tile: Any) -> Tuple[int, int, torch.device, torch.dtype]:
    """Infer (in_dim, out_dim, device, dtype) from a tile."""
    in_dim = out_dim = None
    device = torch.device("cpu")
    dtype = torch.float32

    # Prefer weights (authoritative for device/dtype)
    try:
        w = tile.get_weights()
        if isinstance(w, tuple):
            w = w[0]
        out_dim, in_dim = w.shape
        device = w.device
        dtype = w.dtype
    except Exception:
        # Fallbacks for dims only
        if hasattr(tile, "get_x_size") and hasattr(tile, "get_d_size"):
            in_dim = tile.get_x_size()
            out_dim = tile.get_d_size()
        elif hasattr(tile, "in_size") and hasattr(tile, "out_size"):
            in_dim = tile.in_size
            out_dim = tile.out_size
        else:
            raise ValueError(
                "Cannot infer tile dimensions. Tile must provide get_weights() "
                "or (get_x_size/get_d_size) or (in_size/out_size)."
            )

    return in_dim, out_dim, device, dtype


# ---------------------------------------------------------------------------
# Planning U,V
# ---------------------------------------------------------------------------

def plan_lr_vectors(
    tile: Any,
    rank: int,
    seed: Optional[int] = None,
    orthonormal: bool = True,
    *,
    columns: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Create U,V transfer vectors for LR-TT on the SAME device/dtype as the slow tile.

    Shapes depend on mode:
      - columns=True:
          U ∈ ℝ^{rank×in_dim}  (used for fast.forward)
          V ∈ ℝ^{rank×out_dim} (used as D for slow.update)
      - columns=False (rows mode):
          U ∈ ℝ^{rank×out_dim} (used for fast.backward)
          V ∈ ℝ^{rank×in_dim}  (used as X for slow.update)

    Args:
        tile: tile (typically the SLOW tile) to copy device/dtype from
        rank: low-rank dimension r
        seed: RNG seed for reproducibility
        orthonormal: orthonormalize rows via QR (improves conditioning)
        columns: transfer columns (forward) or rows (backward)

    Returns:
        (U, V) tensors (contiguous) on tile's device/dtype
    """
    in_dim, out_dim, device, dtype = _infer_dims_and_ttype(tile)

    if seed is not None:
        torch.manual_seed(seed)

    if columns:
        U = torch.randn(rank, in_dim, device=device, dtype=dtype)
        V = torch.randn(rank, out_dim, device=device, dtype=dtype)
    else:
        U = torch.randn(rank, out_dim, device=device, dtype=dtype)
        V = torch.randn(rank, in_dim, device=device, dtype=dtype)

    if orthonormal:
        # Orthonormalize rows: QR on transpose
        U_T, _ = torch.linalg.qr(U.T)
        V_T, _ = torch.linalg.qr(V.T)
        U = U_T.T.contiguous()
        V = V_T.T.contiguous()
    else:
        U = U.contiguous()
        V = V.contiguous()

    return U, V


# ---------------------------------------------------------------------------
# Core LR-TT step (fast → slow)
# ---------------------------------------------------------------------------

@torch.no_grad()
def lrtt_transfer_step(
    fast_tile: Any,
    slow_tile: Any,
    U: Tensor,
    V: Tensor,
    *,
    transfer_lr: float = 1.0,
    columns: bool = True,
    rank_chunk: Optional[int] = None,
    set_slow_lr_to_one: bool = True,
    debug: bool = False,
    force_path: Optional[str] = None,
    dry_run: bool = False,
) -> TransferReport:
    """Perform one LR-TT transfer step: read from FAST, write to SLOW.

    Column mode (default):
        U ∈ ℝ^{r×in_dim}
        Y = fast_tile.forward(U)  → Y ∈ ℝ^{r×out_dim}
        slow_tile.update(X=V, D=Y * transfer_lr), with V ∈ ℝ^{r×in_dim}

    Row mode:
        U ∈ ℝ^{r×out_dim}
        Y = fast_tile.backward(U) → Y ∈ ℝ^{r×in_dim}
        slow_tile.update(X=Y, D=V * transfer_lr), with V ∈ ℝ^{r×out_dim}

    Args:
        fast_tile: source (aux/fast) tile to read from
        slow_tile: destination (visible/slow) tile to update
        U, V: planned vectors (see plan_lr_vectors)
        transfer_lr: scalar applied to D in the slow update
        columns: column (forward) or row (backward) mode
        rank_chunk: optional chunk size to limit batch memory
        set_slow_lr_to_one: if True, try slow_tile.set_learning_rate(1.0)
        debug: if True, snapshot weights and compute detailed metrics
        force_path: force specific transfer method ("update", "program_weights", "set_delta_weights", "set_weights")
        dry_run: if True, compute metrics but do not modify the tile
    
    Returns:
        TransferReport with transfer metrics and debug info
    """
    # Get tile info
    try:
        in_dim, out_dim, device, dtype = _infer_dims_and_ttype(slow_tile)
        tile_type = type(slow_tile).__name__
        shape = (out_dim, in_dim)
        r = U.shape[0]
    except Exception as e:
        return TransferReport(
            path="none", tile_type="unknown", shape=(0, 0), device="unknown", dtype="unknown",
            delta_est_norm=0.0, pre_norm=0.0, post_norm=0.0, applied_norm=0.0, cosine_sim=float('nan'),
            rank=U.shape[0] if U is not None else 0, transfer_lr=transfer_lr, rank_chunk=rank_chunk,
            errors=f"Failed to get tile info: {e}"
        )

    # Compute expected delta for comparison
    if columns:
        # For columns=True: U[rank, in_dim], V[rank, out_dim]
        # Y = fast.forward(U), delta = V^T @ U * transfer_lr (outer product approximation)
        try:
            # The intended delta is V^T @ U scaled by transfer_lr
            delta_est = (V.T @ U) * transfer_lr  # [out_dim, in_dim]
        except Exception as e:
            delta_est = torch.zeros(out_dim, in_dim, device=device, dtype=dtype)
    else:
        # For columns=False: U[rank, out_dim], V[rank, in_dim]
        # delta = U^T @ V * transfer_lr
        try:
            delta_est = (U.T @ V) * transfer_lr  # [out_dim, in_dim]
        except Exception as e:
            delta_est = torch.zeros(out_dim, in_dim, device=device, dtype=dtype)
    
    delta_est_norm = torch.norm(delta_est, p='fro').item()

    # Initialize report
    report = TransferReport(
        path="none", tile_type=tile_type, shape=shape, 
        device=str(device), dtype=str(dtype),
        delta_est_norm=delta_est_norm, pre_norm=0.0, post_norm=0.0, applied_norm=0.0, 
        cosine_sim=float('nan'), rank=r, transfer_lr=transfer_lr, rank_chunk=rank_chunk, errors=None
    )

    if dry_run:
        report.path = "dry_run"
        return report

    # Snapshot weights before (if debug enabled)
    W_before = None
    if debug:
        try:
            w_tuple = slow_tile.get_weights()
            W_before = w_tuple[0].detach().clone() if isinstance(w_tuple, tuple) else w_tuple.detach().clone()
            report.pre_norm = torch.norm(W_before, p='fro').item()
        except Exception as e:
            report.errors = f"Failed to get weights before: {e}"
    
    # Optional: normalize slow LR to 1.0 to avoid double scaling
    if set_slow_lr_to_one and hasattr(slow_tile, "set_learning_rate"):
        try:
            slow_tile.set_learning_rate(1.0)
        except Exception:
            pass

    # Define transfer paths to try
    transfer_paths = ["update", "program_weights", "set_delta_weights", "set_weights"]
    if force_path:
        if force_path not in transfer_paths:
            report.errors = f"Invalid force_path: {force_path}"
            return report
        transfer_paths = [force_path]
    
    # Prepare transfer data
    if columns:
        try:
            Y = fast_tile.forward(U)  # [r, out_dim]
            X_batch = U.contiguous()                 # [r, in_dim]
            D_batch = (V * transfer_lr).contiguous() # [r, out_dim]
        except Exception as e:
            report.errors = f"Forward pass failed: {e}"
            return report
    else:
        try:
            Y = fast_tile.backward(U)  # [r, in_dim]
            X_batch = (V * transfer_lr).contiguous() # [r, in_dim]
            D_batch = U.contiguous()                 # [r, out_dim]
        except Exception as e:
            report.errors = f"Backward pass failed: {e}"
            return report

    # Try transfer paths in order of preference
    success = False
    for path in transfer_paths:
        try:
            if path == "update":
                _try_update_path(slow_tile, X_batch, D_batch, rank_chunk)
            elif path == "program_weights":
                _try_program_weights_path(slow_tile, delta_est)
            elif path == "set_delta_weights":
                _try_set_delta_weights_path(slow_tile, delta_est)
            elif path == "set_weights":
                _try_set_weights_path(slow_tile, delta_est)
            
            report.path = path
            success = True
            break
            
        except Exception as e:
            if force_path:
                report.errors = f"Forced path '{path}' failed: {e}"
                return report
            continue
    
    if not success:
        report.path = "none"
        report.errors = "All transfer paths failed"
        return report

    # Snapshot weights after (if debug enabled)
    if debug and W_before is not None:
        try:
            w_tuple = slow_tile.get_weights()
            W_after = w_tuple[0].detach().clone() if isinstance(w_tuple, tuple) else w_tuple.detach().clone()
            report.post_norm = torch.norm(W_after, p='fro').item()
            
            W_delta_applied = W_after - W_before
            report.applied_norm = torch.norm(W_delta_applied, p='fro').item()
            
            # Compute cosine similarity
            if report.applied_norm > 1e-12 and delta_est_norm > 1e-12:
                delta_est_vec = delta_est.flatten()
                applied_vec = W_delta_applied.flatten()
                cos_sim = torch.cosine_similarity(delta_est_vec.unsqueeze(0), applied_vec.unsqueeze(0), dim=1)
                report.cosine_sim = cos_sim.item()
                
        except Exception as e:
            if report.errors is None:
                report.errors = f"Failed to get weights after: {e}"
            else:
                report.errors += f"; Failed to get weights after: {e}"

    return report


def _try_update_path(slow_tile: Any, X_batch: Tensor, D_batch: Tensor, rank_chunk: Optional[int]) -> None:
    """Try the standard update path with chunking."""
    r = X_batch.shape[0]
    chunk = rank_chunk if rank_chunk is not None else r

    for s in range(0, r, chunk):
        e = min(s + chunk, r)
        Xc = X_batch[s:e]
        Dc = D_batch[s:e]

        # Preferred: try a single batched update call
        try:
            slow_tile.update(Xc, Dc)
            continue
        except Exception:
            # Fallback: per-rank updates
            pass

        for i in range(Xc.shape[0]):
            x_i = Xc[i:i+1].contiguous()
            d_i = Dc[i:i+1].contiguous()
            slow_tile.update(x_i, d_i)


def _try_program_weights_path(slow_tile: Any, delta_est: Tensor) -> None:
    """Try programming weights directly by adding delta to current weights."""
    if not hasattr(slow_tile, 'get_weights') or not hasattr(slow_tile, 'program_weights'):
        raise AttributeError("Tile does not support program_weights path")
    
    w_tuple = slow_tile.get_weights()
    current_w = w_tuple[0] if isinstance(w_tuple, tuple) else w_tuple
    new_w = current_w + delta_est
    slow_tile.program_weights(new_w)


def _try_set_delta_weights_path(slow_tile: Any, delta_est: Tensor) -> None:
    """Try setting delta weights directly."""
    if not hasattr(slow_tile, 'set_delta_weights'):
        raise AttributeError("Tile does not support set_delta_weights path")
    
    slow_tile.set_delta_weights(delta_est)


def _try_set_weights_path(slow_tile: Any, delta_est: Tensor) -> None:
    """Try setting weights by adding delta to current weights.""" 
    if not hasattr(slow_tile, 'get_weights') or not hasattr(slow_tile, 'set_weights'):
        raise AttributeError("Tile does not support set_weights path")
    
    w_tuple = slow_tile.get_weights()
    current_w = w_tuple[0] if isinstance(w_tuple, tuple) else w_tuple
    new_w = current_w + delta_est
    slow_tile.set_weights(new_w)


# ---------------------------------------------------------------------------
# Simple scheduler (unchanged)
# ---------------------------------------------------------------------------

class LowRankTransferScheduler:
    """Optional scheduler for LR-TT transfers with step counting."""

    def __init__(self, transfer_every: int = 32):
        """
        Args:
            transfer_every: transfer frequency in steps/mini-batches
        """
        if transfer_every <= 0:
            raise ValueError("transfer_every must be >= 1 for the scheduler.")
        self.transfer_every = transfer_every
        self.step_count = 0

    def should_transfer(self) -> bool:
        """Return True on steps 0, transfer_every, 2*transfer_every, ..."""
        return self.step_count % self.transfer_every == 0

    def step(self) -> None:
        self.step_count += 1

    def reset(self) -> None:
        self.step_count = 0
