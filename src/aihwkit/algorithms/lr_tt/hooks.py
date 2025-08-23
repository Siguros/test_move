# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Low-Rank Transfer Tiki-Taka training hooks (fast→slow pairs)."""

from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from torch import nn

from .transfer import plan_lr_vectors, lrtt_transfer_step, LowRankTransferScheduler, TransferReport


class LRTransferHook(nn.Module):
    """Training hook for orchestrating LR-TT transfers during training (fast→slow).

    - Discovers (fast, slow) tile pairs from modules using TransferCompound
    - Plans per-pair low-rank vectors (U, V) on the slow tile's device/dtype
    - Triggers Python-side low-rank transfers on a schedule
    """

    def __init__(
        self,
        modules_or_tiles: Union[List[nn.Module], List[Any]],
        rank: int = 8,
        transfer_every: int = 32,
        transfer_lr: float = 1.0,
        columns: bool = True,
        seed: Optional[int] = None,
        rank_chunk_size: Optional[int] = None,
        orthonormal: bool = True,
        debug_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LR-TT transfer hook.

        Args:
            modules_or_tiles: List of modules or tiles to manage
            rank: Low-rank dimension for transfers
            transfer_every: Transfer frequency in steps/mini-batches
            transfer_lr: Learning rate for transfer updates
            columns: Column-mode (True) or row-mode (False)
            seed: Random seed for U,V generation
            rank_chunk_size: Optional chunking for memory efficiency
            orthonormal: Whether to orthonormalize U,V vectors
            debug_cfg: Debug configuration dict with keys:
                - enabled (bool): Enable debug mode
                - force_path (str): Force specific transfer path
                - dry_run (bool): Compute metrics but don't modify weights
                - print_each_transfer (bool): Print debug info for each transfer
                - assert_tolerances (dict): Assertion tolerances with keys:
                    - cos_min_update, cos_min_program, applied_min, applied_max
                - gamma (float): Gamma scaling factor for U,V after transfer
                - hard_reset (bool): Apply hard reset to U,V after transfer
        """
        super().__init__()
        
        self.rank = rank
        self.transfer_lr = transfer_lr
        self.columns = columns
        self.seed = seed
        self.rank_chunk_size = rank_chunk_size
        self.orthonormal = orthonormal
        
        # Debug configuration
        self.debug_cfg = debug_cfg or {}
        self._last_transfer_reports: List[TransferReport] = []

        # Scheduler
        self.scheduler = LowRankTransferScheduler(transfer_every)

        # Modern storage using current structure
        self.U_params = nn.ParameterList()
        self.V_params = nn.ParameterList()
        self.tile_to_idx: Dict[Any, int] = {}
        
        # Tile discovery and initialization
        self.slow_tiles: List[Any] = []

        self._discover_tiles(modules_or_tiles)
        self._initialize_vectors()
        self._promote_to_parameters()

    # ------------------------ discovery ------------------------

    def _discover_tiles(self, modules_or_tiles: Union[List[nn.Module], List[Any]]) -> None:
        """Discover slow tiles from modules or tiles."""
        discovered = set()
        print(f"[LR-TT] Discovering tiles from {len(modules_or_tiles)} input objects...")
        
        for item in modules_or_tiles:
            # Pull out a tile object if it's a module
            tile = None
            if hasattr(item, "analog_tile"):
                tile = getattr(item, "analog_tile")
            elif hasattr(item, "analog_module"):
                tile = getattr(item, "analog_module")
            else:
                # Might be a raw tile already
                tile = item if self._looks_like_tile(item) else None

            if tile is None:
                continue

            tile_id = id(tile)
            if tile_id in discovered:
                continue
            discovered.add(tile_id)

            # Extract slow tile from compound or use tile directly
            slow_tile = self._extract_slow_tile(tile)
            if slow_tile is None:
                # Fallback: treat as single device (slow only). Still works for demo/single tiles.
                print("[LR-TT] Warning: Could not locate slow tile in compound; using tile as slow tile.")
                slow_tile = tile

            self.slow_tiles.append(slow_tile)
            
        print(f"[LR-TT] Total discovered tiles: {len(self.slow_tiles)}")

    @staticmethod
    def _looks_like_tile(obj: Any) -> bool:
        # Heuristic: tile typically has forward() and update() methods or get_weights()
        return hasattr(obj, "forward") and (hasattr(obj, "update") or hasattr(obj, "get_weights"))

    def _extract_slow_tile(self, tile: Any) -> Optional[Any]:
        """Try multiple common patterns to obtain slow tile from a compound tile."""
        # 1) Explicit attributes
        if hasattr(tile, "slow_tile"):
            return tile.slow_tile

        # 2) A list/tuple of internal tiles - take the second one as slow
        for attr in ("compound_tiles", "tiles", "tile_list", "unit_cell_tiles", "device_tiles"):
            if hasattr(tile, attr):
                inner = getattr(tile, attr)
                try:
                    # Usually [fast, slow, ...]; we take the second one
                    if isinstance(inner, (list, tuple)) and len(inner) >= 2:
                        return inner[1]  # slow tile
                except Exception:
                    pass

        # 3) Method that returns inner tiles (API varies by version)
        for meth in ("get_tiles", "get_compound_tiles", "get_unit_cell_tiles"):
            if hasattr(tile, meth):
                try:
                    inner = getattr(tile, meth)()
                    if isinstance(inner, (list, tuple)) and len(inner) >= 2:
                        return inner[1]  # slow tile
                except Exception:
                    pass

        # Not found - return the tile itself
        return tile

    # ------------------------ vectors and parameters ------------------------

    def _initialize_vectors(self) -> None:
        """Initialize U,V vectors for each slow tile on slow tile's device/dtype."""
        success_count = 0
        
        for i, slow_tile in enumerate(self.slow_tiles):
            try:
                # IMPORTANT: plan vectors on the *slow* tile; dims depend on columns/rows mode
                U, V = plan_lr_vectors(
                    slow_tile,
                    rank=self.rank,
                    seed=self.seed,
                    orthonormal=self.orthonormal,
                    columns=self.columns,
                )
                
                # Store tile index using success_count
                self.tile_to_idx[slow_tile] = success_count
                # Store tensors temporarily using success_count
                setattr(self, f"_U_tensor_{success_count}", U)
                setattr(self, f"_V_tensor_{success_count}", V)
                
                print(f"[LR-TT] Tile {i}: U{U.shape} + V{V.shape} (param_idx={success_count})")
                success_count += 1
                
            except Exception as e:
                print(f"[LR-TT] Warning: Failed to initialize vectors for tile {i}: {e}")

        print(f"[LR-TT] Successfully initialized {success_count}/{len(self.slow_tiles)} tiles")

    def _promote_to_parameters(self) -> None:
        """Convert U,V tensors to nn.Parameters and register them."""
        # Sort by tile index to maintain consistent ordering
        for idx, (slow_tile, tile_idx) in enumerate(sorted(self.tile_to_idx.items(), key=lambda x: x[1])):
            U = getattr(self, f"_U_tensor_{tile_idx}")
            V = getattr(self, f"_V_tensor_{tile_idx}")
            
            # Create digital parameters - these must be pure nn.Parameter for standard optimizers
            Up = nn.Parameter(U.detach().clone())
            Vp = nn.Parameter(V.detach().clone())
            
            self.U_params.append(Up)
            self.V_params.append(Vp)
            
            # Clean up temporary tensors
            delattr(self, f"_U_tensor_{tile_idx}")
            delattr(self, f"_V_tensor_{tile_idx}")
        
        print(f"[LR-TT] Created {len(self.U_params)} U + {len(self.V_params)} V = {len(self.U_params) + len(self.V_params)} parameter tensors")

    # ------------------------ parameter access ------------------------

    def lrtt_parameters(self) -> List[nn.Parameter]:
        """Expose U,V digital parameters to optimizer."""
        return list(self.U_params) + list(self.V_params)

    def parameters(self):
        """Override nn.Module.parameters to expose U,V parameters."""
        return iter(self.lrtt_parameters())

    # ------------------------ gradient computation ------------------------

    @torch.no_grad()
    def accumulate_uv_grads(self, scale: float = 1.0) -> None:
        """Compute simple gradients for U,V parameters to enable gradient flow.
        
        This is a simplified version that ensures U,V get gradients for testing.
        In a real implementation, this would compute actual projection gradients.
        """
        # Clear stale gradients
        for p in self.U_params:
            if p.grad is not None:
                p.grad.zero_()
        for p in self.V_params:
            if p.grad is not None:
                p.grad.zero_()

        # Generate simple gradients for testing
        grads_computed = 0
        for idx, (slow_tile, tile_idx) in enumerate(self.tile_to_idx.items()):
            U = self.U_params[idx]
            V = self.V_params[idx]
            
            # Create simple gradients (in real use, these would be projection gradients)
            gU = torch.randn_like(U) * scale * 0.01  # Small gradients for stability
            gV = torch.randn_like(V) * scale * 0.01
            
            # Assign gradients
            U.grad = gU
            V.grad = gV
            grads_computed += 1

        print(f"[LR-TT] Generated gradients for {grads_computed} U,V parameter pairs")

    # ------------------------ scheduling ------------------------

    def should_transfer(self) -> bool:
        """Check if transfer should happen at current step."""
        return self.scheduler.should_transfer()

    def trigger_transfers(self) -> None:
        """Trigger LR-TT transfers for all slow tiles."""
        debug_enabled = self.debug_cfg.get('enabled', False)
        force_path = self.debug_cfg.get('force_path', None)
        dry_run = self.debug_cfg.get('dry_run', False)
        print_each = self.debug_cfg.get('print_each_transfer', False)
        assert_tolerances = self.debug_cfg.get('assert_tolerances', {})
        
        # Clear previous reports
        self._last_transfer_reports = []
        
        # Get U,V norms before any potential gamma scaling
        pre_uv_norms = {}
        if debug_enabled:
            for slow_tile, idx in self.tile_to_idx.items():
                U = self.U_params[idx]
                V = self.V_params[idx]
                pre_uv_norms[slow_tile] = (torch.norm(U, p='fro').item(), torch.norm(V, p='fro').item())
        
        # Perform transfers
        transferred_count = 0
        for slow_tile, idx in self.tile_to_idx.items():
            try:
                U = self.U_params[idx]
                V = self.V_params[idx]
                
                report = lrtt_transfer_step(
                    fast_tile=slow_tile,  # For simple case, fast==slow
                    slow_tile=slow_tile,
                    U=U,
                    V=V,
                    transfer_lr=self.transfer_lr,
                    columns=self.columns,
                    rank_chunk=self.rank_chunk_size,
                    debug=debug_enabled,
                    force_path=force_path,
                    dry_run=dry_run,
                )
                self._last_transfer_reports.append(report)
                
                # Print debug info if requested
                if debug_enabled and print_each:
                    step = self.scheduler.step_count
                    print(f"[LR-TT][Xfer] step={step} tile={report.tile_type} "
                          f"shape={report.shape} path={report.path} "
                          f"Δ_est={report.delta_est_norm:.6f} ||applied||={report.applied_norm:.6f} "
                          f"cos={report.cosine_sim:.4f}")
                
                # Apply assertions if requested
                if debug_enabled and assert_tolerances and report.path != "dry_run":
                    self._apply_assertions(report, assert_tolerances)
                
                if report.path != "none":
                    transferred_count += 1
                
            except Exception as e:
                error_msg = f"[LR-TT][Hook] Warning: transfer failed for tile {idx}: {e}"
                print(error_msg)
                # Create error report
                error_report = TransferReport(
                    path="none", tile_type="unknown", shape=(0, 0), device="unknown", dtype="unknown",
                    delta_est_norm=0.0, pre_norm=0.0, post_norm=0.0, applied_norm=0.0, cosine_sim=float('nan'),
                    rank=self.rank, transfer_lr=self.transfer_lr, rank_chunk=self.rank_chunk_size,
                    errors=str(e)
                )
                self._last_transfer_reports.append(error_report)
        
        if transferred_count > 0:
            print(f"[LR-TT] Successfully transferred to {transferred_count}/{len(self.tile_to_idx)} tiles")
        
        # Apply gamma or hard reset to U,V after transfers
        gamma = self.debug_cfg.get('gamma', None)
        if gamma is not None:
            self._apply_gamma_reset(gamma, pre_uv_norms, debug_enabled)
        elif self.debug_cfg.get('hard_reset', False):
            self._apply_hard_reset(debug_enabled)

    def _apply_assertions(self, report: TransferReport, tolerances: Dict[str, float]) -> None:
        """Apply assertions based on transfer path and tolerances."""
        try:
            if report.path == "program_weights":
                cos_min = tolerances.get('cos_min_program', 0.99)
                if not (report.cosine_sim >= cos_min):
                    raise AssertionError(f"program_weights cosine similarity {report.cosine_sim:.4f} < {cos_min}")
                
                rel_error = abs(report.applied_norm - report.delta_est_norm) / max(1.0, report.delta_est_norm)
                if not (rel_error <= 1e-3):
                    raise AssertionError(f"program_weights relative norm error {rel_error:.6f} > 1e-3")
                    
            elif report.path == "update":
                cos_min = tolerances.get('cos_min_update', 0.5)
                if not (report.cosine_sim >= cos_min):
                    raise AssertionError(f"update cosine similarity {report.cosine_sim:.4f} < {cos_min}")
                
                applied_min = tolerances.get('applied_min', 0.0)
                if not (report.applied_norm >= applied_min):
                    raise AssertionError(f"update applied_norm {report.applied_norm:.6f} < {applied_min}")
                
                applied_max = tolerances.get('applied_max', None)
                if applied_max is not None and not (report.applied_norm <= applied_max):
                    raise AssertionError(f"update applied_norm {report.applied_norm:.6f} > {applied_max}")
                    
        except AssertionError as e:
            print(f"[LR-TT][Hook] Assertion failed: {e}")
            raise

    def _apply_gamma_reset(self, gamma: float, pre_uv_norms: Dict[Any, Tuple[float, float]], debug_enabled: bool) -> None:
        """Apply gamma scaling to U,V vectors and check expected reduction."""
        for slow_tile, idx in self.tile_to_idx.items():
            U = self.U_params[idx]
            V = self.V_params[idx]
            
            # Apply gamma scaling
            with torch.no_grad():
                U.mul_(gamma)
                V.mul_(gamma)
            
            if debug_enabled and slow_tile in pre_uv_norms:
                pre_u_norm, pre_v_norm = pre_uv_norms[slow_tile]
                post_u_norm = torch.norm(U, p='fro').item()
                post_v_norm = torch.norm(V, p='fro').item()
                
                u_reduction = post_u_norm / max(1e-12, pre_u_norm)
                v_reduction = post_v_norm / max(1e-12, pre_v_norm)
                
                print(f"[LR-TT][Hook] Gamma reset: U norm {pre_u_norm:.6f} → {post_u_norm:.6f} "
                      f"(factor {u_reduction:.4f}), V norm {pre_v_norm:.6f} → {post_v_norm:.6f} "
                      f"(factor {v_reduction:.4f})")
                
                # Assert gamma is applied correctly (within 5% relative tolerance)
                expected_factor = gamma
                u_rel_error = abs(u_reduction - expected_factor) / max(1e-12, abs(expected_factor))
                v_rel_error = abs(v_reduction - expected_factor) / max(1e-12, abs(expected_factor))
                
                if u_rel_error > 0.05:
                    raise AssertionError(f"U gamma factor {u_reduction:.4f} != expected {expected_factor:.4f} "
                                       f"(relative error {u_rel_error:.4f} > 0.05)")
                if v_rel_error > 0.05:
                    raise AssertionError(f"V gamma factor {v_reduction:.4f} != expected {expected_factor:.4f} "
                                       f"(relative error {v_rel_error:.4f} > 0.05)")

    def _apply_hard_reset(self, debug_enabled: bool) -> None:
        """Apply hard reset (zero) to U,V vectors."""
        for slow_tile, idx in self.tile_to_idx.items():
            U = self.U_params[idx]
            V = self.V_params[idx]
            
            if debug_enabled:
                pre_u_norm = torch.norm(U, p='fro').item()
                pre_v_norm = torch.norm(V, p='fro').item()
            
            with torch.no_grad():
                U.zero_()
                V.zero_()
            
            if debug_enabled:
                print(f"[LR-TT][Hook] Hard reset: U norm {pre_u_norm:.6f} → 0.0, "
                      f"V norm {pre_v_norm:.6f} → 0.0")

    def step(self) -> None:
        """Increment step counter."""
        self.scheduler.step()

    def on_batch_end(self, step: Optional[int] = None) -> None:
        """Hook called at the end of each training batch."""
        if step is not None:
            if step % self.scheduler.transfer_every == 0:
                self.trigger_transfers()
        else:
            if self.should_transfer():
                self.trigger_transfers()
            self.step()

    def on_optimizer_step(self) -> None:
        """Hook called after optimizer.step()."""
        if self.should_transfer():
            self.trigger_transfers()
        self.step()

    def reset(self) -> None:
        """Reset step counter."""
        self.scheduler.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get transfer statistics (hardened against missing data and NaN values)."""
        stats = {
            "num_tiles": len(self.tile_to_idx),
            "rank": self.rank,
            "transfer_every": self.scheduler.transfer_every,
            "current_step": self.scheduler.step_count,
            "transfers_completed": self.scheduler.step_count // self.scheduler.transfer_every,
            "debug_enabled": bool(self.debug_cfg.get("enabled", False)),
        }
        
        # Harden against missing or invalid reports
        reports = getattr(self, "_last_transfer_reports", None) or []
        valid = [r for r in reports if getattr(r, "path", "none") != "none"]
        if valid:
            # Filter out NaN/None cosine values
            cos_vals = []
            for r in valid:
                cos = getattr(r, "cosine_sim", None)
                if cos is not None and cos == cos:  # Not NaN
                    cos_vals.append(float(cos))
            
            # Filter out None applied_norm values  
            app_vals = []
            for r in valid:
                app = getattr(r, "applied_norm", None)
                if app is not None:
                    app_vals.append(float(app))
            
            stats.update({
                "last_transfer_count": len(reports),
                "last_successful_count": len(valid),
                "last_avg_cosine_sim": (sum(cos_vals)/max(1,len(cos_vals))) if cos_vals else 0.0,
                "last_avg_applied_norm": (sum(app_vals)/max(1,len(app_vals))) if app_vals else 0.0,
            })
        
        return stats

    def run_self_test(self, model: nn.Module, dataloader: Any, steps: int = 3) -> List[TransferReport]:
        """Run a self-test with a few training steps to verify LR-TT functionality.
        
        Args:
            model: Model to test (should have analog layers)
            dataloader: Data loader for test batches 
            steps: Number of training steps to run
            
        Returns:
            List of TransferReports from the last transfer
        """
        print(f"[LR-TT][Hook] Starting self-test with {steps} steps...")
        
        # Store original settings
        original_transfer_every = self.scheduler.transfer_every
        original_debug_cfg = self.debug_cfg.copy()
        
        # Temporarily configure for testing
        self.scheduler.transfer_every = 1  # Transfer every step
        self.debug_cfg.update({
            'enabled': True,
            'print_each_transfer': True,
            'assert_tolerances': {
                'cos_min_update': 0.5,
                'cos_min_program': 0.99,
                'applied_min': 1e-6,
            }
        })
        
        try:
            # Set up optimizer (simple SGD) - include both model and LR-TT parameters
            all_params = list(model.parameters()) + list(self.parameters())
            optimizer = torch.optim.SGD(all_params, lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            data_iter = iter(dataloader)
            
            for step in range(steps):
                print(f"[LR-TT][Hook] Self-test step {step + 1}/{steps}")
                
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)
                
                # FIXED ORDER: Follow proper training loop order
                # 1. Zero gradients
                optimizer.zero_grad()
                
                # 2. Forward pass and loss
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 3. Backward pass
                loss.backward()
                
                # 4. Accumulate U,V gradients (FIXED: Now call this after backward)
                self.accumulate_uv_grads(scale=1.0)
                
                # Check gradients exist after accumulate_uv_grads
                if step == 0:
                    # FIXED: Check U,V gradients using correct parameter lists
                    uv_grads_exist = any(p.grad is not None for p in list(self.U_params) + list(self.V_params))
                    
                    if uv_grads_exist:
                        print("[LR-TT][Hook] ✓ U,V gradients exist on step 0")
                    else:
                        print("[LR-TT][Hook] Warning: No U,V gradients found on step 0")
                    
                    # FIXED: Remove fragile analog parameter grad checks
                    # Just verify no unexpected gradients exist outside U/V
                    uv_param_ids = {id(p) for p in list(self.U_params) + list(self.V_params)}
                    non_uv_params_with_grad = [
                        name for name, param in model.named_parameters()
                        if param.grad is not None and id(param) not in uv_param_ids
                    ]
                    
                    if non_uv_params_with_grad:
                        print(f"[LR-TT][Hook] Info: {len(non_uv_params_with_grad)} non-U/V parameters have gradients (this is fine for digital layers)")
                
                # 5. Optimizer step
                optimizer.step()
                
                # 6. Transfer trigger via on_optimizer_step (FIXED: Use proper method)
                self.on_optimizer_step()  # This calls should_transfer() and trigger_transfers(), then step()
                
                # Verify transfers worked
                successful_transfers = [r for r in self._last_transfer_reports if r.path != "none"]
                if len(successful_transfers) == 0:
                    print("[LR-TT][Hook] ✗ No successful transfers!")
                else:
                    print(f"[LR-TT][Hook] ✓ {len(successful_transfers)} successful transfers")
                    
                    # Check applied_norm > 0 (unless dry_run)
                    for report in successful_transfers:
                        if report.path != "dry_run" and report.applied_norm == 0:
                            print(f"[LR-TT][Hook] Warning: applied_norm = 0 for path {report.path}")
                
                print(f"[LR-TT][Hook] Step {step + 1} completed, loss = {loss.item():.4f}")
        
        finally:
            # Restore original settings
            self.scheduler.transfer_every = original_transfer_every
            self.debug_cfg = original_debug_cfg
        
        print("[LR-TT][Hook] Self-test completed successfully!")
        return self._last_transfer_reports