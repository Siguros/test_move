# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Configuration for Low-Rank Tiki-Taka (LR-TT) Transfer Compound device."""

from dataclasses import dataclass, field
from typing import ClassVar, Any, List

from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.parameters.enums import RPUDataType


@dataclass
class LRTTTransferCompound(TransferCompound):
    r"""Low-Rank Tiki-Taka (LR-TT) Transfer compound device."""

    bindings_class: ClassVar[str] = "LRTTTransferResistiveDeviceParameter"

    # === 추가/수정된 필드 ===
    # Python에서 직접 제어할 수 있도록 노출
    units_in_mbatch: bool = False
    # LR‑TT에서 기본은 visible만 가시 가중치로 사용
    gamma_vec: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])

    # === LR-TT 고유 파라미터 ===
    rank: int = 0
    rank_chunk: int = -1
    rank_offset: int = 0
    transfer_lr: float = 1.0
    transfer_every: int = 1

    correct_gradient_magnitudes: bool = False
    swap_xd: bool = False

    # Forward injection
    forward_inject: bool = True
    lora_alpha: float = 1.0

    # BL management (A/B update)
    ab_use_bl_management: bool = True
    ab_use_update_management: bool = True
    ab_desired_bl: float = -1.0

    # BL management (transfer)
    transfer_use_bl_management: bool = False
    transfer_use_update_management: bool = False
    transfer_desired_bl: float = -1.0

    # Legacy (backward-compat only; 가능하면 사용 자제)
    use_bl_management: bool = False
    desired_bl: float = 1.0

    # Reinit
    reinit_gain: float = 1.0

    # Device indices (public)
    idx_fastA: int = 0
    idx_fastB: int = 1
    idx_visible: int = 2

    # Hidden canonical indices
    _idx_fastA: int = field(default=0, init=False, repr=False)
    _idx_fastB: int = field(default=1, init=False, repr=False)
    _idx_visible: int = field(default=2, init=False, repr=False)

    def __post_init__(self) -> None:
        # 부모 __post_init__ 호출(있으면)
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 디바이스 3개 검증
        if len(self.unit_cell_devices) != 3:
            raise ValueError(
                f"LRTTTransferCompound requires exactly 3 devices, got {len(self.unit_cell_devices)}"
            )

        # 인덱스 검증
        for name, idx in (("idx_fastA", self.idx_fastA),
                          ("idx_fastB", self.idx_fastB),
                          ("idx_visible", self.idx_visible)):
            if not (0 <= idx < 3):
                raise ValueError(f"{name} must be in [0,2], got {idx}")
        if len({self.idx_fastA, self.idx_fastB, self.idx_visible}) != 3:
            raise ValueError("Device indices must be unique")

        # 정합화(캐노니컬 순서: 0,1,2)
        self._idx_fastA, self._idx_fastB, self._idx_visible = 0, 1, 2

        # 랭크/수치 검증
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.rank_chunk != -1 and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or -1, got {self.rank_chunk}")
        if self.rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {self.rank_offset}")
        if self.transfer_lr <= 0:
            raise ValueError(f"transfer_lr must be positive, got {self.transfer_lr}")
        if hasattr(self, "transfer_every") and self.transfer_every < 0:
            raise ValueError("transfer_every must be non-negative (0 = inference)")

        # gamma_vec 검증 (A,B,visible 순서)
        if self.gamma_vec is None or len(self.gamma_vec) != 3:
            raise ValueError("gamma_vec must be a list of length 3 [fastA, fastB, visible].")

        # BL 파라미터 검증
        if self.ab_use_bl_management and self.ab_desired_bl != -1.0 and self.ab_desired_bl <= 0:
            raise ValueError("ab_desired_bl must be positive or -1.")
        if self.transfer_use_bl_management and self.transfer_desired_bl != -1.0 and self.transfer_desired_bl <= 0:
            raise ValueError("transfer_desired_bl must be positive or -1.")

        # 레거시 → ab_* 매핑(선택)
        if self.use_bl_management and self.ab_use_bl_management is True:
            import warnings
            warnings.warn("use_bl_management is deprecated. Use ab_use_bl_management instead.", DeprecationWarning)
            self.ab_use_bl_management = self.use_bl_management
        if self.desired_bl != 1.0 and self.ab_desired_bl == -1.0:
            import warnings
            warnings.warn("desired_bl is deprecated. Use ab_desired_bl instead.", DeprecationWarning)
            self.ab_desired_bl = self.desired_bl

        # ⚠ 기존 코드의 강제 고정은 제거합니다.
        # self.n_reads_per_transfer = 0  # <- 삭제 (사용자 설정을 살리기 위해)

    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Create C++ LRTTTransferResistiveDeviceParameter with canonical [A,B,visible]."""
        from aihwkit.exceptions import ConfigError
        from aihwkit.simulator import rpu_base

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError("unit_cell_devices should be a list of devices")
        if len(self.unit_cell_devices) != 3:
            raise ConfigError("LRTTTransferCompound requires exactly 3 devices")

        # 바인딩 객체 생성
        if data_type == RPUDataType.FLOAT:
            lrtt_params = rpu_base.devices.LRTTTransferResistiveDeviceParameter()
        else:
            lrtt_params = rpu_base.devices.LRTTTransferResistiveDeviceParameterDouble()

        # 유저 지정 순서대로 추가 → 내부에서는 0:A, 1:B, 2:visible로 고정
        order = [self.idx_fastA, self.idx_fastB, self.idx_visible]
        for i in order:
            device_params = self.unit_cell_devices[i].as_bindings(data_type)
            if not lrtt_params.append_parameter(device_params):
                raise ConfigError(f"Could not add unit cell device parameter at index {i}")

        lrtt_params.idx_fastA = 0
        lrtt_params.idx_fastB = 1
        lrtt_params.idx_visible = 2

        # LR‑TT 파라미터
        lrtt_params.rank = self.rank
        if hasattr(lrtt_params, "rank_chunk"):
            lrtt_params.rank_chunk = self.rank_chunk
        if hasattr(lrtt_params, "rank_offset"):
            lrtt_params.rank_offset = self.rank_offset

        lrtt_params.transfer_lr = self.transfer_lr
        lrtt_params.transfer_every = self.transfer_every if self.transfer_every >= 0 else 0

        # === 여기서 핵심 전달 ===
        if hasattr(lrtt_params, "units_in_mbatch"):
            lrtt_params.units_in_mbatch = 1 if self.units_in_mbatch else 0

        if hasattr(lrtt_params, "gamma_vec"):
            lrtt_params.gamma_vec = [float(v) for v in self.gamma_vec]

        # A/B 업데이트 및 Transfer BL 관리 플래그
        if hasattr(lrtt_params, "ab_use_bl_management"):
            lrtt_params.ab_use_bl_management = self.ab_use_bl_management
        if hasattr(lrtt_params, "ab_use_update_management"):
            lrtt_params.ab_use_update_management = self.ab_use_update_management
        if hasattr(lrtt_params, "ab_desired_bl"):
            lrtt_params.ab_desired_bl = self.ab_desired_bl

        if hasattr(lrtt_params, "transfer_use_bl_management"):
            lrtt_params.transfer_use_bl_management = self.transfer_use_bl_management
        if hasattr(lrtt_params, "transfer_use_update_management"):
            lrtt_params.transfer_use_update_management = self.transfer_use_update_management
        if hasattr(lrtt_params, "transfer_desired_bl"):
            lrtt_params.transfer_desired_bl = self.transfer_desired_bl

        # 레거시 (있으면 그대로 매핑)
        if hasattr(lrtt_params, "use_bl_management"):
            lrtt_params.use_bl_management = self.use_bl_management
        if hasattr(lrtt_params, "desired_BL"):
            lrtt_params.desired_BL = self.desired_bl

        # Forward-inject 옵션
        if hasattr(lrtt_params, "forward_inject"):
            lrtt_params.forward_inject = self.forward_inject
        if hasattr(lrtt_params, "lora_alpha"):
            lrtt_params.lora_alpha = self.lora_alpha

        # 보정 옵션
        if hasattr(lrtt_params, "correct_gradient_magnitudes"):
            lrtt_params.correct_gradient_magnitudes = self.correct_gradient_magnitudes
        if hasattr(lrtt_params, "swap_xd"):
            lrtt_params.swap_xd = self.swap_xd

        # Reinit
        if hasattr(lrtt_params, "reinit_gain"):
            lrtt_params.reinit_gain = self.reinit_gain

        # 부모에서 내려오는 것들(사용자가 바꿨다면 반영)
        if hasattr(self, "n_reads_per_transfer") and hasattr(lrtt_params, "n_reads_per_transfer"):
            lrtt_params.n_reads_per_transfer = self.n_reads_per_transfer
        if hasattr(self, "with_reset_prob") and hasattr(lrtt_params, "with_reset_prob"):
            lrtt_params.with_reset_prob = self.with_reset_prob

        # 업데이트 룰 고정
        if hasattr(lrtt_params, "update_rule"):
            lrtt_params.update_rule = "LR_TT"

        return lrtt_params

    @property
    def update_rule(self) -> str:
        return "LR_TT"

    @update_rule.setter
    def update_rule(self, value: str) -> None:
        if value != "LR_TT":
            raise ValueError(
                f"LRTTTransferCompound only supports update_rule='LR_TT', got '{value}'"
            )