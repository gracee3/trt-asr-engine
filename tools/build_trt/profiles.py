from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from utils import fmt_shape


@dataclass(frozen=True)
class ShapeProfile:
    min_shapes: Dict[str, Tuple[int, ...]]
    opt_shapes: Dict[str, Tuple[int, ...]]
    max_shapes: Dict[str, Tuple[int, ...]]

    def _fmt_kv(self, shapes: Dict[str, Tuple[int, ...]]) -> str:
        # trtexec expects: name:1x2x3,name2:4x5
        parts = [f"{k}:{fmt_shape(v)}" for k, v in shapes.items()]
        return ",".join(parts)

    def trtexec_flags(self) -> Dict[str, str]:
        return {
            "--minShapes": self._fmt_kv(self.min_shapes),
            "--optShapes": self._fmt_kv(self.opt_shapes),
            "--maxShapes": self._fmt_kv(self.max_shapes),
        }


def encoder_profile(
    *,
    b: int,
    n_mels: int,
    t_min: int,
    t_opt: int,
    t_max: int,
    length_rank: int,
) -> ShapeProfile:
    """
    Encoder inputs:
      - audio_signal: [B, n_mels, T]
      - length: [1] or [1,1] depending on exporter
    """
    if length_rank == 1:
        length_shape = (1,)
    elif length_rank == 2:
        length_shape = (1, 1)
    else:
        # Conservative fallback: treat as 1D
        length_shape = (1,)

    return ShapeProfile(
        min_shapes={"audio_signal": (b, n_mels, t_min), "length": length_shape},
        opt_shapes={"audio_signal": (b, n_mels, t_opt), "length": length_shape},
        max_shapes={"audio_signal": (b, n_mels, t_max), "length": length_shape},
    )


def predictor_profile(
    *,
    b: int,
    u_min: int,
    u_opt: int,
    u_max: int,
    num_layers: int,
    hidden_size: int,
) -> ShapeProfile:
    """
    Predictor inputs:
      - y: [B, U]
      - h/c: [L, B, H]
    """
    fixed_h = (num_layers, b, hidden_size)
    return ShapeProfile(
        min_shapes={"y": (b, u_min), "h": fixed_h, "c": fixed_h},
        opt_shapes={"y": (b, u_opt), "h": fixed_h, "c": fixed_h},
        max_shapes={"y": (b, u_max), "h": fixed_h, "c": fixed_h},
    )


def joint_profile(
    *,
    b: int,
    d_enc: int,
    d_pred: int,
    t_min: int,
    t_opt: int,
    t_max: int,
    u_min: int,
    u_opt: int,
    u_max: int,
) -> ShapeProfile:
    """
    Joint inputs:
      - encoder_output: [B, D_enc, T]
      - predictor_output: [B, D_pred, U]
    """
    return ShapeProfile(
        min_shapes={
            "encoder_output": (b, d_enc, t_min),
            "predictor_output": (b, d_pred, u_min),
        },
        opt_shapes={
            "encoder_output": (b, d_enc, t_opt),
            "predictor_output": (b, d_pred, u_opt),
        },
        max_shapes={
            "encoder_output": (b, d_enc, t_max),
            "predictor_output": (b, d_pred, u_max),
        },
    )


