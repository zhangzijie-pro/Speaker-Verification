from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelCfg:
    channels: int = 512
    emb_dim: int = 192
    feat_dim: int = 80  # fbank mel bins
    sample_rate: int = 16000


def build_ckpt(
    *,
    model: torch.nn.Module,
    head: Optional[torch.nn.Module],
    optim: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    best_eer: float,
    label_map: Optional[Dict[str, int]],
    model_cfg: ModelCfg,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ckpt = {
        "epoch": epoch,
        "best_eer": float(best_eer),
        "model_state": model.state_dict(),
        "head_state": None if head is None else head.state_dict(),
        "optim_state": None if optim is None else optim.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "label_map": label_map or {},
        "model_cfg": asdict(model_cfg),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def save_ckpt(path: str, ckpt: Dict[str, Any]) -> None:
    torch.save(ckpt, path)


def load_ckpt(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if "model_cfg" not in ckpt:
        # 兼容老 ckpt：给出默认值，但你最好尽快重新训练/重新导出
        ckpt["model_cfg"] = asdict(ModelCfg())
    return ckpt