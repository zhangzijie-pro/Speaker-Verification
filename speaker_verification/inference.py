# speaker_verification/inference.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .checkpointing import ModelCfg, load_ckpt
from .models.ecapa import ECAPA_TDNN
from .audio.features import load_wav_mono, wav_to_fbank

try:
    import onnxruntime as ort

    _HAS_ONNX = True
except ImportError:
    ort = None
    _HAS_ONNX = False


@dataclass
class SVModelPT:
    model: torch.nn.Module
    model_cfg: ModelCfg
    device: torch.device


@dataclass
class SVModelONNX:
    session: Any  # ort.InferenceSession
    input_name: str
    model_cfg: ModelCfg
    providers: Tuple[str, ...]


SVModel = Union[SVModelPT, SVModelONNX]


def _l2norm_t(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def _l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_model_pt(model_cfg: ModelCfg, device: torch.device) -> torch.nn.Module:
    model = ECAPA_TDNN(
        channels=model_cfg.channels,
        emb_dim=model_cfg.emb_dim,
        feat_dim=model_cfg.feat_dim,
    )
    model.to(device)
    model.eval()
    return model


def load_sv(
    model_path: str,
    *,
    device: str = "cpu",
    use_onnx: Optional[bool] = None,
    providers: Optional[list[str]] = None,
) -> Tuple[SVModel, Dict]:
    """
    统一加载入口：
    - .pt: 返回 SVModelPT
    - .onnx: 返回 SVModelONNX
    """
    p = Path(model_path)
    suffix = p.suffix.lower()
    if use_onnx is None:
        use_onnx = suffix == ".onnx"

    if use_onnx:
        if not _HAS_ONNX:
            raise ImportError(
                "onnxruntime 未安装。请安装：pip install onnxruntime  或  onnxruntime-gpu"
            )
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )

        session = ort.InferenceSession(str(p), providers=providers)
        input_name = session.get_inputs()[0].name
        
        model_cfg = ModelCfg()

        sv = SVModelONNX(
            session=session,
            input_name=input_name,
            model_cfg=model_cfg,
            providers=tuple(providers),
        )
        return sv, {"path": str(p), "backend": "onnx", "providers": providers}

    # PyTorch
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = load_ckpt(str(p), map_location="cpu")
    model_cfg = ModelCfg(**ckpt["model_cfg"])
    model = build_model_pt(model_cfg, dev)
    model.load_state_dict(ckpt["model_state"], strict=True)

    sv = SVModelPT(model=model, model_cfg=model_cfg, device=dev)
    return sv, ckpt


@torch.no_grad()
def extract_embedding_pt(
    sv: SVModelPT,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> torch.Tensor:
    wav = load_wav_mono(wav_path, target_sr=sv.model_cfg.sample_rate)  # [T]
    feat = wav_to_fbank(
        wav,
        n_mels=sv.model_cfg.feat_dim,
        num_crops=num_crops,
        crop_sec=crop_sec,
    )  # [N, T, F] or [N, T, n_mels]
    x = feat.to(sv.device)

    emb = sv.model(x)  # [N, D] or [D]
    if emb.dim() == 2:
        emb = emb.mean(dim=0)  # crop average
    emb = _l2norm_t(emb).squeeze(0).detach().cpu()
    return emb


def extract_embedding_onnx(
    sv: SVModelONNX,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> np.ndarray:
    wav = load_wav_mono(wav_path, target_sr=sv.model_cfg.sample_rate)  # [T]
    feat = wav_to_fbank(
        wav,
        n_mels=sv.model_cfg.feat_dim,
        num_crops=num_crops,
        crop_sec=crop_sec,
    )  # torch.Tensor [N, T, F]
    x = feat.numpy().astype(np.float32)

    outs = sv.session.run(None, {sv.input_name: x})
    emb = outs[0]  # [N, D] or [1, D] depending export
    if emb.ndim == 2 and emb.shape[0] > 1:
        emb = emb.mean(axis=0)
    else:
        emb = emb.reshape(-1)

    emb = _l2norm_np(emb).reshape(-1)
    return emb


def extract_embedding(
    sv: SVModel,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(sv, SVModelPT):
        return extract_embedding_pt(sv, wav_path, num_crops=num_crops, crop_sec=crop_sec)
    return extract_embedding_onnx(sv, wav_path, num_crops=num_crops, crop_sec=crop_sec)


def cosine_score(
    sv: SVModel,
    wav1: str,
    wav2: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> float:
    e1 = extract_embedding(sv, wav1, num_crops=num_crops, crop_sec=crop_sec)
    e2 = extract_embedding(sv, wav2, num_crops=num_crops, crop_sec=crop_sec)

    if isinstance(e1, torch.Tensor):
        return float(torch.sum(e1 * e2).item())
    return float(np.dot(e1, e2))