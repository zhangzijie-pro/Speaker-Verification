import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from utils.meters import compute_eer
from utils.path_utils import _resolve_path


# =========================
# Validation (verification EER) - 采样验证（和 train.py 一致）
# =========================
@torch.no_grad()
def validate_eer_sampled(
    model: torch.nn.Module,
    val_meta_path: str,
    device: torch.device,
    crop_frames: int = 400,
    num_crops: int = 6,
    max_spk: int = 120,
    per_spk: int = 3,
    num_pos: int = 3000,
    num_neg: int = 3000,
    seed: int = 1234,
) -> dict:
    model.eval()
    rng = random.Random(seed)

    items = []
    base_dir = os.path.dirname(os.path.abspath(val_meta_path))
    with open(val_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            spk = str(j["spk"])
            p = _resolve_path(j["feat"], base_dir)
            items.append((spk, p))

    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)

    spks = [s for s in spk2paths if len(spk2paths[s]) >= 2]
    rng.shuffle(spks)
    spks = spks[:max_spk]

    sample_paths, sample_spk = [], []
    for s in spks:
        ps = spk2paths[s][:]
        rng.shuffle(ps)
        ps = ps[:per_spk]
        for p in ps:
            sample_paths.append(p)
            sample_spk.append(s)

    emb_cache = {}
    for p in sample_paths:
        feat = torch.load(p, map_location="cpu")  # [T,80]
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        T = feat.size(0)
        embs = []
        if T <= crop_frames:
            x = feat.unsqueeze(0).to(device)
            e = model(x).squeeze(0).detach().cpu()
            embs.append(e)
        else:
            for _ in range(num_crops):
                s0 = rng.randint(0, T - crop_frames)
                chunk = feat[s0 : s0 + crop_frames]
                x = chunk.unsqueeze(0).to(device)
                e = model(x).squeeze(0).detach().cpu()
                embs.append(e)

        e = torch.stack(embs, 0).mean(0)
        e = e / (e.norm() + 1e-12)
        emb_cache[p] = e

    spk2idx = defaultdict(list)
    for s, p in zip(sample_spk, sample_paths):
        spk2idx[s].append(p)

    spks_with2 = [s for s in spk2idx if len(spk2idx[s]) >= 2]
    all_spks = list(spk2idx.keys())
    if len(all_spks) < 2 or len(spks_with2) == 0:
        return {"eer": 1.0, "pos_mean": 0.0, "neg_mean": 0.0}

    labels, scores = [], []
    for _ in range(num_pos):
        s = rng.choice(spks_with2)
        p1, p2 = rng.sample(spk2idx[s], 2)
        sc = float((emb_cache[p1] * emb_cache[p2]).sum().item())
        labels.append(1)
        scores.append(sc)

    for _ in range(num_neg):
        s1, s2 = rng.sample(all_spks, 2)
        p1 = rng.choice(spk2idx[s1])
        p2 = rng.choice(spk2idx[s2])
        sc = float((emb_cache[p1] * emb_cache[p2]).sum().item())
        labels.append(0)
        scores.append(sc)

    eer, _ = compute_eer(labels, scores)
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    return {
        "eer": eer,
        "pos_mean": float(np.mean(pos)),
        "neg_mean": float(np.mean(neg)),
    }