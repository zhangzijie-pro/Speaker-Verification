"""
Speaker Verification Evaluation Script
使用方式：
    python verify.py --val_meta processed/cn_celeb2/val_meta.jsonl --ckpt outputs/best.pt
"""

import os
import json
import random
import argparse
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.meters import compute_eer, roc_points, det_points, recall_at_k, _l2norm
from utils.path_utils import _resolve_path
from models.ecapa import ECAPA_TDNN

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def read_meta_jsonl(meta_path: str):
    meta_path = os.path.abspath(meta_path)
    base_dir = os.path.dirname(meta_path)
    items = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            spk = str(j["spk"])
            feat = _resolve_path(j["feat"], base_dir)
            items.append((spk, feat))
    return items


# =========================
# Pair building
# =========================
def build_pairs(items, num_pos=3000, num_neg=3000, seed=1234):
    random.seed(seed)
    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)

    spks_with2 = [s for s in spk2paths if len(spk2paths[s]) >= 2]
    all_spks = list(spk2paths.keys())

    if len(spks_with2) == 0:
        raise RuntimeError("Not enough speakers to generate positive pairs.")

    pairs = []

    # Positive pairs
    for _ in range(num_pos):
        spk = random.choice(spks_with2)
        p1, p2 = random.sample(spk2paths[spk], 2)
        pairs.append((1, p1, p2))

    # Negative pairs
    for _ in range(num_neg):
        s1, s2 = random.sample(all_spks, 2)
        p1 = random.choice(spk2paths[s1])
        p2 = random.choice(spk2paths[s2])
        pairs.append((0, p1, p2))

    random.shuffle(pairs)
    return pairs


# =========================
# Embedding Extraction
# =========================
@torch.no_grad()
def load_feat_pt(feat_path: str):
    if not os.path.exists(feat_path):
        return None
    try:
        feat = torch.load(feat_path, map_location="cpu", weights_only=True)
    except:
        feat = torch.load(feat_path, map_location="cpu")
    if not torch.is_tensor(feat) or feat.dim() != 2:
        return None
    return feat


@torch.no_grad()
def embed_from_feat(model, feat: torch.Tensor, device, crop_frames=400, num_crops=6, seed=1234):
    rng = random.Random(seed)
    T = feat.size(0)

    if T <= crop_frames:
        x = feat.unsqueeze(0).to(device)
        emb = model(x).squeeze(0).cpu()
        return _l2norm(emb)

    embs = []
    for _ in range(num_crops):
        s = rng.randint(0, T - crop_frames)
        chunk = feat[s:s + crop_frames]
        x = chunk.unsqueeze(0).to(device)
        embs.append(model(x).squeeze(0).cpu())

    emb = torch.stack(embs, 0).mean(0)
    return _l2norm(emb)


@torch.no_grad()
def embed_from_fbank_pt(model, feat_path: str, device, crop_frames=400, num_crops=6):
    feat = load_feat_pt(feat_path)
    if feat is None:
        return None
    return embed_from_feat(model, feat, device, crop_frames, num_crops)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


# =========================
# t-SNE + Recall@K
# =========================
@torch.no_grad()
def collect_embeddings_for_tsne(model, items, device, max_spk=20, per_spk=25,
                                crop_frames=400, num_crops=6, seed=1234):
    random.seed(seed)
    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)

    spks = [s for s in spk2paths if len(spk2paths[s]) >= 2]
    random.shuffle(spks)
    spks = spks[:max_spk]

    X_list, y_list = [], []
    for spk in spks:
        paths = random.sample(spk2paths[spk], min(per_spk, len(spk2paths[spk])))
        for p in paths:
            emb = embed_from_fbank_pt(model, p, device, crop_frames, num_crops)
            if emb is not None:
                X_list.append(emb.numpy())
                y_list.append(spk)

    if len(X_list) == 0:
        return None, None

    uniq = sorted(set(y_list))
    spk2id = {s: i for i, s in enumerate(uniq)}
    y = np.array([spk2id[s] for s in y_list], dtype=np.int64)

    return np.stack(X_list), y


# =========================
# Main Function
# =========================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Speaker Verification Evaluation")
    print("=" * 70)
    print(f"VAL_META   : {args.val_meta}")
    print(f"CHECKPOINT : {args.ckpt}")
    print(f"OUTPUT DIR : {args.out_dir}")
    print(f"Crop Frames: {args.crop_frames} | Num Crops: {args.num_crops}")
    print(f"Pairs      : {args.num_pos} pos + {args.num_neg} neg")
    print(f"Device     : {args.device}")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load meta
    items = read_meta_jsonl(args.val_meta)
    print(f"Loaded {len(items)} utterances from {len(set(spk for spk, _ in items))} speakers\n")

    # 2. Build pairs
    pairs = build_pairs(items, num_pos=args.num_pos, num_neg=args.num_neg, seed=args.seed)
    print(f"Generated {len(pairs)} pairs\n")

    # 3. Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ECAPA_TDNN(
        in_channels=80,
        channels=args.channels,
        embd_dim=args.emb_dim
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Model loaded: ECAPA-TDNN (channels={args.channels}, emb_dim={args.emb_dim})\n")

    # 4. Scoring
    emb_cache = {}
    labels, scores = [], []
    missing = 0

    for is_same, p1, p2 in tqdm(pairs, desc="Scoring"):
        if p1 not in emb_cache:
            emb_cache[p1] = embed_from_fbank_pt(model, p1, device, args.crop_frames, args.num_crops)
        if p2 not in emb_cache:
            emb_cache[p2] = embed_from_fbank_pt(model, p2, device, args.crop_frames, args.num_crops)

        e1 = emb_cache[p1]
        e2 = emb_cache[p2]

        if e1 is None or e2 is None:
            missing += 1
            continue

        scores.append(cosine_sim(e1, e2))
        labels.append(is_same)

    print(f"\nScoring completed! Used pairs: {len(scores)}, Skipped: {missing}")

    if len(scores) == 0:
        print("[ERROR] No valid pairs! Check your feature paths.")
        return

    # 5. EER & Metrics
    eer, th = compute_eer(labels, scores)
    print(f"\n>>> EER = {eer*100:.3f}%   (threshold ≈ {th:.4f})")

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    print(f"Pos mean: {np.mean(pos):.4f} | Neg mean: {np.mean(neg):.4f}")

    # 6. Save Plots
    fpr, tpr = roc_points(labels, scores, num_th=200)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (EER = {eer*100:.2f}%)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "roc.png"), dpi=300)
    plt.close()

    fars, frrs = det_points(labels, scores, num_th=400)
    plt.figure(figsize=(8, 6))
    plt.plot(fars, frrs)
    plt.title(f"DET Curve (EER = {eer*100:.2f}%)")
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("False Rejection Rate")
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "det.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(pos, bins=80, alpha=0.7, label="Same Speaker")
    plt.hist(neg, bins=80, alpha=0.7, label="Different Speaker")
    plt.title("Score Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "score_hist.png"), dpi=300)
    plt.close()

    # 7. t-SNE + Recall@K
    X, y_tsne = collect_embeddings_for_tsne(
        model, items, device,
        max_spk=20, per_spk=25,
        crop_frames=args.crop_frames,
        num_crops=args.num_crops,
        seed=args.seed
    )

    if X is not None and _HAS_SKLEARN:
        tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, init="pca")
        Z = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        for spk_id in np.unique(y_tsne):
            mask = (y_tsne == spk_id)
            plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.8)
        plt.title("t-SNE Visualization of Speaker Embeddings")
        plt.grid(True)
        plt.savefig(os.path.join(args.out_dir, "tsne.png"), dpi=300)
        plt.close()

        # Recall@K
        emb_t = torch.from_numpy(X).float()
        emb_t = emb_t / (emb_t.norm(dim=1, keepdim=True) + 1e-12)
        lab_t = torch.from_numpy(y_tsne).long()
        recall = recall_at_k(emb_t, lab_t, ks=(1, 5, 10))

        print("\nRecall@K (sampled):")
        for k, v in recall.items():
            print(f"  R@{k}: {v*100:.2f}%")

        with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
            f.write(f"EER: {eer*100:.3f}%\n")
            f.write(f"Threshold: {th:.4f}\n")
            for k, v in recall.items():
                f.write(f"Recall@{k}: {v*100:.2f}%\n")

    print(f"\n所有结果已保存至: {args.out_dir}")
    print("文件: roc.png, det.png, score_hist.png, tsne.png, metrics.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Verification Evaluation Tool")

    parser.add_argument("--val_meta", type=str, required=True,
                        help="Path to validation meta.jsonl file")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt)")

    parser.add_argument("--out_dir", type=str, default="outputs_eval",
                        help="Directory to save evaluation results (default: outputs_eval)")

    parser.add_argument("--crop_frames", type=int, default=400,
                        help="Number of frames per crop (default: 400 ≈ 4秒)")

    parser.add_argument("--num_crops", type=int, default=6,
                        help="Number of crops to average (default: 6)")

    parser.add_argument("--num_pos", type=int, default=3000,
                        help="Number of positive pairs")

    parser.add_argument("--num_neg", type=int, default=3000,
                        help="Number of negative pairs")

    parser.add_argument("--emb_dim", type=int, default=256,
                        help="Embedding dimension (must match checkpoint)")

    parser.add_argument("--channels", type=int, default=512,
                        help="ECAPA channels (default: 512)")

    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use")

    args = parser.parse_args()

    main(args)