import os
import json
import random
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.ecapa import ECAPA_TDNN

# t-SNE 可视化（需要 pip install scikit-learn）
try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _clean_path_str(p: str) -> str:
    p = p.strip().strip('"').strip("'").strip()
    p = p.lstrip("\ufeff")
    p = p.replace("\\", "/")
    return p


def _resolve_path(p: str, base_dir: str) -> str:
    p = _clean_path_str(p)
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(base_dir, p))
    else:
        p = os.path.abspath(p)

    p = os.path.normpath(p).replace("\\", "/")
    # 常见错误：processed/processed
    p = p.replace("/processed/processed/", "/processed/")
    return p


def read_meta_jsonl(meta_path: str):
    """
    读取 meta.jsonl:
      {"split":"val","spk":"id0001","feat":".../xxx.pt","audio":"..."}
    返回 items: [(spk_str, feat_abs_path)]
    """
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
    """
    items: [(spk_str, feat_path)]
    返回 pairs: [(is_same(1/0), p1, p2)]
    """
    random.seed(seed)

    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)

    spks_with2 = [s for s in spk2paths.keys() if len(spk2paths[s]) >= 2]
    all_spks = list(spk2paths.keys())

    if len(spks_with2) == 0 or len(all_spks) < 2:
        raise RuntimeError(
            f"Not enough speakers for pair sampling: speakers={len(all_spks)}, speakers_with2={len(spks_with2)}"
        )

    pairs = []

    # 正对：同一说话人不同语句
    for _ in range(num_pos):
        spk = random.choice(spks_with2)
        p1, p2 = random.sample(spk2paths[spk], 2)
        pairs.append((1, p1, p2))

    # 负对：不同说话人
    for _ in range(num_neg):
        s1, s2 = random.sample(all_spks, 2)
        p1 = random.choice(spk2paths[s1])
        p2 = random.choice(spk2paths[s2])
        pairs.append((0, p1, p2))

    random.shuffle(pairs)
    return pairs


# =========================
# Embedding
# =========================
def _l2norm(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


@torch.no_grad()
def load_feat_pt(feat_path: str):
    feat_path = os.path.normpath(feat_path)

    if not os.path.exists(feat_path):
        return None

    try:
        feat = torch.load(feat_path, map_location="cpu", weights_only=True)
    except TypeError:
        feat = torch.load(feat_path, map_location="cpu")
    except Exception:
        return None

    if (not torch.is_tensor(feat)) or feat.dim() != 2:
        return None
    return feat  # [T, 80]


@torch.no_grad()
def embed_from_feat(model, feat: torch.Tensor, device: str,
                    crop_frames: int = 300, num_crops: int = 5, seed: int = 1234):
    """
    验证推荐：固定长度crop + 多crop平均
    feat: [T, 80]
    return emb: [D] (cpu, normalized)
    """
    # 为了复现性：同一个feat，每次评估切片一致（可选）
    rng = random.Random(seed)

    T = feat.size(0)
    if T <= crop_frames:
        x = feat.unsqueeze(0).to(device)  # [1,T,80]
        emb = model(x).squeeze(0).cpu()
        return _l2norm(emb)

    embs = []
    for _ in range(num_crops):
        s = rng.randint(0, T - crop_frames)
        chunk = feat[s:s + crop_frames]
        x = chunk.unsqueeze(0).to(device)
        embs.append(model(x).squeeze(0).cpu())

    emb = torch.stack(embs, dim=0).mean(dim=0)
    return _l2norm(emb)


@torch.no_grad()
def embed_from_fbank_pt(model, feat_path: str, device: str,
                        crop_frames: int = 300, num_crops: int = 5):
    feat = load_feat_pt(feat_path)
    if feat is None:
        return None
    emb = embed_from_feat(model, feat, device, crop_frames=crop_frames, num_crops=num_crops, seed=1234)
    return emb


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b 已归一化时：点积=cos
    return float((a * b).sum().item())


# =========================
# Metrics
# =========================
def compute_eer(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    P = sum(labels)
    N = len(labels) - P
    fa = N
    fr = 0

    best_diff = 1.0
    eer = 1.0
    best_th = None

    for th, lab in pairs:
        if lab == 1:
            fr += 1
        else:
            fa -= 1

        far = fa / max(1, N)
        frr = fr / max(1, P)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = 1-((far + frr) / 2.0)
            best_th = th

    return eer, best_th


def roc_points(labels, scores, num_th=200):
    mn, mx = min(scores), max(scores)
    if mx == mn:
        mx = mn + 1e-6
    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    tpr, fpr = [], []
    for th in ths:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= th)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    return fpr, tpr


def det_points(labels, scores, num_th=400):
    mn, mx = min(scores), max(scores)
    if mx == mn:
        mx = mn + 1e-6
    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    fars, frrs = [], []
    for th in ths:
        fa = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        fr = sum(1 for l, s in zip(labels, scores) if l == 1 and s < th)
        fars.append(fa / max(1, N))
        frrs.append(fr / max(1, P))
    return fars, frrs


def recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)):
    """
    embeddings: [M, D] normalized
    labels: [M] speaker id int
    """
    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-1e9)
    idx = torch.argsort(sims, dim=1, descending=True)

    M = sims.size(0)
    res = {}
    for k in ks:
        hit = 0
        for i in range(M):
            topk = idx[i, :k]
            if (labels[topk] == labels[i]).any().item():
                hit += 1
        res[k] = hit / M
    return res


# =========================
# t-SNE sampling
# =========================
@torch.no_grad()
def collect_embeddings_for_tsne(model, items, device,
                                max_spk=20, per_spk=25,
                                crop_frames=300, num_crops=5, seed=1234):
    random.seed(seed)
    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)

    spks = [s for s in spk2paths.keys() if len(spk2paths[s]) >= 2]
    random.shuffle(spks)
    spks = spks[:max_spk]

    X_list, y_list = [], []
    for spk in spks:
        paths = spk2paths[spk][:]
        random.shuffle(paths)
        paths = paths[:per_spk]
        for p in paths:
            emb = embed_from_fbank_pt(model, p, device, crop_frames=crop_frames, num_crops=num_crops)
            if emb is None:
                continue
            X_list.append(emb.numpy())
            y_list.append(spk)

    if len(X_list) == 0:
        return None, None

    # speaker string -> int id（只用于可视化/recall）
    uniq = sorted(set(y_list))
    spk2id = {s: i for i, s in enumerate(uniq)}
    y = np.array([spk2id[s] for s in y_list], dtype=np.int64)

    return np.stack(X_list, axis=0), y


# =========================
# Main
# =========================
def main():
    # ======= 你主要改这里 =======
    VAL_META = r"processed/cn_celeb2/val_meta.jsonl"
    CKPT     = r"outputs/best.pt"
    OUT_DIR  = r"outputs_eval"
    # crop参数：大约 300 帧≈3秒（10ms帧移）
    CROP_FRAMES = 400
    NUM_CROPS   = 6

    NUM_POS = 3000
    NUM_NEG = 3000
    SEED = 1234
    # ============================

    VAL_META = os.path.abspath(VAL_META)
    CKPT = os.path.abspath(CKPT)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("VAL_META =", VAL_META)
    print("CKPT     =", CKPT)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 读 val_meta
    items = read_meta_jsonl(VAL_META)
    print("items:", len(items))

    # 2) 基本检查
    sample = random.sample(items, k=min(8, len(items)))
    exist_cnt = sum(1 for _, p in sample if os.path.exists(p))
    print(f"sample exists: {exist_cnt}/{len(sample)}")
    if exist_cnt == 0:
        print("[ERROR] meta paths do not exist on disk. Example paths:")
        for _, p in sample:
            print(" ", p)
        return

    # speaker 统计
    spk2paths = defaultdict(list)
    for spk, p in items:
        spk2paths[spk].append(p)
    spks = list(spk2paths.keys())
    print("unique speakers in val:", len(spks))
    lens = sorted([len(v) for v in spk2paths.values()], reverse=True)
    print("per-spk utt count top5:", lens[:5])

    # 3) build pairs
    pairs = build_pairs(items, num_pos=NUM_POS, num_neg=NUM_NEG, seed=SEED)
    print("pairs:", len(pairs))

    # 4) load model
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=256).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 5) scoring
    emb_cache = {}
    labels, scores = [], []
    missing = 0
    used = 0

    for is_same, p1, p2 in tqdm(pairs, desc="Scoring"):
        if p1 not in emb_cache:
            emb_cache[p1] = embed_from_fbank_pt(
                model, p1, device,
                crop_frames=CROP_FRAMES, num_crops=NUM_CROPS
            )
        if p2 not in emb_cache:
            emb_cache[p2] = embed_from_fbank_pt(
                model, p2, device,
                crop_frames=CROP_FRAMES, num_crops=NUM_CROPS
            )

        e1 = emb_cache[p1]
        e2 = emb_cache[p2]
        if e1 is None or e2 is None:
            missing += 1
            continue

        scores.append(cosine_sim(e1, e2))
        labels.append(is_same)
        used += 1

    print(f"Scoring used pairs: {used}, skipped(missing feats): {missing}")
    if used == 0:
        print("[ERROR] used==0: cannot load any feature.")
        return

    # 6) EER
    eer, th = compute_eer(labels, scores)
    print(f"EER = {eer*100:.2f}%  (best_th≈{th:.4f})")

    # 7) 额外自检：正负分数均值
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    print(f"pos mean={np.mean(pos_scores):.4f} std={np.std(pos_scores):.4f}  "
          f"neg mean={np.mean(neg_scores):.4f} std={np.std(neg_scores):.4f}")

    # 8) ROC
    fpr, tpr = roc_points(labels, scores, num_th=200)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (EER={eer*100:.2f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc.png"))
    plt.close()

    # 9) DET
    fars, frrs = det_points(labels, scores, num_th=400)
    plt.figure()
    plt.plot(fars, frrs)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title(f"DET (EER={eer*100:.2f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "det.png"))
    plt.close()

    # 10) Score histogram
    plt.figure()
    plt.hist(pos_scores, bins=60, alpha=0.6, label="same")
    plt.hist(neg_scores, bins=60, alpha=0.6, label="diff")
    plt.legend()
    plt.title("Score distribution (cosine, normalized emb)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_hist.png"))
    plt.close()

    # 11) t-SNE + Recall@K（采样）
    X, y_tsne = collect_embeddings_for_tsne(
        model, items, device,
        max_spk=20, per_spk=25,
        crop_frames=CROP_FRAMES, num_crops=NUM_CROPS,
        seed=SEED
    )

    if X is not None and y_tsne is not None:
        if _HAS_SKLEARN:
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, max(5, (len(X) // 3))),
                init="pca",
                learning_rate="auto",
                random_state=SEED,
            )
            Z = tsne.fit_transform(X)

            plt.figure()
            uniq = sorted(set(y_tsne.tolist()))
            for spk_id in uniq:
                mask = (y_tsne == spk_id)
                plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.8)
            plt.title("t-SNE of Speaker Embeddings (sampled, crop-avg)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "tsne.png"))
            plt.close()
        else:
            print("[WARN] sklearn not found, skip t-SNE. Install: pip install scikit-learn")

        emb_t = torch.from_numpy(X).float()
        emb_t = emb_t / (emb_t.norm(dim=1, keepdim=True) + 1e-12)
        lab_t = torch.from_numpy(y_tsne).long()

        r = recall_at_k(emb_t, lab_t, ks=(1, 5, 10))
        print("Recall@K (sampled):", {f"R@{k}": round(v * 100, 2) for k, v in r.items()})

        with open(os.path.join(OUT_DIR, "recall_at_k.txt"), "w", encoding="utf-8") as f:
            for k, v in r.items():
                f.write(f"Recall@{k}: {v*100:.2f}%\n")
    else:
        print("[WARN] No embeddings collected for t-SNE/Recall@K")

    print(f"Saved to: {OUT_DIR} (roc.png, det.png, score_hist.png, tsne.png?, recall_at_k.txt)")


if __name__ == "__main__":
    main()
