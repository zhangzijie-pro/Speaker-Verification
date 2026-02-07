import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
from dataclasses import asdict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from configs.train_config import TrainConfig
from models.ecapa import ECAPA_TDNN
from loss_head.aamsoftmax import AAMSoftmax
from dataset.pk_sampler import PKBatchSampler
from dataset.dataset import TrainFbankPtDataset, collate_fixed, ValMetaDataset, collate_val

from utils.seed import set_seed
from utils.meters import AverageMeter, top1_accuracy

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


# =========================
# Utils
# =========================
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
    p = p.replace("/processed/processed/", "/processed/")
    return p


def l2norm(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def compute_eer(labels, scores):
    """
    labels: list[int], 1=same, 0=diff
    scores: list[float], 越大越像同一个人
    规则：score >= th 判为 same
    返回：eer, th（th 是扫到最接近 EER 点对应的分数阈值）
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    idx = np.argsort(scores)[::-1]
    scores = scores[idx]
    labels = labels[idx]

    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return 1.0, float(scores[0] if len(scores) else 0.0)

    tp = 0
    fp = 0

    best_diff = 1e9
    best_eer = 1.0
    best_th = scores[0]

    far0 = 0.0
    frr0 = 1.0
    best_diff = abs(far0 - frr0)
    best_eer = (far0 + frr0) / 2.0
    best_th = scores[0] + 1e-6  # 比最大分数略大

    for s, lab in zip(scores, labels):
        if lab == 1:
            tp += 1
        else:
            fp += 1

        far = fp / N
        frr = 1.0 - (tp / P)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
            best_th = s

    return float(best_eer), float(best_th)

# =========================
# Validation (verification EER)
# =========================
@torch.no_grad()
def embed_batch_cropavg(model, x, lengths, device, crop_frames=300, num_crops=5, seed=1234):
    """
    x: [B,T,80] padded
    lengths: [B]
    return emb: [B,D] normalized (cpu)
    """
    rng = random.Random(seed)
    B = x.size(0)
    embs = []

    for i in range(B):
        feat = x[i, :lengths[i]].cpu() # [Ti,80]
        Ti = feat.size(0)
        if Ti <= crop_frames:
            emb = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()
            embs.append(emb)
        else:
            crops = []
            for _ in range(num_crops):
                s = rng.randint(0, Ti - crop_frames)
                chunk = feat[s:s + crop_frames]
                crops.append(model(chunk.unsqueeze(0)).squeeze(0))
            embs.append(torch.stack(crops, 0).mean(0))

    embs = torch.stack(embs, 0).cpu()
    return l2norm(embs)


@torch.no_grad()
def validate_eer_sampled(model, val_meta_path, device,
                         crop_frames=200, num_crops=3,
                         max_spk=120, per_spk=3,
                         num_pos=2000, num_neg=2000, seed=1234):
    model.eval()
    rng = random.Random(seed)

    # 读 meta
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

    # 采样 utterances
    sample_paths = []
    sample_spk = []
    for s in spks:
        ps = spk2paths[s][:]
        rng.shuffle(ps)
        ps = ps[:per_spk]
        for p in ps:
            sample_paths.append(p)
            sample_spk.append(s)

    # 提 embedding（逐条提，显存最稳；也可以小 batch）
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
                chunk = feat[s0:s0+crop_frames]
                x = chunk.unsqueeze(0).to(device)
                e = model(x).squeeze(0).detach().cpu()
                embs.append(e)

        e = torch.stack(embs, 0).mean(0)
        e = e / (e.norm() + 1e-12)
        emb_cache[p] = e

    spk2idx = defaultdict(list)
    for i, (s, p) in enumerate(zip(sample_spk, sample_paths)):
        spk2idx[s].append(p)

    spks_with2 = [s for s in spk2idx if len(spk2idx[s]) >= 2]
    all_spks = list(spk2idx.keys())
    if len(all_spks) < 2 or len(spks_with2) == 0:
        return {"eer": 1.0, "pos_mean": 0.0, "neg_mean": 0.0}

    # 采样 pairs
    labels, scores = [], []

    for _ in range(num_pos):
        s = rng.choice(spks_with2)
        p1, p2 = rng.sample(spk2idx[s], 2)
        sc = float((emb_cache[p1] * emb_cache[p2]).sum().item())
        labels.append(1); scores.append(sc)

    for _ in range(num_neg):
        s1, s2 = rng.sample(all_spks, 2)
        p1 = rng.choice(spk2idx[s1])
        p2 = rng.choice(spk2idx[s2])
        sc = float((emb_cache[p1] * emb_cache[p2]).sum().item())
        labels.append(0); scores.append(sc)

    eer, _ = compute_eer(labels, scores)
    
    eer1, th1 = compute_eer(labels, scores)
    eer2, th2 = compute_eer(labels, [-s for s in scores])
    print("="*30)
    print("eer(score) =", eer1, "eer(-score) =", eer2)
    print("="*30)

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    return {"eer": eer, "pos_mean": float(np.mean(pos)), "neg_mean": float(np.mean(neg))}


# =========================
# Train one epoch
# =========================
def train_one_epoch(model, head, loader, device, num_classes, optim, scaler, use_amp, params, grad_clip):
    model.train()
    head.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="TRAIN", ncols=110)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if y.min().item() < 0 or y.max().item() >= num_classes:
            raise RuntimeError(f"[TRAIN] label out of range: min={y.min().item()}, max={y.max().item()}, C={num_classes}")

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb = model(x)
            if not torch.isfinite(emb).all():
                raise RuntimeError("[TRAIN] Non-finite embedding detected (NaN/Inf).")
            loss, logits = head(emb, y)

        if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
            raise RuntimeError("[TRAIN] Non-finite loss/logits detected (NaN/Inf).")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optim.step()

        acc = top1_accuracy(logits, y)

        bs = y.size(0)
        loss_meter.update(float(loss.item()), bs)
        acc_meter.update(float(acc), bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}",
                         lr=f"{optim.param_groups[0]['lr']:.2e}")

    return loss_meter.avg, acc_meter.avg


# =========================
# Main
# =========================
def main():
    cfg = TrainConfig()
    set_seed(1234)

    os.makedirs(cfg.out_dir, exist_ok=True)

    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = cfg.__dict__
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    # device
    use_cuda = torch.cuda.is_available() and (str(cfg.device).startswith("cuda") if isinstance(cfg.device, str) else True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # ======================
    # Train dataset/loader
    # ======================
    train_ds = TrainFbankPtDataset(
        cfg.train_list,
        crop_frames=200
    )
    num_classes = train_ds.num_classes
    print("num_classes =", num_classes)

    train_labels = [y for (y, _) in train_ds.items]
    
    pk_sampler = PKBatchSampler(train_labels, P=32, K=4, drop_last=True, seed=1234)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fixed,
        pin_memory=(device.type == "cuda"),
    )

    val_ds = ValMetaDataset(cfg.val_list, crop_frames=200)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_val,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # 模型+头
    model = ECAPA_TDNN(in_channels=cfg.feat_dim, channels=cfg.channels, embd_dim=cfg.emb_dim).to(device)
    head = AAMSoftmax(cfg.emb_dim, num_classes, s=cfg.scale, m=cfg.margin).to(device)

    # 优化器/调度
    params = list(model.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    # AMP
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_eer": [], "val_pos_mean": [], "val_neg_mean": []}
    best_val_eer = 1e9

    CROP_FRAMES = 400   # ~3s
    NUM_CROPS = 6
    NUM_POS = 3000
    NUM_NEG = 3000

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        # train
        train_loss, train_acc = train_one_epoch(
            model, head, train_loader, device, num_classes,
            optim, scaler, use_amp, params, cfg.grad_clip
        )
        scheduler.step()
        
        torch.cuda.empty_cache()
        # val verification (EER)
        val_info = validate_eer_sampled(
            model, cfg.val_list, device,
            crop_frames=CROP_FRAMES, num_crops=NUM_CROPS,
            max_spk=120, per_spk=3,
            num_pos=NUM_POS, num_neg=NUM_NEG,
            seed=1234
        )

        val_eer = val_info["eer"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_eer"].append(val_eer)
        history["val_pos_mean"].append(val_info["pos_mean"])
        history["val_neg_mean"].append(val_info["neg_mean"])

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_EER={val_eer*100:.2f}% (pos_mean={val_info['pos_mean']:.3f}, neg_mean={val_info['neg_mean']:.3f})"
        )
        torch.cuda.empty_cache()

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optim": optim.state_dict(),
            "history": history,
            "num_classes": num_classes,
            "label_map": train_ds.label_map,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        if cfg.save_best and val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
            print(">> saved best.pt, best_val_eer =", best_val_eer)

        # curves
        if _HAS_PLOT:
            try:
                plot_curves(cfg.out_dir, history)
            except Exception as e:
                print("[WARN] plot_curves failed:", repr(e))

    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("训练完成！输出目录：", cfg.out_dir)


if __name__ == "__main__":
    main()
