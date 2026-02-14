import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from speaker_verification.models.ecapa import ECAPA_TDNN
from speaker_verification.head.aamsoftmax import AAMSoftmax
from dataset.pk_sampler import PKBatchSampler
from dataset.dataset import TrainFbankPtDataset, collate_fixed

from speaker_verification.checkpointing import ModelCfg, build_ckpt, save_ckpt
from utils.seed import set_seed
from utils.meters import AverageMeter, top1_accuracy, compute_eer
from utils.path_utils import _resolve_path

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


# =========================
# Validation (verification EER) - 采样验证
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
    """采样验证 EER，支持多 crop 平均，提升稳定性"""
    model.eval()
    rng = random.Random(seed)

    # 读取 meta
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

    # 采样说话人
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

    # 提取 embedding（逐条 + 多 crop 平均）
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

    # 构造 spk -> idx
    spk2idx = defaultdict(list)
    for i, (s, p) in enumerate(zip(sample_spk, sample_paths)):
        spk2idx[s].append(p)

    spks_with2 = [s for s in spk2idx if len(spk2idx[s]) >= 2]
    all_spks = list(spk2idx.keys())

    if len(all_spks) < 2 or len(spks_with2) == 0:
        return {"eer": 1.0, "pos_mean": 0.0, "neg_mean": 0.0}

    # 采样正负样本对
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

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.4f}",
            lr=f"{optim.param_groups[0]['lr']:.2e}",
        )

    return loss_meter.avg, acc_meter.avg


# =========================
# Main
# =========================
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed if hasattr(cfg, "seed") else 1234)

    os.makedirs(cfg.out_dir, exist_ok=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TrainFbankPtDataset(
        cfg.train_list,
        crop_frames=cfg.get("crop_frames", 200),
    )
    num_classes = train_ds.num_classes
    print(f"num_classes = {num_classes}")

    train_labels = [y for (y, _) in train_ds.items]

    pk_sampler = PKBatchSampler(
        train_labels,
        P=cfg.get("p", 32),
        K=cfg.get("k", 4),
        drop_last=True,
        seed=cfg.get("seed", 1234),
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fixed,
        pin_memory=(device.type == "cuda"),
    )

    model = ECAPA_TDNN(
        in_channels=cfg.feat_dim,
        channels=cfg.channels,
        embd_dim=cfg.emb_dim,
    ).to(device)

    head = AAMSoftmax(
        cfg.emb_dim,
        num_classes,
        s=cfg.scale,
        m=cfg.margin,
    ).to(device)

    params = list(model.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    # AMP
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_eer": [], "val_pos_mean": [], "val_neg_mean": []}
    best_val_eer = 1e9

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, head, train_loader, device, num_classes,
            optim, scaler, use_amp, params, cfg.grad_clip,
        )
        scheduler.step()

        torch.cuda.empty_cache()

        # Validation (EER)
        val_info = validate_eer_sampled(
            model=model,
            val_meta_path=cfg.val_list,
            device=device,
            crop_frames=cfg.get("crop_frames_val", 400),
            num_crops=cfg.get("num_crops", 6),
            max_spk=cfg.get("max_spk", 120),
            per_spk=cfg.get("per_spk", 3),
            num_pos=cfg.get("num_pos", 3000),
            num_neg=cfg.get("num_neg", 3000),
            seed=cfg.get("seed", 1234),
        )
        val_eer = val_info["eer"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_eer"].append(val_eer)
        history["val_pos_mean"].append(val_info["pos_mean"])
        history["val_neg_mean"].append(val_info["neg_mean"])

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_EER={val_eer*100:.2f}% (pos={val_info['pos_mean']:.3f}, neg={val_info['neg_mean']:.3f})"
        )

        # torch.cuda.empty_cache()

        model_cfg = ModelCfg(
            channels=cfg.model.channels,
            emb_dim=cfg.model.emb_dim,
            feat_dim=cfg.feature.n_mels,
            sample_rate=cfg.audio.sample_rate,
        )

        ckpt = build_ckpt(
            model=model,
            head=head,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            best_eer=val_eer,
            label_map=train_ds.label_map,
            model_cfg=model_cfg,
            extra={"cfg_text": cfg.to_yaml_string() if hasattr(cfg, "to_yaml_string") else None},
        )
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), ckpt)
        
        if _HAS_PLOT:
            try:
                plot_curves(cfg.out_dir, history)
            except Exception as e:
                print(f"[WARN] plot_curves failed: {e}")

    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"训练完成！输出目录：{cfg.out_dir}")


if __name__ == "__main__":
    main()