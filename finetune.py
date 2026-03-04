import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from speaker_verification.models.ecapa import ECAPA_TDNN
from speaker_verification.head.aamsoftmax import AAMSoftmax
from speaker_verification.checkpointing import ModelCfg, build_ckpt, save_ckpt, load_ckpt

from dataset.pk_sampler import PKBatchSampler
from dataset.dataset import TrainFbankPtDataset, collate_fixed

from utils.seed import set_seed
from utils.meters import AverageMeter, top1_accuracy

from scripts.tools import validate_eer_sampled
try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def build_optimizer(cfg: DictConfig, model: torch.nn.Module, head: torch.nn.Module, phase: str):
    """
    phase:
      - "head_only": 只训练 head（backbone 冻结）
      - "full": 训练 backbone + head（支持不同 lr 倍率）
    """
    lr = float(cfg.lr)
    wd = float(cfg.weight_decay)

    fin = cfg.get("finetune", {})
    backbone_mult = float(fin.get("backbone_lr_mult", 0.5))
    head_mult = float(fin.get("head_lr_mult", 1.0))

    if phase == "head_only":
        params = [{"params": head.parameters(), "lr": lr * head_mult}]
    else:
        params = [
            {"params": model.parameters(), "lr": lr * backbone_mult},
            {"params": head.parameters(), "lr": lr * head_mult},
        ]

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return optim


def train_one_epoch(model, head, loader, device, num_classes, optim, scaler, use_amp, grad_clip):
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
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
            optim.step()

        acc = top1_accuracy(logits, y)
        bs = y.size(0)
        loss_meter.update(float(loss.item()), bs)
        acc_meter.update(float(acc), bs)

        lr0 = optim.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}", lr=f"{lr0:.2e}")

    return loss_meter.avg, acc_meter.avg


def load_backbone_weights(cfg: DictConfig, model: torch.nn.Module, device: torch.device):
    fin = cfg.get("finetune", {})
    model_only = fin.get("pretrained_model_only", "")
    full_ckpt = fin.get("pretrained_full_ckpt", "")

    if (not model_only) and (not full_ckpt):
        print("[FINETUNE] No pretrained weights provided. Training from scratch.")
        return None

    if model_only and full_ckpt:
        print("[FINETUNE] Both pretrained_model_only and pretrained_full_ckpt are set. "
              "Prefer pretrained_full_ckpt (has model_cfg).")
        model_only = ""

    if full_ckpt:
        ckpt = load_ckpt(full_ckpt, map_location=str(device))
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"[FINETUNE] Loaded backbone from full ckpt: {full_ckpt}")
        return ckpt
    else:
        sd = torch.load(model_only, map_location=str(device))
        model.load_state_dict(sd, strict=True)
        print(f"[FINETUNE] Loaded backbone from model-only state_dict: {model_only}")
        return None


@hydra.main(version_base=None, config_path="configs", config_name="finetune")
def main(cfg: DictConfig):
    set_seed(cfg.seed if hasattr(cfg, "seed") else 1234)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------
    # Dataset
    # --------
    train_ds = TrainFbankPtDataset(cfg.train_list, crop_frames=cfg.get("crop_frames", 200))
    num_classes = train_ds.num_classes
    print(f"num_classes (NEW DATASET) = {num_classes}")

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

    # --------
    # Model
    # --------
    model = ECAPA_TDNN(
        in_channels=int(cfg.feat_dim),
        channels=int(cfg.channels),
        embd_dim=int(cfg.emb_dim),
    ).to(device)

    ckpt = load_backbone_weights(cfg, model, device)

    head = AAMSoftmax(
        int(cfg.emb_dim),
        int(num_classes),
        s=float(cfg.scale),
        m=float(cfg.margin),
    ).to(device)

    # --------
    # Freeze schedule
    # --------
    fin = cfg.get("finetune", {})
    freeze_epochs = int(fin.get("freeze_backbone_epochs", 3))
    if freeze_epochs > 0:
        set_requires_grad(model, False)
        set_requires_grad(head, True)
        print(f"[FINETUNE] Freeze backbone for {freeze_epochs} epochs (train head only).")
        optim = build_optimizer(cfg, model, head, phase="head_only")
    else:
        set_requires_grad(model, True)
        set_requires_grad(head, True)
        optim = build_optimizer(cfg, model, head, phase="full")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg.epochs))

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_eer": [], "val_pos_mean": [], "val_neg_mean": []}
    best_val_eer = 1e9

    for epoch in range(1, int(cfg.epochs) + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print("[FINETUNE] Unfreeze backbone and switch to full training.")
            set_requires_grad(model, True)
            # 重新建优化器（更干净）
            optim = build_optimizer(cfg, model, head, phase="full")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg.epochs) - epoch + 1)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_loss, train_acc = train_one_epoch(
            model=model,
            head=head,
            loader=train_loader,
            device=device,
            num_classes=num_classes,
            optim=optim,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=float(cfg.grad_clip),
        )

        scheduler.step()
        torch.cuda.empty_cache()

        val_info = validate_eer_sampled(
            model=model,
            val_meta_path=cfg.val_list,
            device=device,
            crop_frames=int(cfg.get("crop_frames_val", 400)),
            num_crops=int(cfg.get("num_crops", 6)),
            max_spk=int(cfg.get("max_spk", 120)),
            per_spk=int(cfg.get("per_spk", 3)),
            num_pos=int(cfg.get("num_pos", 3000)),
            num_neg=int(cfg.get("num_neg", 3000)),
            seed=int(cfg.get("seed", 1234)),
        )
        val_eer = float(val_info["eer"])

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_eer"].append(val_eer)
        history["val_pos_mean"].append(float(val_info["pos_mean"]))
        history["val_neg_mean"].append(float(val_info["neg_mean"]))

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_EER={val_eer*100:.2f}% (pos={val_info['pos_mean']:.3f}, neg={val_info['neg_mean']:.3f})"
        )

        model_cfg = ModelCfg(
            channels=int(cfg.channels),
            emb_dim=int(cfg.emb_dim),
            feat_dim=int(cfg.feat_dim),
            sample_rate=int(cfg.get("sample_rate", 16000)),
        )
        ckpt_out = build_ckpt(
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
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), ckpt_out)

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"), ckpt_out)
            print(f"[FINETUNE] New best EER: {best_val_eer*100:.2f}% -> saved best.pt")

        if _HAS_PLOT:
            try:
                plot_curves(cfg.out_dir, history)
            except Exception as e:
                print(f"[WARN] plot_curves failed: {e}")

        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Finetune done! out_dir = {cfg.out_dir}")


if __name__ == "__main__":
    main()