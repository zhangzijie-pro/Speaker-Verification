import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from speaker_verification.models.resowave import ResoWave
from speaker_verification.loss.mulit_task import MultiTaskLoss
from dataset.mixer_dataset import MultiSpeakerMixingDataset

from speaker_verification.checkpointing import ModelCfg, build_ckpt, save_ckpt
from utils.seed import set_seed
from utils.meters import AverageMeter, compute_eer, diarization_error_rate
from utils.path_utils import _resolve_path

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except:
    _HAS_PLOT = False


@torch.no_grad()
def validate(model, loader, device, max_batches=200):
    """同时验证 SV EER 和 Diarization DER"""
    model.eval()
    eer_scores, eer_labels = [], []
    der_list = []

    pbar = tqdm(loader, desc="VALID", total=min(len(loader), max_batches))
    for i, batch in enumerate(pbar):
        if i >= max_batches:
            break
        fbank = batch['fbank'].to(device)
        spk_label = batch['spk_label'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_act = batch['target_activity'].to(device)

        emb, pred_ids, pred_act, pred_count = model(fbank, return_diarization=True)

        # SV: cosine similarity (random pairs in batch)
        for i1 in range(len(emb)):
            for i2 in range(i1+1, len(emb)):
                sc = torch.cosine_similarity(emb[i1], emb[i2], dim=0).item()
                label = 1 if spk_label[i1] == spk_label[i2] else 0
                eer_scores.append(sc)
                eer_labels.append(label)

        # Diarization DER
        der = diarization_error_rate(pred_ids, target_ids, pred_act, target_act)
        der_list.append(der.item())

        pbar.set_postfix(DER=f"{np.mean(der_list)*100:.2f}%")

    eer, _ = compute_eer(eer_labels, eer_scores)
    avg_der = np.mean(der_list) * 100
    return {"eer": eer, "der": avg_der, "pos_mean": np.mean([s for s,l in zip(eer_scores,eer_labels) if l==1]),
            "neg_mean": np.mean([s for s,l in zip(eer_scores,eer_labels) if l==0])}


def train_one_epoch(model, loss_fn, loader, device, optim, scaler, use_amp, grad_clip):
    model.train()
    loss_meter = AverageMeter()
    sv_acc_meter = AverageMeter()   # 保留 verification acc 用于监控

    pbar = tqdm(loader, desc="TRAIN", ncols=120)
    for batch in pbar:
        fbank = batch['fbank'].to(device, non_blocking=True)
        spk_label = batch['spk_label'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_act = batch['target_activity'].to(device)
        target_count = batch['target_count'].to(device)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb, pred_ids, pred_act, pred_count = model(fbank, return_diarization=True)
            loss = loss_fn(emb, pred_ids, pred_act, pred_count,
                           spk_label, target_ids, target_act, target_count)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        bs = fbank.size(0)
        loss_meter.update(float(loss.item()), bs)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            LR=f"{optim.param_groups[0]['lr']:.2e}"
        )

    return loss_meter.avg


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====================== 新数据集 ======================
    train_dataset = MultiSpeakerMixingDataset(
        processed_dir=cfg.processed_dir,
        max_mix=cfg.get("max_mix", 5),
        min_mix=cfg.get("min_mix", 2),
        crop_sec=cfg.get("crop_sec", 4.0),
        noise_dir=cfg.get("noise_dir", None),
        noise_prob=cfg.get("noise_prob", 0.3)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    # ====================== 新模型 ======================
    model = ResoWave(
        in_channels=cfg.feat_dim,
        channels=cfg.channels,
        embd_dim=cfg.emb_dim,
    ).to(device)

    # ====================== 新多任务损失 ======================
    loss_fn = MultiTaskLoss(
        embedding_dim=cfg.emb_dim,
        num_classes=len(train_dataset.speakers),   # 动态获取说话人数量
        lambda_ver=cfg.get("lambda_ver", 1.0),
        lambda_diar=cfg.get("lambda_diar", 0.5)
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_eer": [], "val_der": []}
    best_eer = 1e9

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        train_loss = train_one_epoch(
            model, loss_fn, train_loader, device, optim, scaler, use_amp, cfg.grad_clip
        )
        scheduler.step()

        # ====================== Validation ======================
        val_info = validate(model, train_loader, device, max_batches=cfg.get("val_batches", 100))

        history["train_loss"].append(train_loss)
        history["val_eer"].append(val_info["eer"])
        history["val_der"].append(val_info["der"])

        print(f"[Epoch {epoch}] Loss={train_loss:.4f} | "
              f"SV-EER={val_info['eer']*100:.2f}% | DER={val_info['der']:.2f}%")

        # ====================== Checkpoint ======================
        ckpt = build_ckpt(
            model=model,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            best_eer=val_info["eer"],
            model_cfg=ModelCfg(
                channels=cfg.channels,
                emb_dim=cfg.emb_dim,
                feat_dim=cfg.feat_dim,
                sample_rate=16000
            )
        )
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), ckpt)

        if val_info["eer"] < best_eer:
            best_eer = val_info["eer"]
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"), ckpt)
            print(f"★★★ New best EER: {best_eer*100:.2f}% ★★★")

        if _HAS_PLOT:
            try:
                plot_curves(cfg.out_dir, history)
            except Exception as e:
                print(f"[WARN] plot failed: {e}")

    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"✅ Training finished! Best EER: {best_eer*100:.2f}% | Output: {cfg.out_dir}")


if __name__ == "__main__":
    main()