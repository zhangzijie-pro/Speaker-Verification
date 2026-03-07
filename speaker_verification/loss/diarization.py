import torch
import torch.nn as nn
import torch.nn.functional as F


class DiarizationLoss(nn.Module):
    def __init__(
        self,
        max_spk: int,
        act_w: float = 1.0,
        id_w: float = 1.0,
        cnt_w: float = 1.0,
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.max_spk = max_spk
        self.act_w = act_w
        self.id_w = id_w
        self.cnt_w = cnt_w
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        pred_ids,          # [B,T,K] logits
        pred_activity,     # [B,T] logits
        pred_count,        # [B,K] logits
        target_ids,        # [B,T] long
        target_activity,   # [B,T] float 0/1
        target_count,      # [B] long/int, usually 1..K
        valid_mask=None,   # [B,T] bool
    ):
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
            target_activity = target_activity.unsqueeze(0)
            pred_activity = pred_activity.unsqueeze(0)
            if pred_ids.dim() == 2:
                pred_ids = pred_ids.unsqueeze(0)

        B, T = target_ids.shape
        device = target_ids.device

        if valid_mask is None:
            valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool()

        # ---------- activity loss ----------
        if pred_activity.dim() == 3 and pred_activity.size(-1) == 1:
            pred_activity = pred_activity.squeeze(-1)

        if pred_activity.shape != (B, T):
            raise ValueError(f"pred_activity shape {tuple(pred_activity.shape)} != {(B, T)}")

        act_loss_raw = F.binary_cross_entropy_with_logits(
            pred_activity,
            target_activity.float(),
            reduction="none",
            pos_weight=self.pos_weight.to(device),
        )

        if valid_mask.any():
            act_loss = act_loss_raw[valid_mask].mean()
        else:
            act_loss = torch.tensor(0.0, device=device)

        # ---------- id loss ----------
        id_loss = torch.tensor(0.0, device=device)

        if pred_ids is not None:
            if pred_ids.dim() != 3:
                raise ValueError(f"pred_ids should be [B,T,K] or [B,K,T], got {tuple(pred_ids.shape)}")

            # [B,K,T] -> [B,T,K]
            if pred_ids.size(1) != T and pred_ids.size(2) == T:
                pred_ids = pred_ids.permute(0, 2, 1).contiguous()

            if pred_ids.shape[:2] != (B, T):
                raise ValueError(f"pred_ids shape incompatible: {tuple(pred_ids.shape)}")

            spk_mask = (target_activity >= 0.5) & valid_mask
            if spk_mask.any():
                logits = pred_ids[spk_mask]              # [N,K]
                targets = target_ids[spk_mask].long()    # [N]
                targets = targets.clamp(0, pred_ids.size(-1) - 1)
                id_loss = F.cross_entropy(logits, targets)

        # ---------- count loss ----------
        cnt_loss = torch.tensor(0.0, device=device)
        if pred_count is not None:
            if pred_count.dim() != 2 or pred_count.size(0) != B:
                raise ValueError(f"pred_count should be [B,K], got {tuple(pred_count.shape)}")

            tc = target_count.long()

            # 如果 target_count 是 1..K，把它映射到 0..K-1
            if tc.min().item() >= 1:
                tc = tc - 1

            tc = tc.clamp(0, pred_count.size(1) - 1)
            cnt_loss = F.cross_entropy(pred_count, tc)

        total = self.act_w * act_loss + self.id_w * id_loss + self.cnt_w * cnt_loss
        return total