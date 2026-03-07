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
        ignore_index: int = -100,
    ):
        super().__init__()
        self.max_spk = int(max_spk)
        self.act_w = float(act_w)
        self.id_w = float(id_w)
        self.cnt_w = float(cnt_w)
        self.ignore_index = int(ignore_index)

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(
        self,
        pred_ids,          # [B,T,C] logits, C=全局speaker类别数
        pred_activity,     # [B,T] logits
        pred_count,        # [B,K] logits, K=max_mix_speakers
        target_ids,        # [B,T] long，全局speaker id，静音/忽略可设为-100或其他后续过滤
        target_activity,   # [B,T] float 0/1
        target_count,      # [B] long/int，通常为 1..K
        valid_mask=None,   # [B,T] bool
    ):
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
            target_activity = target_activity.unsqueeze(0)
            pred_activity = pred_activity.unsqueeze(0)
            if pred_ids is not None and pred_ids.dim() == 2:
                pred_ids = pred_ids.unsqueeze(0)

        B, T = target_ids.shape
        device = target_ids.device

        if valid_mask is None:
            valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

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
        act_loss = act_loss_raw[valid_mask].mean() if valid_mask.any() else pred_activity.new_tensor(0.0)

        # ---------- id loss ----------
        id_loss = pred_activity.new_tensor(0.0)

        if pred_ids is not None:
            if pred_ids.dim() != 3:
                raise ValueError(f"pred_ids should be [B,T,C] or [B,C,T], got {tuple(pred_ids.shape)}")

            # [B,C,T] -> [B,T,C]
            if pred_ids.size(1) != T and pred_ids.size(2) == T:
                pred_ids = pred_ids.permute(0, 2, 1).contiguous()

            if pred_ids.shape[:2] != (B, T):
                raise ValueError(
                    f"pred_ids shape incompatible: got {tuple(pred_ids.shape)}, expected [B,T,C] with B={B}, T={T}"
                )

            spk_mask = (target_activity >= 0.5) & valid_mask & (target_ids != self.ignore_index)

            if spk_mask.any():
                logits = pred_ids[spk_mask]               # [N,C]
                targets = target_ids[spk_mask].long()     # [N]

                num_classes = pred_ids.size(-1)
                bad = (targets < 0) | (targets >= num_classes)
                if bad.any():
                    bad_min = int(targets.min().item())
                    bad_max = int(targets.max().item())
                    raise ValueError(
                        f"target_ids out of range for pred_ids classes={num_classes}: "
                        f"min={bad_min}, max={bad_max}"
                    )

                id_loss = F.cross_entropy(logits, targets)

        # ---------- count loss ----------
        cnt_loss = pred_activity.new_tensor(0.0)
        if pred_count is not None:
            if pred_count.dim() != 2 or pred_count.size(0) != B:
                raise ValueError(f"pred_count should be [B,K], got {tuple(pred_count.shape)}")

            tc = target_count.long()

            # 目标若是 1..K -> 映射成 0..K-1
            if tc.numel() > 0 and tc.min().item() >= 1:
                tc = tc - 1

            K = pred_count.size(1)
            bad = (tc < 0) | (tc >= K)
            if bad.any():
                bad_min = int(tc.min().item())
                bad_max = int(tc.max().item())
                raise ValueError(
                    f"target_count out of range for pred_count classes={K}: min={bad_min}, max={bad_max}"
                )

            cnt_loss = F.cross_entropy(pred_count, tc)

        total = self.act_w * act_loss + self.id_w * id_loss + self.cnt_w * cnt_loss
        return total