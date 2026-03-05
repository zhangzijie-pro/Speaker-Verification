import torch
import torch.nn as nn
import torch.nn.functional as F


class DiarizationLoss(nn.Module):

    def __init__(self, max_spk: int, act_w: float = 1.0, id_w: float = 1.0, cnt_w: float = 1.0, act_th: float = 0.5):
        super().__init__()
        self.max_spk = max_spk
        self.act_w = act_w
        self.id_w = id_w
        self.cnt_w = cnt_w
        self.act_th = act_th
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_ids,            # [B,T,K] logits (推荐) 或 [B,T] id
        pred_activity,       # [B,T] logits 或 prob（推荐 logits）
        pred_count,          # [B] logits 或 回归
        target_ids,          # [B,T] long
        target_activity,     # [B,T] float 0/1
        target_count,        # [B] long/int
    ):
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
            target_activity = target_activity.unsqueeze(0)
            pred_activity = pred_activity.unsqueeze(0)
            if pred_ids.dim() == 2:
                pred_ids = pred_ids.unsqueeze(0)

        B, T = target_ids.shape

        # ---- activity loss ----
        if pred_activity.shape != (B, T):
            # 常见情况：[B,T,1]
            if pred_activity.dim() == 3 and pred_activity.size(-1) == 1:
                pred_activity = pred_activity.squeeze(-1)
            else:
                raise ValueError(f"pred_activity shape {tuple(pred_activity.shape)} != {(B,T)}")

        # BCEWithLogitsLoss 需要 logits。若你 pred_activity 已经是概率，可用 logit 变换：
        if pred_activity.min().item() >= 0.0 and pred_activity.max().item() <= 1.0:
            pred_act_logits = torch.logit(pred_activity.clamp(1e-4, 1 - 1e-4))
        else:
            pred_act_logits = pred_activity

        act_loss_per = self.bce(pred_act_logits, target_activity.float())  # [B,T]
        weight = 1.0 + target_activity.float()
        act_loss = (act_loss_per * weight).mean()

        # ---- id loss ----
        id_loss = torch.tensor(0.0, device=target_ids.device)

        if pred_ids.dim() == 3:
            # pred_ids: [B,T,K] or [B,K,T]  logits
            if pred_ids.dtype in (torch.long, torch.int64, torch.int32):
                id_loss = torch.tensor(0.0, device=target_ids.device)
            else:
                # 确保 float
                pred_ids = pred_ids.float()

                # 如果是 [B,K,T] 转 [B,T,K]
                if pred_ids.size(1) != target_ids.size(1) and pred_ids.size(2) == target_ids.size(1):
                    pred_ids = pred_ids.permute(0, 2, 1).contiguous()

                mask = (target_activity >= 0.5)
                if mask.any():
                    logits = pred_ids[mask]              # [N,K] float
                    targets = target_ids[mask].long()    # [N]
                    id_loss = F.cross_entropy(logits, targets)
        else:
            # pred_ids 是 [B,T] 离散 id -> 不算 CE（不可导）
            id_loss = torch.tensor(0.0, device=target_ids.device)

        # ---- count loss ----
        cnt_loss = torch.tensor(0.0, device=target_ids.device)
        if pred_count is not None:
            # 常见：pred_count 是 [B, max_mix] logits 分类
            if pred_count.dim() == 2 and pred_count.size(0) == B:
                # target_count 通常是 2..5，你要把它映射成 0..C-1（比如减 1 或减 min_mix）
                # 这里先做一个通用映射：假设类别从 0 开始，对齐即可。
                tc = target_count
                # 如果你的 target_count 是 1..k，且 pred_count 类别是 0..K-1，可以减 1
                if tc.min().item() >= 1 and tc.max().item() == pred_count.size(1):
                    tc = tc - 1
                cnt_loss = F.cross_entropy(pred_count, tc.clamp(0, pred_count.size(1) - 1))
            # 或者 pred_count 是 [B] 回归
            elif pred_count.dim() == 1 and pred_count.size(0) == B:
                cnt_loss = F.mse_loss(pred_count.float(), target_count.float())
            else:
                # 不匹配就不算
                cnt_loss = torch.tensor(0.0, device=target_ids.device)

        total = self.act_w * act_loss + self.id_w * id_loss + self.cnt_w * cnt_loss
        # return total, {"act": float(act_loss.item()), "id": float(id_loss.item()), "cnt": float(cnt_loss.item())}
        return total