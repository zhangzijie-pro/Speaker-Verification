import torch
import torch.nn as nn

from speaker_verification.loss.aamsoftmax import AAMSoftmax
from speaker_verification.loss.diarization import DiarizationLoss


def _to_scalar_loss(x, device):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (float, int)):
        return torch.tensor(float(x), device=device)
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return torch.tensor(0.0, device=device)
        return _to_scalar_loss(x[0], device)
    if isinstance(x, dict):
        for k in ("loss", "total", "diar_loss"):
            if k in x:
                return _to_scalar_loss(x[k], device)
        s = 0.0
        found = False
        for v in x.values():
            if torch.is_tensor(v):
                s = s + v
                found = True
            elif isinstance(v, (float, int)):
                s = s + float(v)
                found = True
        if found:
            return _to_scalar_loss(s, device)
        return torch.tensor(0.0, device=device)
    raise TypeError(f"Unsupported loss type: {type(x)}")


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        embedding_dim=192,
        num_classes=1000,          # 全局speaker类别数，给AAMSoftmax和frame_logits用
        lambda_ver=1.0,
        lambda_diar=0.5,
        max_spk=4,                 # 最大混合人数，给count head用
        act_w=1.0,
        id_w=1.0,
        cnt_w=1.0,
        pos_weight=2.0,
    ):
        super().__init__()
        self.ver_loss = AAMSoftmax(embedding_dim, num_classes)
        self.diar_loss = DiarizationLoss(
            max_spk=max_spk,
            act_w=act_w,
            id_w=id_w,
            cnt_w=cnt_w,
            pos_weight=pos_weight,
        )
        self.lambda_ver = float(lambda_ver)
        self.lambda_diar = float(lambda_diar)

    def forward(
        self,
        emb,
        pred_ids,
        pred_activity,
        pred_count,
        label,
        target_ids,
        target_activity,
        target_count,
        valid_mask=None,
    ):
        device = emb.device

        if not torch.is_tensor(label):
            label = torch.tensor(label, device=device)
        label = label.long()

        ver_loss = self.ver_loss(emb, label)
        ver_loss = _to_scalar_loss(ver_loss, device)

        diar_loss = self.diar_loss(
            pred_ids,
            pred_activity,
            pred_count,
            target_ids,
            target_activity,
            target_count,
            valid_mask=valid_mask,
        )
        diar_loss = _to_scalar_loss(diar_loss, device)

        total = self.lambda_ver * ver_loss + self.lambda_diar * diar_loss
        return total