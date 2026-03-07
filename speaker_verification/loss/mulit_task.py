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
    def __init__(self, embedding_dim=192, num_classes=1000, lambda_ver=1.0, lambda_diar=0.5, max_spk=10):
        super().__init__()
        self.ver_loss = AAMSoftmax(embedding_dim, num_classes)
        self.diar_loss = DiarizationLoss(max_spk=max_spk)
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