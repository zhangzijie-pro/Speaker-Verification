import torch
import torch.nn as nn
from speaker_verification.loss.aamsoftmax import AAMSoftmax
from speaker_verification.loss.diarization import DiarizationLoss


def _to_scalar_loss(x, device):
    """
    把各种可能的返回值 (Tensor/float/tuple/list/dict) 统一转成 标量Tensor
    """
    # 1) Tensor
    if torch.is_tensor(x):
        return x

    # 2) float/int
    if isinstance(x, (float, int)):
        return torch.tensor(float(x), device=device)

    # 3) tuple/list：优先取第一个元素
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return torch.tensor(0.0, device=device)
        return _to_scalar_loss(x[0], device)

    # 4) dict：优先找常见 key
    if isinstance(x, dict):
        for k in ("loss", "total", "diar_loss"):
            if k in x:
                return _to_scalar_loss(x[k], device)
        # 找不到就把所有 tensor/数值相加兜底
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

    # 5) 其他类型：直接报出来
    raise TypeError(f"Unsupported loss type: {type(x)}")


class MultiTaskLoss(nn.Module):
    def __init__(self, embedding_dim=192, num_classes=1000, lambda_ver=1.0, lambda_diar=0.5, max_spk=10):
        super().__init__()
        self.ver_loss = AAMSoftmax(embedding_dim, num_classes)
        self.diar_loss = DiarizationLoss(max_spk=max_spk)

        self.lambda_ver = float(lambda_ver)
        self.lambda_diar = float(lambda_diar)

        self._printed = False

    def forward(self, emb, pred_ids, pred_activity, pred_count,
                label, target_ids, target_activity, target_count):

        device = emb.device

        # AAMSoftmax label 必须 long
        if not torch.is_tensor(label):
            label = torch.tensor(label, device=device)
        label = label.long()

        ver_loss = self.ver_loss(emb, label)
        ver_loss = _to_scalar_loss(ver_loss, device)

        diar_out = self.diar_loss(
            pred_ids, pred_activity, pred_count,
            target_ids, target_activity, target_count
        )
        diar_loss = _to_scalar_loss(diar_out, device)

        # 只打印一次，确认类型（非常关键：确保你改的文件真的生效）
        if not self._printed:
            self._printed = True
            print("[DBG] ver_loss type:", type(ver_loss), "shape:", getattr(ver_loss, "shape", None))
            print("[DBG] diar_out type:", type(diar_out))
            if isinstance(diar_out, (tuple, list)):
                print("[DBG] diar_out[0] type:", type(diar_out[0]))
            if isinstance(diar_out, dict):
                print("[DBG] diar_out keys:", list(diar_out.keys()))
            print("[DBG] diar_loss final type:", type(diar_loss), "shape:", getattr(diar_loss, "shape", None))

        total = self.lambda_ver * ver_loss + self.lambda_diar * diar_loss
        return total