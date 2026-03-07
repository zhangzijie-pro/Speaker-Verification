import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: [B, C]
    targets: [B]
    """
    if logits.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    return correct / max(1, int(targets.size(0)))


def compute_eer(labels, scores):
    """
    labels: list/np[int], 1=same, 0=diff
    scores: list/np[float], 越大越像同一个人
    rule: score >= th -> same
    return: eer, th
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0 or scores.size == 0:
        return 1.0, 0.0
    if labels.size != scores.size:
        raise ValueError(f"labels.size ({labels.size}) != scores.size ({scores.size})")

    P = int(labels.sum())
    N = int(labels.size - P)
    if P == 0 or N == 0:
        return 1.0, float(scores.max())

    order = np.argsort(scores)[::-1]
    scores_s = scores[order]
    labels_s = labels[order]

    tp = 0
    fp = 0

    # 阈值 > max(score) 时：全部判为 diff => FAR=0, FRR=1
    best_eer = 0.5
    best_th = float(scores_s[0] + 1e-6)
    best_diff = abs(0.0 - 1.0)

    i = 0
    while i < labels_s.size:
        th = scores_s[i]
        j = i
        while j < labels_s.size and scores_s[j] == th:
            if labels_s[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        far = fp / N               # false accept rate
        frr = 1.0 - (tp / P)       # false reject rate
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
            best_th = float(th)

        i = j

    return float(best_eer), float(best_th)


def roc_points(labels, scores, num_th=200):
    labels = list(labels)
    scores = list(scores)
    if len(scores) == 0:
        return [], []
    mn, mx = float(min(scores)), float(max(scores))
    if mx == mn:
        mx = mn + 1e-6

    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    tpr, fpr = [], []
    for th in ths:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= th)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    return fpr, tpr


def det_points(labels, scores, num_th=400):
    labels = list(labels)
    scores = list(scores)
    if len(scores) == 0:
        return [], []
    mn, mx = float(min(scores)), float(max(scores))
    if mx == mn:
        mx = mn + 1e-6

    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    fars, frrs = [], []
    for th in ths:
        fa = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        fr = sum(1 for l, s in zip(labels, scores) if l == 1 and s < th)
        fars.append(fa / max(1, N))
        frrs.append(fr / max(1, P))
    return fars, frrs


def recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)):
    """
    embeddings: [M, D] (建议已归一化)
    labels: [M] speaker id int
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be [M,D], got {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError("labels shape mismatch")

    # [M,M]
    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-1e9)
    idx = torch.argsort(sims, dim=1, descending=True)

    M = sims.size(0)
    res = {}
    for k in ks:
        k = int(k)
        hit = 0
        topk = idx[:, :k]  # [M,k]
        for i in range(M):
            if (labels[topk[i]] == labels[i]).any().item():
                hit += 1
        res[k] = hit / max(1, M)
    return res


def l2norm(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

def diarization_error_rate(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pred_activity: torch.Tensor,
    target_activity: torch.Tensor,
    act_th: float = 0.5,
    valid_mask: torch.Tensor = None,
    return_detail: bool = False,
):
    """
    帧级 DER:
      FA   = pred_active=1 & gt_active=0
      MISS = pred_active=0 & gt_active=1
      CONF = pred_active=1 & gt_active=1 & pred_id != gt_id

    注意:
      这个定义默认 target_ids 和 pred_ids 的类别语义一致。
      若你的 target_ids 是“每条混合语音内部重新编号”的局部 ID，
      那还需要再做 permutation matching。
    """
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
        target_activity = target_activity.unsqueeze(0)
        pred_activity = pred_activity.unsqueeze(0)
        if pred_ids.dim() == 1:
            pred_ids = pred_ids.unsqueeze(0)

    B, T = target_ids.shape

    # ---------- decode pred ids ----------
    if pred_ids.dim() == 3:
        if pred_ids.size(1) == T:
            pred_id = pred_ids.argmax(dim=-1)  # [B,T,K] -> [B,T]
        elif pred_ids.size(2) == T:
            pred_id = pred_ids.permute(0, 2, 1).contiguous().argmax(dim=-1)
        else:
            raise ValueError(f"pred_ids shape not compatible with T={T}: {tuple(pred_ids.shape)}")
    elif pred_ids.dim() == 2:
        pred_id = pred_ids
    else:
        raise ValueError(f"Unsupported pred_ids shape: {tuple(pred_ids.shape)}")

    # ---------- logits -> prob ----------
    def to_prob(x: torch.Tensor) -> torch.Tensor:
        if not x.dtype.is_floating_point:
            return x.float()
        xmin = float(x.min().item())
        xmax = float(x.max().item())
        if 0.0 <= xmin and xmax <= 1.0:
            return x
        return torch.sigmoid(x)

    pred_p = to_prob(pred_activity)
    gt_p = to_prob(target_activity)

    pred_act = pred_p >= act_th
    gt_act = gt_p >= act_th

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_act, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    pred_act = pred_act & valid_mask
    gt_act = gt_act & valid_mask

    fa = (pred_act & ~gt_act).sum().float()
    miss = (~pred_act & gt_act).sum().float()
    both = pred_act & gt_act
    conf = (both & (pred_id != target_ids)).sum().float()

    denom = gt_act.sum().float().clamp_min(1.0)
    der = (fa + miss + conf) / denom

    if return_detail:
        return der, {
            "fa": float(fa.item()),
            "miss": float(miss.item()),
            "conf": float(conf.item()),
            "gt_active": float(denom.item()),
            "pred_active": float(pred_act.sum().item()),
        }

    return der