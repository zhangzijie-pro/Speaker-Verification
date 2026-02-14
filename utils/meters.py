import numpy as np
import torch

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


def top1_accuracy(logits, targets):
    # logits: [B,C], targets: [B]
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    return correct / max(1, targets.size(0))


def compute_eer(labels, scores):
    """
    labels: list[int], 1=same, 0=diff
    scores: list[float], 越大越像同一个人
    规则：score >= th 判为 same
    返回：eer, th（th 是扫到最接近 EER 点对应的分数阈值）
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    idx = np.argsort(scores)[::-1]
    scores = scores[idx]
    labels = labels[idx]

    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return 1.0, float(scores[0] if len(scores) else 0.0)

    tp = 0
    fp = 0

    best_diff = 1e9
    best_eer = 1.0
    best_th = scores[0]

    far0 = 0.0
    frr0 = 1.0
    best_diff = abs(far0 - frr0)
    best_eer = (far0 + frr0) / 2.0
    best_th = scores[0] + 1e-6  # 比最大分数略大

    for s, lab in zip(scores, labels):
        if lab == 1:
            tp += 1
        else:
            fp += 1

        far = fp / N
        frr = 1.0 - (tp / P)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
            best_th = s

    return float(best_eer), float(best_th)

def roc_points(labels, scores, num_th=200):
    mn, mx = min(scores), max(scores)
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
    mn, mx = min(scores), max(scores)
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
    embeddings: [M, D] normalized
    labels: [M] speaker id int
    """
    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-1e9)
    idx = torch.argsort(sims, dim=1, descending=True)

    M = sims.size(0)
    res = {}
    for k in ks:
        hit = 0
        for i in range(M):
            topk = idx[i, :k]
            if (labels[topk] == labels[i]).any().item():
                hit += 1
        res[k] = hit / M
    return res


def _l2norm(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
