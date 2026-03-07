import torch
import torch.nn.functional as F


class SpeakerMemory:
    """
    在线 speaker memory
    - 使用 cosine similarity 匹配
    - EMA 更新 prototype
    - 相似度低于阈值时创建新 speaker id
    """

    def __init__(
        self,
        emb_dim: int,
        device="cpu",
        match_th: float = 0.72,
        update_momentum: float = 0.90,
        max_speakers: int = 128,
        min_frames_to_register: int = 3,
    ):
        self.emb_dim = emb_dim
        self.device = device
        self.match_th = float(match_th)
        self.update_momentum = float(update_momentum)
        self.max_speakers = int(max_speakers)
        self.min_frames_to_register = int(min_frames_to_register)

        self.next_id = 0
        self.prototypes = {}   # spk_id -> [D]
        self.counts = {}       # spk_id -> int
        self.last_seen = {}    # spk_id -> step
        self.step = 0

    def reset(self):
        self.next_id = 0
        self.prototypes.clear()
        self.counts.clear()
        self.last_seen.clear()
        self.step = 0

    def _normalize(self, x):
        return F.normalize(x, dim=-1)

    def _similarities(self, emb: torch.Tensor):
        """
        emb: [D]
        return:
          ids: list[int]
          sims: [N]
        """
        if len(self.prototypes) == 0:
            return [], None

        ids = sorted(self.prototypes.keys())
        bank = torch.stack([self.prototypes[i] for i in ids], dim=0).to(emb.device)  # [N,D]
        sims = torch.matmul(bank, emb.unsqueeze(-1)).squeeze(-1)  # [N]
        return ids, sims

    def register_new(self, emb: torch.Tensor):
        if len(self.prototypes) >= self.max_speakers:
            ids = sorted(self.prototypes.keys())
            victim = min(ids, key=lambda x: self.last_seen.get(x, -1))
            del self.prototypes[victim]
            del self.counts[victim]
            del self.last_seen[victim]

        spk_id = self.next_id
        self.next_id += 1

        emb = self._normalize(emb.detach())
        self.prototypes[spk_id] = emb
        self.counts[spk_id] = 1
        self.last_seen[spk_id] = self.step
        return spk_id, 1.0

    def update(self, spk_id: int, emb: torch.Tensor):
        emb = self._normalize(emb.detach())
        old = self.prototypes[spk_id]
        new = self.update_momentum * old + (1.0 - self.update_momentum) * emb
        new = self._normalize(new)

        self.prototypes[spk_id] = new
        self.counts[spk_id] += 1
        self.last_seen[spk_id] = self.step

    def match(self, emb: torch.Tensor):
        """
        return:
          spk_id, score, is_new
        """
        self.step += 1
        emb = self._normalize(emb.detach())

        if len(self.prototypes) == 0:
            spk_id, score = self.register_new(emb)
            return spk_id, score, True

        ids, sims = self._similarities(emb)
        best_idx = torch.argmax(sims).item()
        best_id = ids[best_idx]
        best_score = float(sims[best_idx].item())

        if best_score >= self.match_th:
            self.update(best_id, emb)
            return best_id, best_score, False

        spk_id, score = self.register_new(emb)
        return spk_id, score, True