import random
from collections import defaultdict
from torch.utils.data import Sampler

class PKBatchSampler(Sampler):
    """
    P-K sampler: 每个 batch 采样 P 个说话人，每个说话人 K 条语音
    适用于声纹/度量学习。

    你只要保证 Dataset 内部能通过 idx 取到 (feat, label, ...) 即可。
    """

    def __init__(
        self,
        labels,             # list[int] 与 dataset 索引对齐：labels[i] = speaker_id
        P=8,                # speakers per batch
        K=4,                # utterances per speaker
        drop_last=True,
        seed=1234
    ):
        self.labels = list(labels)
        self.P = int(P)
        self.K = int(K)
        self.bs = self.P * self.K
        self.drop_last = drop_last
        self.seed = seed

        self.spk2idx = defaultdict(list)
        for i, lab in enumerate(self.labels):
            self.spk2idx[int(lab)].append(i)

        self.valid_spks = [s for s, idxs in self.spk2idx.items() if len(idxs) >= self.K]
        if len(self.valid_spks) < self.P:
            raise ValueError(f"Not enough speakers with >=K samples. valid_spks={len(self.valid_spks)} < P={self.P}")

        self.rng = random.Random(self.seed)

        total = sum(len(self.spk2idx[s]) for s in self.valid_spks)
        self.num_batches = total // self.bs if drop_last else (total + self.bs - 1) // self.bs

    def __iter__(self):
        spks = self.valid_spks[:]
        self.rng.shuffle(spks)

        spk_ptr = 0

        for _ in range(self.num_batches):
            if spk_ptr + self.P > len(spks):
                self.rng.shuffle(spks)
                spk_ptr = 0
            batch_spks = spks[spk_ptr:spk_ptr + self.P]
            spk_ptr += self.P

            batch_indices = []
            for s in batch_spks:
                idxs = self.spk2idx[s]
                if len(idxs) >= self.K:
                    chosen = self.rng.sample(idxs, self.K)
                else:
                    chosen = [self.rng.choice(idxs) for _ in range(self.K)]
                batch_indices.extend(chosen)

            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches
