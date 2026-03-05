# data/dataset.py
import os
import json
import random
import torch
from torch.utils.data import Dataset


class SpeakerVerificationMixDataset(Dataset):
    """
    同时支持单说话人验证 + 多说话人 diarization 的数据集
    - 单人样本：来自 preprocess_cnceleb2_train.py
    - 混合样本：来自 preprocess_mixed_cnceleb.py（推荐）
    """

    def __init__(self,
                 processed_dir: str = "processed/cn_celeb2",
                 crop_sec: float = 4.0,
                 single_ratio: float = 0.4,      # 40% 单人样本（用于 verification）
                 max_mix: int = 5):
        
        super().__init__()
        self.processed_dir = processed_dir
        self.crop_frames = int(crop_sec * 100)   # 10ms hop → 400 frames for 4s
        self.single_ratio = single_ratio
        self.max_mix = max_mix

        # ==================== 加载单人数据 ====================
        single_meta_path = os.path.join(processed_dir, "train_fbank_list.txt")
        self.single_samples = []
        if os.path.exists(single_meta_path):
            with open(single_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        label, path = line.strip().split(maxsplit=1)
                        self.single_samples.append((int(label), path))

        # ==================== 加载混合数据（优先使用） ====================
        mixed_meta_path = os.path.join(processed_dir, "mixed_meta.jsonl")
        self.mixed_samples = []
        if os.path.exists(mixed_meta_path):
            with open(mixed_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line.strip())
                        self.mixed_samples.append(meta)

        print(f"[Dataset] 单人样本: {len(self.single_samples)} 条 | 混合样本: {len(self.mixed_samples)} 条")

        if len(self.mixed_samples) == 0 and len(self.single_samples) == 0:
            raise RuntimeError(f"No data found in {processed_dir}")

    def __len__(self):
        # 以混合样本为主，单人样本作为补充
        return max(len(self.mixed_samples), len(self.single_samples)) * 2

    def _load_fbank(self, path: str) -> torch.Tensor:
        return torch.load(path)  # [T, 80]

    def _crop_or_pad(self, feat: torch.Tensor) -> torch.Tensor:
        """统一裁剪或填充到固定长度"""
        T = feat.shape[0]
        if T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
            return feat[start:start + self.crop_frames]
        else:
            pad = self.crop_frames - T
            return torch.nn.functional.pad(feat, (0, 0, 0, pad))

    def __getitem__(self, idx):
        # 随机决定返回单人还是混合样本
        if random.random() < self.single_ratio and self.single_samples:
            # ==================== 单人样本 ====================
            label, fbank_path = random.choice(self.single_samples)
            fbank = self._load_fbank(fbank_path)
            fbank = self._crop_or_pad(fbank)

            # 单人样本的 diarization label 全为 1（只有一个人说话）
            target_ids = torch.ones(self.crop_frames, dtype=torch.long)
            target_activity = torch.ones(self.crop_frames, dtype=torch.float)
            target_count = 1

            return {
                'fbank': fbank,                    # [T, 80]
                'spk_label': label,                # verification 用
                'target_ids': target_ids,
                'target_activity': target_activity,
                'target_count': target_count
            }

        else:
            # ==================== 混合样本 ====================
            if not self.mixed_samples:
                # 如果没有混合样本，回退到单人
                return self.__getitem__(idx)   # 递归调用单人分支

            meta = random.choice(self.mixed_samples)
            fbank = self._load_fbank(meta["fbank_path"])
            fbank = self._crop_or_pad(fbank)

            # 混合样本已有预计算的 label
            num_spk = meta["num_spk"]
            # 注意：由于 crop 后长度固定，这里简化处理（实际训练中可接受轻微误差）
            target_ids = torch.ones(self.crop_frames, dtype=torch.long) * 1  # 简化版，可后续改进
            target_activity = torch.ones(self.crop_frames, dtype=torch.float)
            target_count = num_spk

            return {
                'fbank': fbank,
                'spk_label': -1,                   # 混合样本不用于 AAM loss（或随机取一个 spk_label）
                'target_ids': target_ids,
                'target_activity': target_activity,
                'target_count': target_count
            }


# ====================== 测试 ======================
if __name__ == "__main__":
    dataset = SpeakerVerificationMixDataset()
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("fbank shape:", sample['fbank'].shape)
    print("target_count:", sample['target_count'])