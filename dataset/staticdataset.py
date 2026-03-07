import os
import json
import random
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.path_utils import _resolve_path


class StaticMixDataset(Dataset):
    """
    每个 pt 内部保存:
      fbank: [T,80]
      spk_label: int
      target_ids: [T]
      target_activity: [T]
      target_count: int
    """

    def __init__(
        self,
        out_dir: str = "processed/static_mix_cnceleb2",
        manifest: str = "train_manifest.jsonl",
        crop_sec: float = 4.0,
        shuffle: bool = True,
    ):
        super().__init__()
        self.out_dir = os.path.abspath(out_dir)
        self.manifest_path = os.path.join(self.out_dir, manifest)
        assert os.path.isfile(self.manifest_path), f"Missing manifest: {self.manifest_path}"

        spk2id_path = os.path.join(self.out_dir, "spk2id.json")
        assert os.path.isfile(spk2id_path), f"Missing spk2id.json: {spk2id_path}"

        with open(spk2id_path, "r", encoding="utf-8") as f:
            self.spk2id = json.load(f)

        self.num_classes = len(self.spk2id)
        self.items: List[str] = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                self.items.append(j["pt"])

        if shuffle:
            random.shuffle(self.items)

        self.crop_frames = int(float(crop_sec) * 100)  # 10ms hop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, Any]:
        rel_pt = self.items[idx]
        abs_pt = _resolve_path(rel_pt, self.out_dir)
        pack = torch.load(abs_pt, map_location="cpu")

        fbank = pack["fbank"].float()                   # [T,80]
        target_ids = pack["target_ids"].long()         # [T]
        activity = pack["target_activity"].float()     # [T]
        spk_label = int(pack["spk_label"])
        target_count = int(pack["target_count"])

        T = fbank.size(0)
        valid_len = min(T, self.crop_frames)

        if T > self.crop_frames:
            s = random.randint(0, T - self.crop_frames)
            fbank = fbank[s:s + self.crop_frames]
            target_ids = target_ids[s:s + self.crop_frames]
            activity = activity[s:s + self.crop_frames]
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)

        elif T < self.crop_frames:
            pad = self.crop_frames - T
            fbank = F.pad(fbank, (0, 0, 0, pad))
            target_ids = F.pad(target_ids, (0, pad), value=0)
            activity = F.pad(activity, (0, pad), value=0.0)

            valid_mask = torch.zeros(self.crop_frames, dtype=torch.bool)
            valid_mask[:T] = True

        else:
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)

        return {
            "fbank": fbank,  # [T,80]
            "spk_label": torch.tensor(spk_label, dtype=torch.long),
            "target_ids": target_ids,
            "target_activity": activity,
            "target_count": torch.tensor(target_count, dtype=torch.long),
            "valid_mask": valid_mask,
        }