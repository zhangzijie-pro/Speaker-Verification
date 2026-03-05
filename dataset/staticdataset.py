# dataset/static_mix_dataset.py
import os
import json
import random
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from utils.path_utils import _resolve_path


class StaticMixDataset(Dataset):
    """
    读取 preprocess_static_mix.py 生成的:
      - out_dir/train_manifest.jsonl  (每行 {"pt": "mix_pt/00000001.pt"})
      - out_dir/spk2id.json           (用来拿 num_classes 或做 sanity check)

    每个 pt 内部保存 dict:
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
        fbank = pack["fbank"].float()  # [T,80]
        target_ids = pack["target_ids"].long()
        activity = pack["target_activity"].float()
        spk_label = int(pack["spk_label"])
        target_count = int(pack["target_count"])

        # 兜底：保证长度 == crop_frames（如果你以后生成可变长 mix，这里仍可工作）
        T = fbank.size(0)
        if T > self.crop_frames:
            s = random.randint(0, T - self.crop_frames)
            fbank = fbank[s:s + self.crop_frames]
            target_ids = target_ids[s:s + self.crop_frames]
            activity = activity[s:s + self.crop_frames]
        elif T < self.crop_frames:
            pad = self.crop_frames - T
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad))
            target_ids = torch.nn.functional.pad(target_ids, (0, pad))
            activity = torch.nn.functional.pad(activity, (0, pad))

        return {
            "fbank": fbank,  # [T,80]
            "spk_label": torch.tensor(spk_label, dtype=torch.long),
            "target_ids": target_ids,
            "target_activity": activity,
            "target_count": torch.tensor(target_count, dtype=torch.long),
        }