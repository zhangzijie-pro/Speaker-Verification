import json
import os
import random
import torch
from torch.utils.data import Dataset
from utils.path_utils import _resolve_path


class TrainFbankPtDataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, crop_frames: int = 200):
        self.list_path = os.path.abspath(list_path)
        base_dir = os.path.dirname(self.list_path)

        raw_items = []
        raw_labels = []

        with open(self.list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                lab_str, p = line.split(maxsplit=1)
                lab = int(lab_str)

                p = _resolve_path(p, base_dir)

                raw_items.append((lab, p))
                raw_labels.append(lab)

        # label 连续化
        uniq = sorted(set(raw_labels))
        self.label_map = {old: new for new, old in enumerate(uniq)}
        self.num_classes = len(uniq)

        self.items = [(self.label_map[lab], p) for lab, p in raw_items]

        self.crop_frames = int(crop_frames)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        y, feat_path = self.items[idx]

        feat = torch.load(feat_path, map_location="cpu")  # [T,80]
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        T = feat.size(0)

        # 固定长度裁剪（关键防 OOM）
        if T > self.crop_frames:
            s = random.randint(0, T - self.crop_frames)
            feat = feat[s:s + self.crop_frames]
        else:
            reps = (self.crop_frames + T - 1) // T
            feat = feat.repeat(reps, 1)[:self.crop_frames]

        return feat, int(y)

class ValMetaDataset(torch.utils.data.Dataset):
    """
    读取 val_meta.jsonl:
      {"spk":"id0001","feat":"...pt"}
    """
    def __init__(self, meta_path: str, crop_frames: int = 200):
        self.meta_path = os.path.abspath(meta_path)
        base_dir = os.path.dirname(self.meta_path)

        self.items = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                spk = str(j["spk"])
                feat = _resolve_path(j["feat"], base_dir)
                self.items.append((spk, feat))

        self.crop_frames = crop_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        spk, feat_path = self.items[idx]

        feat = torch.load(feat_path, map_location="cpu")
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        T = feat.size(0)

        # 固定 crop（验证也必须 crop）
        if T > self.crop_frames:
            s = random.randint(0, T - self.crop_frames)
            feat = feat[s:s + self.crop_frames]
        else:
            reps = (self.crop_frames + T - 1) // T
            feat = feat.repeat(reps, 1)[:self.crop_frames]

        return feat, spk



def spec_augment(feat, time_mask=20, freq_mask=8, p=0.5):
    # feat: [T,80]
    if random.random() > p:
        return feat
    T, F = feat.size(0), feat.size(1)

    # time mask
    t = random.randint(0, time_mask)
    t0 = random.randint(0, max(0, T - t))
    feat[t0:t0+t, :] = 0

    # freq mask
    f = random.randint(0, freq_mask)
    f0 = random.randint(0, max(0, F - f))
    feat[:, f0:f0+f] = 0
    return feat

def collate_val(batch):
    feats, spks = zip(*batch)
    x = torch.stack(feats, dim=0)
    return x, list(spks)


def collate_fixed(batch):
    feats, ys = zip(*batch)
    x = torch.stack(feats, dim=0)  # [B, crop_frames, 80]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

