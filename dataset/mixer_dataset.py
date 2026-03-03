# data/dataset.py
import torch
import json
import random
import os
from torch.utils.data import Dataset
from speaker_verification.audio.features import extract_fbank, load_wav_mono

class MultiSpeakerMixingDataset(Dataset):
    def __init__(self, processed_dir="processed/cn_celeb2",
                 max_mix=5, min_mix=2, crop_sec=4.0,
                 noise_dir=None, noise_prob=0.3):
        super().__init__()
        self.processed_dir = processed_dir
        self.max_mix = max_mix
        self.min_mix = min_mix
        self.crop_frames = int(crop_sec * 100)  # 10ms hop
        self.noise_prob = noise_prob

        with open(f"{processed_dir}/spk_to_utterances.json", encoding="utf-8") as f:
            self.spk_to_utters = json.load(f)
        self.speakers = list(self.spk_to_utters.keys())
        self.all_utters = [p for lst in self.spk_to_utters.values() for p in lst]

        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for f in files:
                    if f.lower().endswith((".wav", ".flac")):
                        self.noise_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.all_utters) * 4

    def _load_fbank(self, path):
        return torch.load(path)  # [T, 80]

    def _mix_utterances(self):
        num_spk = random.randint(self.min_mix, self.max_mix)
        selected_spks = random.sample(self.speakers, num_spk)
        segments = []
        max_len = 0
        for spk in selected_spks:
            utt_path = random.choice(self.spk_to_utters[spk])
            feat = self._load_fbank(utt_path)
            segments.append((feat, int(spk)))
            max_len = max(max_len, len(feat))

        mixed = torch.zeros(max_len, 80)
        target_ids = torch.zeros(max_len, dtype=torch.long)
        activity = torch.zeros(max_len)

        for i, (seg, _) in enumerate(segments):
            len_s = len(seg)
            start = random.randint(0, max_len - len_s)
            snr = random.uniform(-5, 5)
            gain = 10 ** (snr / 20.0)
            mixed[start:start+len_s] += seg * gain
            target_ids[start:start+len_s] = i + 1
            activity[start:start+len_s] = 1.0

        if self.noise_files and random.random() < self.noise_prob:
            noise_path = random.choice(self.noise_files)
            noise_feat = self._load_fbank(noise_path)[:max_len]
            noise_snr = random.uniform(-10, 0)
            noise_gain = 10 ** (noise_snr / 20.0)
            mixed += noise_feat * noise_gain

        return mixed, target_ids, activity, num_spk

    def __getitem__(self, idx):
        fbank_mix, target_ids, activity, spk_count = self._mix_utterances()

        if len(fbank_mix) > self.crop_frames:
            start = random.randint(0, len(fbank_mix) - self.crop_frames)
            fbank_mix = fbank_mix[start:start + self.crop_frames]
            target_ids = target_ids[start:start + self.crop_frames]
            activity = activity[start:start + self.crop_frames]
        else:
            pad = self.crop_frames - len(fbank_mix)
            fbank_mix = torch.nn.functional.pad(fbank_mix, (0, 0, 0, pad))
            target_ids = torch.nn.functional.pad(target_ids, (0, pad))
            activity = torch.nn.functional.pad(activity, (0, pad))

        # 单说话人样本（用于 verification）
        single_path = random.choice(self.all_utters)
        single_feat = self._load_fbank(single_path)
        single_label = int(single_path.split("__")[0].split("/")[-1])

        return {
            'fbank': fbank_mix,              # [T, 80]
            'spk_label': single_label,
            'target_ids': target_ids,
            'target_activity': activity,
            'target_count': spk_count
        }