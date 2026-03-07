# processed/preprocess_mixspeaker.py
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
try:
    from speaker_verification.audio.features import wav_to_fbank, load_wav_mono
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False


@dataclass
class StaticMixPrepCfg:
    # 你 preprocess_cnceleb2_train.py 的输出目录（里面有 spk_to_utterances.json）
    processed_dir: str = "cn_celeb2"

    # 静态混合输出目录
    out_dir: str = "static_mix_cnceleb2"

    # 生成多少条混合样本
    num_mixes: int = 200_000

    # 每条样本混合说话人数
    min_mix: int = 2
    max_mix: int = 5

    # 每条样本期望时长（秒），用于截取/补齐
    crop_sec: float = 4.0

    # 说话人片段 SNR（相对增益）范围（dB）
    spk_snr_min: float = -5.0
    spk_snr_max: float = 5.0

    # 噪声设置（可选）：
    # 这里建议你也用 “noise_fbank_pt_dir” (里面是 .pt 的 fbank)；
    # 因为你现在训练混合也是在 fbank 域做的
    noise_fbank_pt_dir: str = ""   # e.g. "processed/musan_fbank_pt"
    noise_prob: float = 0.3
    noise_snr_min: float = -10.0
    noise_snr_max: float = 0.0

    # 随机种子
    seed: int = 1234


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def list_pt_files(root: str) -> List[str]:
    if not root or (not os.path.isdir(root)):
        return []
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pt"):
                out.append(os.path.join(r, f))
    return out


@torch.no_grad()
def load_feat_any(path: str) -> torch.Tensor:
    """
    统一读取特征：
    - 如果 path 是 .pt：直接 torch.load，支持保存成 Tensor 或 dict{'fbank':Tensor}
    - 如果 path 是音频：需要 _HAS_AUDIO，走 load_wav_mono + extract_fbank
    返回: [T,80] float32 CPU
    """
    lp = path.lower()
    if lp.endswith(".pt"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "fbank" in obj:
                feat = obj["fbank"]
            else:
                # 兜底：取第一个 tensor
                tens = [v for v in obj.values() if torch.is_tensor(v)]
                if not tens:
                    raise ValueError(f"PT dict has no tensor: {path}")
                feat = tens[0]
        else:
            feat = obj

        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        feat = feat.float().cpu()
        # 可能是 [1,T,80]
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat[0]
        # 可能是 [T,80] 或 [80,T]
        if feat.dim() != 2:
            raise ValueError(f"Unexpected feat shape {tuple(feat.shape)} in {path}")
        if feat.size(1) != 80 and feat.size(0) == 80:
            feat = feat.transpose(0, 1)  # [T,80]
        if feat.size(1) != 80:
            raise ValueError(f"Expected mel=80, got shape {tuple(feat.shape)} in {path}")
        return feat

    if not _HAS_AUDIO:
        raise RuntimeError(f"Audio backend not available, but got audio file: {path}")

    wav = load_wav_mono(path, target_sr=16000)
    feat = wav_to_fbank(wav, n_mels=80)  # [T,80]
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float().cpu()
    if feat.dim() != 2:
        raise ValueError(f"Unexpected fbank shape {tuple(feat.shape)} for audio {path}")
    return feat


def crop_or_pad_feat(x: torch.Tensor, crop_frames: int) -> torch.Tensor:
    """x: [T,80] -> [crop_frames,80]"""
    T = x.size(0)
    if T >= crop_frames:
        s = random.randint(0, T - crop_frames)
        return x[s:s + crop_frames]
    pad = crop_frames - T
    return F.pad(x, (0, 0, 0, pad))


def main():
    cfg = StaticMixPrepCfg()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    spk_map_path = os.path.join(cfg.processed_dir, "spk_to_utterances.json")
    assert os.path.isfile(spk_map_path), f"Missing {spk_map_path}. Run your preprocess first."

    with open(spk_map_path, "r", encoding="utf-8") as f:
        spk_to_utters: Dict[str, List[str]] = json.load(f)

    speakers = sorted(list(spk_to_utters.keys()))
    assert len(speakers) >= cfg.max_mix, f"speakers={len(speakers)} < max_mix={cfg.max_mix}"

    # 建一个稳定的 spk->class_id
    spk2id = {spk: i for i, spk in enumerate(speakers)}

    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    # 噪声 fbank .pt 池（推荐）
    noise_pts = list_pt_files(cfg.noise_fbank_pt_dir)
    print(f"[StaticMixPrep] speakers={len(speakers)} | noise_pts={len(noise_pts)}")

    mix_dir = os.path.join(cfg.out_dir, "mix_pt")
    ensure_dir(mix_dir)

    manifest_path = os.path.join(cfg.out_dir, "train_manifest.jsonl")
    crop_frames = int(cfg.crop_sec * 100)  # 10ms hop -> 100fps

    # 轻 cache：避免重复 load
    feat_cache: Dict[str, torch.Tensor] = {}

    def get_feat(p: str) -> torch.Tensor:
        # 你 spk_to_utterances.json 里多半是相对路径：../processed/...
        # 以 processed_dir 为基准做一次 resolve
        # 如果已经是绝对路径就不用管
        if not os.path.isabs(p):
            # 注意：你的 p 示例是 "../processed/cn_celeb2/fbank_pt/xxx.pt"
            # 在 processed_dir 旁边运行时是对的；这里稳一点，按脚本所在位置 resolve：
            # 直接用 os.path.normpath 相对当前工作目录
            p2 = os.path.normpath(p)
        else:
            p2 = p

        if p2 in feat_cache:
            return feat_cache[p2]
        feat = load_feat_any(p2)
        # cache 小一点的，避免爆内存
        if feat.size(0) <= 800:  # <= 8s
            feat_cache[p2] = feat
        return feat

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx in tqdm(range(cfg.num_mixes), desc="Generating static mixes"):
            k = random.randint(cfg.min_mix, cfg.max_mix)
            spks = random.sample(speakers, k)

            segs: List[torch.Tensor] = []
            for spk in spks:
                utt = random.choice(spk_to_utters[spk])  # 可能是 .pt
                feat = crop_or_pad_feat(get_feat(utt), crop_frames)
                segs.append(feat)

            mixed = torch.zeros(crop_frames, 80, dtype=torch.float32)
            target_ids = torch.zeros(crop_frames, dtype=torch.long)     # 0=none, 1..k=slot
            activity = torch.zeros(crop_frames, dtype=torch.float32)

            for slot_i, seg in enumerate(segs):
                # 随机 offset，制造重叠
                offset = random.randint(-crop_frames // 2, crop_frames // 2)
                snr_db = random.uniform(cfg.spk_snr_min, cfg.spk_snr_max)
                gain = db_to_gain(snr_db)

                src_s0 = 0
                dst_s0 = offset
                if dst_s0 < 0:
                    src_s0 = -dst_s0
                    dst_s0 = 0
                length = crop_frames - dst_s0
                length = min(length, crop_frames - src_s0)
                if length <= 0:
                    continue

                mixed[dst_s0:dst_s0 + length] += seg[src_s0:src_s0 + length] * gain
                target_ids[dst_s0:dst_s0 + length] = slot_i + 1
                activity[dst_s0:dst_s0 + length] = 1.0

            # 加噪（fbank 域）
            if noise_pts and (random.random() < cfg.noise_prob):
                npt = random.choice(noise_pts)
                nfeat = crop_or_pad_feat(load_feat_any(npt), crop_frames)
                ndb = random.uniform(cfg.noise_snr_min, cfg.noise_snr_max)
                mixed += nfeat * db_to_gain(ndb)

            # verification label：从本 mix 里随机挑一个说话人作为“目标”
            enroll_spk = random.choice(spks)
            spk_label = spk2id[enroll_spk]

            rel_pt = os.path.join("mix_pt", f"{idx:08d}.pt").replace("\\", "/")
            abs_pt = os.path.join(cfg.out_dir, rel_pt)

            torch.save(
                {
                    "fbank": mixed,                 # [T,80]
                    "spk_label": int(spk_label),    # int
                    "target_ids": target_ids,       # [T]
                    "target_activity": activity,    # [T]
                    "target_count": int(k),         # int
                },
                abs_pt,
            )
            mf.write(json.dumps({"pt": rel_pt}, ensure_ascii=False) + "\n")

    print(f"✅ Done. Manifest: {manifest_path}")
    print(f"✅ spk2id: {os.path.join(cfg.out_dir, 'spk2id.json')}")


if __name__ == "__main__":
    main()