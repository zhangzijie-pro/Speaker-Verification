# processed/preprocess_mixspeaker.py
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    processed_dir: str = "../processed/cn_celeb2"
    out_dir: str = "../processed/static_mix_cnceleb2"

    num_train_mixes: int = 200_000
    num_val_mixes: int = 20_000

    min_mix: int = 2
    max_mix: int = 4

    crop_sec: float = 4.0

    spk_snr_min: float = -5.0
    spk_snr_max: float = 5.0

    noise_fbank_pt_dir: str = ""
    noise_prob: float = 0.3
    noise_snr_min: float = -10.0
    noise_snr_max: float = 0.0

    allow_overlap: bool = True
    max_offset_ratio: float = 0.35

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
    lp = path.lower()
    if lp.endswith(".pt"):
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            if "fbank" in obj:
                feat = obj["fbank"]
            else:
                tens = [v for v in obj.values() if torch.is_tensor(v)]
                if not tens:
                    raise ValueError(f"PT dict has no tensor: {path}")
                feat = tens[0]
        else:
            feat = obj

        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        feat = feat.float().cpu()
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat[0]
        if feat.dim() != 2:
            raise ValueError(f"Unexpected feat shape {tuple(feat.shape)} in {path}")
        if feat.size(1) != 80 and feat.size(0) == 80:
            feat = feat.transpose(0, 1)
        if feat.size(1) != 80:
            raise ValueError(f"Expected mel=80, got shape {tuple(feat.shape)} in {path}")
        return feat

    if not _HAS_AUDIO:
        raise RuntimeError(f"Audio backend not available, but got audio file: {path}")

    wav = load_wav_mono(path, target_sr=16000)
    feat = wav_to_fbank(wav, n_mels=80)
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float().cpu()
    if feat.dim() != 2:
        raise ValueError(f"Unexpected fbank shape {tuple(feat.shape)} for audio {path}")
    return feat


def crop_or_pad_feat(x: torch.Tensor, crop_frames: int) -> torch.Tensor:
    T = x.size(0)
    if T >= crop_frames:
        s = random.randint(0, T - crop_frames)
        return x[s:s + crop_frames]
    pad = crop_frames - T
    return F.pad(x, (0, 0, 0, pad))


def resolve_path(p: str) -> str:
    return os.path.normpath(p)


def generate_one_mix(
    speakers: List[str],
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    crop_frames: int,
    cfg: StaticMixPrepCfg,
    feat_cache: Dict[str, torch.Tensor],
    noise_pts: List[str],
):
    def get_feat(p: str) -> torch.Tensor:
        p2 = resolve_path(p)
        if p2 in feat_cache:
            return feat_cache[p2]
        feat = load_feat_any(p2)
        if feat.size(0) <= 800:
            feat_cache[p2] = feat
        return feat

    k = random.randint(cfg.min_mix, cfg.max_mix)
    spks = random.sample(speakers, k)

    mixed = torch.zeros(crop_frames, 80, dtype=torch.float32)

    # 单标签 frame supervision：
    # 用“当前帧主导说话人”做 target_ids
    owner_gain = torch.full((crop_frames,), -1e9, dtype=torch.float32)
    target_ids = torch.full((crop_frames,), -100, dtype=torch.long)  # ignore index for silence
    activity = torch.zeros(crop_frames, dtype=torch.float32)

    used_global_ids = []

    for spk in spks:
        utt = random.choice(spk_to_utters[spk])
        feat = crop_or_pad_feat(get_feat(utt), crop_frames)

        spk_id = int(spk2id[spk])
        used_global_ids.append(spk_id)

        snr_db = random.uniform(cfg.spk_snr_min, cfg.spk_snr_max)
        gain = db_to_gain(snr_db)

        if cfg.allow_overlap:
            max_shift = int(crop_frames * cfg.max_offset_ratio)
            offset = random.randint(-max_shift, max_shift)
        else:
            offset = 0

        src_s0 = 0
        dst_s0 = offset
        if dst_s0 < 0:
            src_s0 = -dst_s0
            dst_s0 = 0

        length = crop_frames - dst_s0
        length = min(length, crop_frames - src_s0)
        if length <= 0:
            continue

        seg = feat[src_s0:src_s0 + length] * gain
        mixed[dst_s0:dst_s0 + length] += seg
        activity[dst_s0:dst_s0 + length] = 1.0

        # 用 gain 作为主导者近似，谁大谁拿 label
        cur_gain = torch.full((length,), float(gain), dtype=torch.float32)
        old_gain = owner_gain[dst_s0:dst_s0 + length]
        take = cur_gain > old_gain

        owner_gain[dst_s0:dst_s0 + length][take] = cur_gain[take]
        target_ids[dst_s0:dst_s0 + length][take] = spk_id

    # 加噪
    if noise_pts and (random.random() < cfg.noise_prob):
        npt = random.choice(noise_pts)
        nfeat = crop_or_pad_feat(load_feat_any(npt), crop_frames)
        ndb = random.uniform(cfg.noise_snr_min, cfg.noise_snr_max)
        mixed += nfeat * db_to_gain(ndb)

    enroll_spk = random.choice(spks)
    spk_label = int(spk2id[enroll_spk])

    target_ids = torch.where(activity > 0.5, target_ids.clamp_min(0), torch.zeros_like(target_ids))

    return {
        "fbank": mixed,                         # [T,80]
        "spk_label": spk_label,                # 全局 speaker id
        "target_ids": target_ids,              # [T] 全局 speaker id
        "target_activity": activity,           # [T]
        "target_count": int(len(set(used_global_ids))),
        "speaker_ids": sorted(list(set(used_global_ids))),
    }


def generate_split(
    split_name: str,
    num_mixes: int,
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    cfg: StaticMixPrepCfg,
    noise_pts: List[str],
):
    speakers = sorted(list(spk_to_utters.keys()))
    assert len(speakers) >= cfg.max_mix, f"[{split_name}] speakers={len(speakers)} < max_mix={cfg.max_mix}"

    mix_dir = os.path.join(cfg.out_dir, "mix_pt", split_name)
    ensure_dir(mix_dir)

    manifest_path = os.path.join(cfg.out_dir, f"{split_name}_manifest.jsonl")
    crop_frames = int(cfg.crop_sec * 100)

    feat_cache: Dict[str, torch.Tensor] = {}

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx in tqdm(range(num_mixes), desc=f"Generating {split_name} mixes"):
            pack = generate_one_mix(
                speakers=speakers,
                spk_to_utters=spk_to_utters,
                spk2id=spk2id,
                crop_frames=crop_frames,
                cfg=cfg,
                feat_cache=feat_cache,
                noise_pts=noise_pts,
            )

            rel_pt = os.path.join("mix_pt", split_name, f"{idx:08d}.pt").replace("\\", "/")
            abs_pt = os.path.join(cfg.out_dir, rel_pt)

            torch.save(pack, abs_pt)
            mf.write(json.dumps({"pt": rel_pt}, ensure_ascii=False) + "\n")

    print(f"✅ [{split_name}] Manifest: {manifest_path}")


def main():
    cfg = StaticMixPrepCfg()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_map_path = os.path.join(cfg.processed_dir, "spk_to_utterances_train.json")
    val_map_path = os.path.join(cfg.processed_dir, "spk_to_utterances_val.json")
    spk2id_path = os.path.join(cfg.processed_dir, "spk2id.json")

    assert os.path.isfile(train_map_path), f"Missing {train_map_path}"
    assert os.path.isfile(val_map_path), f"Missing {val_map_path}"
    assert os.path.isfile(spk2id_path), f"Missing {spk2id_path}"

    with open(train_map_path, "r", encoding="utf-8") as f:
        spk_to_utters_train: Dict[str, List[str]] = json.load(f)

    with open(val_map_path, "r", encoding="utf-8") as f:
        spk_to_utters_val: Dict[str, List[str]] = json.load(f)

    with open(spk2id_path, "r", encoding="utf-8") as f:
        spk2id: Dict[str, int] = json.load(f)

    ensure_dir(cfg.out_dir)

    # 直接复用统一 spk2id
    with open(os.path.join(cfg.out_dir, "spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    noise_pts = list_pt_files(cfg.noise_fbank_pt_dir)
    print(f"[StaticMixPrep] train_spks={len(spk_to_utters_train)} | val_spks={len(spk_to_utters_val)} | noise_pts={len(noise_pts)}")

    generate_split(
        split_name="train",
        num_mixes=cfg.num_train_mixes,
        spk_to_utters=spk_to_utters_train,
        spk2id=spk2id,
        cfg=cfg,
        noise_pts=noise_pts,
    )

    generate_split(
        split_name="val",
        num_mixes=cfg.num_val_mixes,
        spk_to_utters=spk_to_utters_val,
        spk2id=spk2id,
        cfg=cfg,
        noise_pts=noise_pts,
    )

    print(f"✅ Done. out_dir = {cfg.out_dir}")
    print(f"✅ train_manifest = {os.path.join(cfg.out_dir, 'train_manifest.jsonl')}")
    print(f"✅ val_manifest   = {os.path.join(cfg.out_dir, 'val_manifest.jsonl')}")
    print(f"✅ spk2id         = {os.path.join(cfg.out_dir, 'spk2id.json')}")


if __name__ == "__main__":
    main()