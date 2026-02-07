import os
import json
import random
import subprocess
from dataclasses import dataclass
from collections import defaultdict

import torch
import torchaudio
from tqdm import tqdm


# =========================
# Config
# =========================
@dataclass
class PrepConfig:
    CN_ROOT: str = r"..\CN-Celeb_flac"            # 数据集根目录
    OUT_DIR: str = r"..\processed\cn_celeb2"      # 输出目录

    USE_DEV: bool = False                       
    VAL_SPK_RATIO: float = 0.1                   

    FFMPEG: str = "ffmpeg"
    TARGET_SR: int = 16000
    N_MELS: int = 80
    MIN_SEC: float = 1.0
    SEED: int = 1234

    # 特征归一化：每条utterance减去时间维均值（CMN）
    APPLY_CMN: bool = True
    SAVE_FP16: bool = False


CFG = PrepConfig()


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_rel_key(path: str, root: str):
    """
    用相对路径生成唯一key，防止同名覆盖：
    data/id0001/a.flac -> data__id0001__a
    """
    rel = os.path.relpath(path, root)
    rel = rel.replace("\\", "/")
    rel_no_ext = os.path.splitext(rel)[0]
    return rel_no_ext.replace("/", "__")


def try_load_audio(path):
    """
    torchaudio直接读（flac/wav）
    return wav[T] float, sr
    """
    wav, sr = torchaudio.load(path)  # [C,T]
    wav = wav.mean(dim=0)            # mono [T]
    return wav, sr


def ffmpeg_to_wav(src_path, dst_path, sr):
    ensure_dir(os.path.dirname(dst_path))
    cmd = [
        CFG.FFMPEG, "-y", "-i", src_path,
        "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        dst_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode == 0, p.stdout


def load_audio_16k_mono(path):
    """
    优先 torchaudio 读；失败则 ffmpeg 转码到 fixed_wav 再读
    返回 wav[T] 16k mono
    """
    try:
        wav, sr = try_load_audio(path)
    except Exception as e1:
        # 转码文件名必须唯一：用相对路径key
        key = safe_rel_key(path, CFG.CN_ROOT)
        dst = os.path.join(CFG.OUT_DIR, "fixed_wav", key + ".wav")
        ok, log = ffmpeg_to_wav(path, dst, CFG.TARGET_SR)
        if not ok:
            raise RuntimeError(f"ffmpeg failed for {path}\n{log}") from e1
        wav, sr = try_load_audio(dst)

    if sr != CFG.TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, CFG.TARGET_SR)
    return wav.contiguous()


def wav_to_fbank(wav_16k):
    """
    wav_16k: [T]
    return feat: [T_frames, N_MELS]
    """
    wav_16k = wav_16k.unsqueeze(0)  # [1, T]
    feat = torchaudio.compliance.kaldi.fbank(
        wav_16k,
        sample_frequency=CFG.TARGET_SR,
        num_mel_bins=CFG.N_MELS,
        frame_length=25,
        frame_shift=10,
        use_energy=False,
        window_type="povey",
        dither=0.0
    )
    return feat


def scan_split_dir(split_dir):
    """
    split_dir: CN_ROOT/data 或 CN_ROOT/dev 或 CN_ROOT/eval
    返回 spk2files: dict{spk: [paths]}
    """
    spk2files = defaultdict(list)
    if not os.path.isdir(split_dir):
        return spk2files

    for spk in os.listdir(split_dir):
        spk_path = os.path.join(split_dir, spk)
        if not os.path.isdir(spk_path):
            continue
        for fn in os.listdir(spk_path):
            fn_l = fn.lower()
            if fn_l.endswith(".flac") or fn_l.endswith(".wav"):
                spk2files[spk].append(os.path.join(spk_path, fn))
    return spk2files


# =========================
# Main
# =========================
def main():
    random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)

    ensure_dir(CFG.OUT_DIR)
    feat_dir = os.path.join(CFG.OUT_DIR, "fbank_pt")
    ensure_dir(feat_dir)

    bad_log_path = os.path.join(CFG.OUT_DIR, "bad_files.txt")
    if os.path.exists(bad_log_path):
        os.remove(bad_log_path)

    data_dir = os.path.join(CFG.CN_ROOT, "data")
    dev_dir  = os.path.join(CFG.CN_ROOT, "dev")

    spk2files = scan_split_dir(data_dir)

    if CFG.USE_DEV:
        spk2files_dev = scan_split_dir(dev_dir)
        for spk, files in spk2files_dev.items():
            spk2files[spk].extend(files)

    spks = sorted([s for s in spk2files.keys() if len(spk2files[s]) > 0])
    print(f"[INFO] total speakers scanned: {len(spks)}")

    spks_shuf = spks[:]
    random.shuffle(spks_shuf)
    n_val = max(1, int(len(spks_shuf) * CFG.VAL_SPK_RATIO))
    val_spks = set(spks_shuf[:n_val])
    train_spks = [s for s in spks if s not in val_spks]
    val_spks_sorted = sorted(list(val_spks))

    print(f"[INFO] train speakers: {len(train_spks)}, val speakers: {len(val_spks_sorted)}")

    # 3) train speakers 单独编号（用于 AAM-Softmax 分类头）
    train_spk2id = {spk: i for i, spk in enumerate(train_spks)}

    # 4) 输出文件
    train_list_path = os.path.join(CFG.OUT_DIR, "train_fbank_list.txt")
    val_list_path   = os.path.join(CFG.OUT_DIR, "val_fbank_list.txt")
    train_meta_path = os.path.join(CFG.OUT_DIR, "train_meta.jsonl")
    val_meta_path   = os.path.join(CFG.OUT_DIR, "val_meta.jsonl")
    stats_path      = os.path.join(CFG.OUT_DIR, "stats.json")

    for p in [train_list_path, val_list_path, train_meta_path, val_meta_path]:
        if os.path.exists(p):
            os.remove(p)

    ok, bad = 0, 0
    spk_counts = {spk: 0 for spk in spks}

    with open(train_list_path, "w", encoding="utf-8") as ftrain, \
         open(val_list_path, "w", encoding="utf-8") as fval, \
         open(train_meta_path, "w", encoding="utf-8") as ftrainm, \
         open(val_meta_path, "w", encoding="utf-8") as fvalm, \
         open(bad_log_path, "a", encoding="utf-8") as fbad:

        for spk in tqdm(spks, desc="Extract fbank"):
            is_val = (spk in val_spks)
            files = spk2files[spk]

            for ap in files:
                try:
                    wav = load_audio_16k_mono(ap)
                    if wav.numel() < int(CFG.TARGET_SR * CFG.MIN_SEC):
                        bad += 1
                        fbad.write(f"TOO_SHORT\t{ap}\n")
                        continue

                    feat = wav_to_fbank(wav)  # [T, 80]

                    if CFG.APPLY_CMN:
                        feat = feat - feat.mean(dim=0, keepdim=True)

                    if CFG.SAVE_FP16:
                        feat = feat.half()

                    # 保存pt：包含 speaker + 相对路径key，保证唯一
                    key = safe_rel_key(ap, CFG.CN_ROOT)
                    feat_path = os.path.join(feat_dir, f"{spk}__{key}.pt")
                    torch.save(feat, feat_path)

                    feat_path_norm = feat_path.replace("\\", "/")

                    if not is_val:
                        label = train_spk2id[spk]
                        ftrain.write(f"{label} {feat_path_norm}\n")
                        meta = {"split": "train", "spk": spk, "label": int(label), "feat": feat_path_norm, "audio": ap}
                        ftrainm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    else:
                        # val 不参与分类头：label=-1，仅用于检索/验证或 metric 计算
                        fval.write(f"-1 {feat_path_norm}\n")
                        meta = {"split": "val", "spk": spk, "label": -1, "feat": feat_path_norm, "audio": ap}
                        fvalm.write(json.dumps(meta, ensure_ascii=False) + "\n")

                    spk_counts[spk] += 1
                    ok += 1

                except Exception as e:
                    bad += 1
                    fbad.write(f"ERR\t{ap}\t{repr(e)}\n")
                    continue

    with open(os.path.join(CFG.OUT_DIR, "train_spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(train_spk2id, f, ensure_ascii=False, indent=2)

    stats = {
        "cn_root": CFG.CN_ROOT,
        "use_dev": CFG.USE_DEV,
        "target_sr": CFG.TARGET_SR,
        "n_mels": CFG.N_MELS,
        "min_sec": CFG.MIN_SEC,
        "apply_cmn": CFG.APPLY_CMN,
        "save_fp16": CFG.SAVE_FP16,
        "total_speakers": len(spks),
        "train_speakers": len(train_spks),
        "val_speakers": len(val_spks_sorted),
        "total_ok_files": ok,
        "total_bad_files": bad,
        "spk_counts": spk_counts,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done!")
    print("[INFO] ok:", ok, "bad:", bad)
    print("[INFO] train_list:", train_list_path)
    print("[INFO] val_list:", val_list_path)
    print("[INFO] feat_dir:", feat_dir)
    print("[INFO] train_spk2id:", os.path.join(CFG.OUT_DIR, "train_spk2id.json"))
    print("[INFO] bad_files:", bad_log_path)
    print("[INFO] stats:", stats_path)


if __name__ == "__main__":
    main()
