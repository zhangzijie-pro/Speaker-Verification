# processed/preprocess_cnceleb2_train.py
import os
import json
import random
import torch
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append("..")
from speaker_verification.audio.features import wav_to_fbank, load_wav_mono


class PrepConfig:
    CN_ROOT = r"..\CN-Celeb_flac"
    OUT_DIR = r"..\processed\cn_celeb2"
    TARGET_SR = 16000
    N_MELS = 80
    MIN_SEC = 1.0

    VAL_UTT_RATIO = 0.1

    # 至少要有多少条音频，才保留这个 speaker
    MIN_UTTS_PER_SPK = 2

    SEED = 1234


CFG = PrepConfig()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_key(path, root):
    rel = os.path.relpath(path, root).replace("\\", "/")
    return os.path.splitext(rel)[0].replace("/", "__")


def main():
    random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)

    ensure_dir(CFG.OUT_DIR)
    feat_dir = os.path.join(CFG.OUT_DIR, "fbank_pt")
    ensure_dir(feat_dir)

    data_dir = os.path.join(CFG.CN_ROOT, "data")
    assert os.path.isdir(data_dir), f"Missing data dir: {data_dir}"

    spk2files = defaultdict(list)
    for spk in os.listdir(data_dir):
        spk_path = os.path.join(data_dir, spk)
        if not os.path.isdir(spk_path):
            continue
        for f in os.listdir(spk_path):
            if f.lower().endswith((".flac", ".wav")):
                spk2files[spk].append(os.path.join(spk_path, f))

    # 过滤过短 speaker
    kept_spks = []
    for spk, files in spk2files.items():
        if len(files) >= CFG.MIN_UTTS_PER_SPK:
            kept_spks.append(spk)

    kept_spks = sorted(kept_spks)
    print(f"[Prep] kept speakers = {len(kept_spks)}")

    # 全局统一 spk2id
    spk2id = {spk: i for i, spk in enumerate(kept_spks)}

    # 训练/验证 utterance 映射
    spk_to_utters_train = defaultdict(list)
    spk_to_utters_val = defaultdict(list)

    train_list_path = os.path.join(CFG.OUT_DIR, "train_fbank_list.txt")
    val_list_path = os.path.join(CFG.OUT_DIR, "val_fbank_list.txt")
    spk_train_path = os.path.join(CFG.OUT_DIR, "spk_to_utterances_train.json")
    spk_val_path = os.path.join(CFG.OUT_DIR, "spk_to_utterances_val.json")
    spk2id_path = os.path.join(CFG.OUT_DIR, "spk2id.json")
    val_meta_path = os.path.join(CFG.OUT_DIR, "val_meta.jsonl")

    val_meta = []

    with open(train_list_path, "w", encoding="utf-8") as ftrain, \
         open(val_list_path, "w", encoding="utf-8") as fval:

        for spk in tqdm(kept_spks, desc="Preprocessing CN-Celeb2"):
            files = sorted(spk2files[spk])
            random.shuffle(files)

            n_val = max(1, int(len(files) * CFG.VAL_UTT_RATIO))
            if n_val >= len(files):
                n_val = len(files) - 1

            val_files = set(files[:n_val])
            train_files = files[n_val:]

            for wav_path in files:
                try:
                    wav = load_wav_mono(wav_path, target_sr=CFG.TARGET_SR)
                    if len(wav) < CFG.TARGET_SR * CFG.MIN_SEC:
                        continue

                    feat = wav_to_fbank(
                        wav,
                        n_mels=CFG.N_MELS,
                        num_crops=1,
                        crop_sec=max(CFG.MIN_SEC, float(len(wav)) / CFG.TARGET_SR),
                    )[0]

                    key = safe_key(wav_path, CFG.CN_ROOT)
                    feat_path = os.path.join(feat_dir, f"{spk}__{key}.pt").replace("\\", "/")

                    # 存成 dict，更稳一点
                    torch.save(
                        {
                            "fbank": feat,              # [T,80]
                            "speaker": spk,
                            "spk_id": int(spk2id[spk]),
                            "wav_path": wav_path.replace("\\", "/"),
                        },
                        feat_path,
                    )

                    label = spk2id[spk]

                    if wav_path in val_files:
                        spk_to_utters_val[spk].append(feat_path)
                        fval.write(f"{label} {feat_path}\n")
                        val_meta.append({
                            "speaker": spk,
                            "feat_path": feat_path,
                            "spk_id": int(label),
                        })
                    else:
                        spk_to_utters_train[spk].append(feat_path)
                        ftrain.write(f"{label} {feat_path}\n")

                except Exception as e:
                    print(f"Skip {wav_path}: {e}")
                    continue

    with open(spk_train_path, "w", encoding="utf-8") as f:
        json.dump(spk_to_utters_train, f, ensure_ascii=False, indent=2)

    with open(spk_val_path, "w", encoding="utf-8") as f:
        json.dump(spk_to_utters_val, f, ensure_ascii=False, indent=2)

    with open(spk2id_path, "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    with open(val_meta_path, "w", encoding="utf-8") as f:
        for item in val_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Preprocessing done!")
    print(f"  speakers: {len(spk2id)}")
    print(f"  train speakers with utts: {len(spk_to_utters_train)}")
    print(f"  val speakers with utts: {len(spk_to_utters_val)}")
    print(f"  spk2id: {spk2id_path}")
    print(f"  train list: {train_list_path}")
    print(f"  val list: {val_list_path}")


if __name__ == "__main__":
    main()