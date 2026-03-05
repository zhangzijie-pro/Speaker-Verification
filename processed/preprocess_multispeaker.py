# processed/preprocess_cnceleb2_train.py
import os
import json
import random
import torch
import torchaudio
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
    VAL_SPK_RATIO = 0.1
    SEED = 1234

CFG = PrepConfig()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

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
    spk2files = defaultdict(list)
    for spk in os.listdir(data_dir):
        spk_path = os.path.join(data_dir, spk)
        if os.path.isdir(spk_path):
            for f in os.listdir(spk_path):
                if f.lower().endswith((".flac", ".wav")):
                    spk2files[spk].append(os.path.join(spk_path, f))

    spks = sorted(spk2files.keys())
    random.shuffle(spks)
    n_val = int(len(spks) * CFG.VAL_SPK_RATIO)
    val_spks = set(spks[:n_val])
    train_spks = [s for s in spks if s not in val_spks]

    train_spk2id = {spk: i for i, spk in enumerate(train_spks)}
    spk_to_utters = defaultdict(list)
    val_meta = []

    train_list_path = os.path.join(CFG.OUT_DIR, "train_fbank_list.txt")
    spk_utter_path = os.path.join(CFG.OUT_DIR, "spk_to_utterances.json")
    val_meta_path = os.path.join(CFG.OUT_DIR, "val_meta.jsonl")

    with open(train_list_path, "w", encoding="utf-8") as ftrain:
        for spk in tqdm(train_spks + list(val_spks), desc="Preprocessing CN-Celeb2"):
            is_val = spk in val_spks
            for wav_path in spk2files[spk]:
                try:
                    wav = load_wav_mono(wav_path, target_sr=CFG.TARGET_SR)
                    if len(wav) < CFG.TARGET_SR * CFG.MIN_SEC:
                        continue
                    
                    fbank = wav_to_fbank(
                        wav,
                        n_mels=CFG.N_MELS,
                        num_crops=1,
                        crop_sec=max(CFG.MIN_SEC, float(len(wav)) / CFG.TARGET_SR),  # 让 crop 覆盖整段音频
                    )[0]
                    key = safe_key(wav_path, CFG.CN_ROOT)
                    feat_path = os.path.join(feat_dir, f"{spk}__{key}.pt").replace("\\", "/")
                    torch.save(fbank, feat_path)

                    spk_to_utters[spk].append(feat_path)

                    if not is_val:
                        label = train_spk2id[spk]
                        ftrain.write(f"{label} {feat_path}\n")
                    else:
                        val_meta.append({"speaker": spk, "feat_path": feat_path, "spk_id": train_spk2id.get(spk, -1)})
                except Exception as e:
                    print(f"Skip {wav_path}: {e}")
                    continue

    with open(spk_utter_path, "w", encoding="utf-8") as f:
        json.dump(spk_to_utters, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CFG.OUT_DIR, "train_spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(train_spk2id, f, ensure_ascii=False, indent=2)
    with open(val_meta_path, "w", encoding="utf-8") as f:
        for item in val_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Preprocessing done! {len(spk_to_utters)} speakers | val_meta.jsonl: {len(val_meta)} utterances")

if __name__ == "__main__":
    main()