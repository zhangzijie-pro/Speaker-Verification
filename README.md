
---

# 🎙️ Sound-Groove: Speaker Verification with ECAPA-TDNN


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/zzj-pro/CN_Celeb_v2)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)



<div align="center">

[中文](README_ch.md) | [English](README.md)

</div>

> A practical speaker verification system based on **ECAPA-TDNN + AAM-Softmax**, trained and evaluated on **CN-Celeb**.
---

## 📌 Features

* ✅ ECAPA-TDNN backbone (Res2Net + SE + Attentive Statistics Pooling)
* ✅ AAM-Softmax loss for discriminative speaker embeddings
* ✅ Speaker-balanced PK sampling
* ✅ Verification-oriented evaluation (EER, score distribution, t-SNE)
* ✅ Crop-average inference for stable embeddings
* ✅ Designed for **limited GPU memory (≈6GB)**

---

## 📂 Project Structure

```
sound-groove-model/
├── CN-Celeb_flac/          # Original CN-Celeb dataset (FLAC/WAV)
│
├── processed/              # Preprocessed features & metadata
│   └── cn_celeb2/
│       ├── fbank_pt/       # Saved fbank features (*.pt)
│       ├── train_fbank_list.txt
│       ├── val_meta.jsonl  # Validation metadata (speaker, feature path)
│       └── spk2id.json
│
├── configs/
│   └── train_config.py     # Training hyperparameters
│
├── data/
│   ├── dataset.py          # Train / validation datasets
│   ├── pk_sampler.py       # PK batch sampler (speaker-balanced)
│   └── ...
│
├── models/
│   └── ecapa.py            # ECAPA-TDNN implementation
│
├── loss/
│   └── aamsoftmax.py       # AAM-Softmax loss
│
├── utils/
│   ├── meters.py           # Accuracy, average meters
│   ├── seed.py             # Reproducibility
│   ├── plot.py             # Training curves
│   └── ...
│
├── outputs/                # Training outputs (checkpoints, curves)
├── outputs_eval/           # Verification results (EER, ROC, DET, t-SNE)
│
├── train.py                # Main training script
├── verify_pairs.py         # Pairwise speaker verification
├── compare_two_wavs.py     # Compare two audio files
├── split_pt.py / turn.py   # Utility / debug scripts
│
├── README.md
├── README_ch.md
└── LICENSE
```

---

## 🧠 Model Overview

### Backbone

* **ECAPA-TDNN**

  * Res2Net-style temporal convolutions
  * Squeeze-and-Excitation (SE)
  * Attentive Statistics Pooling
* Output embedding dimension: **192 / 256**

### Loss

* **AAM-Softmax (Additive Angular Margin Softmax)**

  * Encourages large inter-speaker margin
  * Used only during training

### Embedding

* L2-normalized speaker embeddings
* Cosine similarity used for verification

---

## 📊 Dataset

* **CN-Celeb**

  * ~1000 speakers
  * Diverse recording conditions
* Data split:

  * `train`: speaker-disjoint training set
  * `val`: speaker-disjoint validation set
* Features:

  * 80-dim log Mel-filterbank
  * 16kHz sampling rate

---

## 🔧 Preprocessing

1. Convert audio to mono 16kHz
2. Extract fbank features using `torchaudio.compliance.kaldi.fbank`
3. Save features as `.pt` files
4. Generate:

   * `train_fbank_list.txt`
   * `val_meta.jsonl`

Each training sample:

```
<label> <absolute_path_to_fbank.pt>
```

---

## 🚀 Training

### Run training

```bash
python train.py
```

### Key training strategies

* **PK Sampling**

  * P speakers × K utterances per speaker
  * Example: `P=32, K=4` → batch size = 128
* **Random temporal cropping**

  * Training crop: ~2s (`crop_frames=200`)
* **AMP (mixed precision)** enabled
* **Gradient clipping** for stability

---

## 📈 Evaluation (Speaker Verification)

### Metrics

* **EER (Equal Error Rate)** – primary metric
* Score distribution (same vs diff)
* t-SNE visualization of embeddings
* Recall@K (optional, sampled)

### Validation strategy

* **Longer crops + multi-crop average**

  * `crop_frames = 400`
  * `num_crops = 5~10`
* Embeddings are averaged and L2-normalized

### Run verification

```bash
python verify_pairs.py
```

Outputs:

* `outputs_eval/roc.png`
* `outputs_eval/det.png`
* `outputs_eval/score_hist.png`
* `outputs_eval/tsne.png`

---

## 🧪 Example Results (CN-Celeb)

* Training accuracy: **>80%**
* Validation EER (sampled): **~20–25%**
* Clear separation between same / different speaker scores
* t-SNE shows clustered speaker embeddings

> Note: ECAPA-TDNN is **slow to converge**. Significant EER improvement often appears after 40–80 epochs.

---

## 🛠️ Recommended Hyperparameters (6GB GPU)

```python
emb_dim = 256
P = 32
K = 4
crop_frames_train = 200
crop_frames_val = 400
num_crops_val = 10
margin = 0.30 → 0.35 (later epochs)
scale = 30 → 35
epochs = 60–150
```

---

## 🔍 Known Limitations

* Validation EER still sensitive to crop length
* No explicit noise / reverberation augmentation (yet)
* CN-Celeb intra-speaker variability is high

---

## 📌 Future Improvements

* [ ] SpecAugment on fbank
* [ ] Noise / RIR augmentation
* [ ] Hard negative mining
* [ ] Adaptive margin scheduling
* [ ] ONNX / TensorRT inference export

---

## 📜 License

This project is released under the **Apache License**.
CN-Celeb dataset follows its original license and usage terms.

---

## 🙋 Notes

This repository is intended for:

* Learning speaker verification systems

It is **not** an off-the-shelf commercial system.

---