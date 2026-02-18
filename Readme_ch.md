
---

# 🎙️ 说话人验证和声纹识别

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)

<div align="center">
  <a href="README.md">English</a> • 
  <a href="https://github.com/zhangzijie-pro/Speaker-Verification">GitHub</a> • 
  <a href="https://huggingface.co/zzj-pro">Hugging Face</a>
</div>

> 一个实用且高效的**中文说话人验证系统**，基于 **ECAPA-TDNN + AAM-Softmax**，在 **CN-Celeb** 数据集上训练与评估。

---

## ✨ 核心特性

- **SOTA 主干网络**：ECAPA-TDNN（Res2Net + SE + 注意力统计池化）
- **强判别损失**：AAM-Softmax（带角度边距）
- **均衡采样**：PK Batch Sampler（说话人均衡）
- **完整评估体系**：EER、分数分布、t-SNE、Recall@K
- **稳定推理**：多段裁剪平均（crop-average）
- **低显存优化**：专为 ≈6GB GPU 设计（AMP + 梯度裁剪）

---

## 📂 项目结构

```
Speaker-Verification/
├── CN-Celeb_flac/          # Original CN-Celeb dataset (FLAC/WAV)
│
├── processed/              # Preprocessed features & metadata
│   └── cn_celeb2/
│       ├── fbank_pt/       # Saved fbank features (*.pt)
│       ├── train_fbank_list.txt
│       ├── val_meta.jsonl  # Validation metadata (speaker, feature path)
│       └── spk2id.json
│
├── configs/
│   ├── train.yaml
│   └── train_config.py     # Training hyperparameters
│
├── demos/
│   └── real_time.py        # real time to listen audio to test
│
├── data/
│   ├── dataset.py          # Train / validation datasets
│   └── pk_sampler.py       # PK batch sampler (speaker-balanced)
│
├── speaker_verification/
│   ├── checkpointing.py    
│   ├── inference.py
│   ├── head/
│   │   └── aamsoftmax.py   # AAM-Softmax loss
│   ├── models/
│   │   └── epaca.py        # model
│   └── audio/
│       └── features.py     # extract features
│
├── utils/
│   ├── meters.py           # Accuracy, average meters
│   ├── seed.py             # Reproducibility
│   ├── plot.py             # Training curves
│   └── path_utils.py       # Deal to path error
│
├── scripts/
│   └── export.py           # export onnx/mnn and split model, head
│
├── outputs/                # Training outputs (checkpoints, curves)
├── outputs_eval/           # Verification results (EER, ROC, DET, t-SNE)
│
├── train.py                # Main training script
├── finetune.py             # Main finetune script
├── verify_pairs.py         # Pairwise speaker verification
├── compare_two_wavs.py     # Compare two audio files
│
├── README.md
├── README_ch.md
└── LICENSE
```
---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python processed/preprocess_cnceleb2_train.py
```

### 3. 训练

```bash
# 使用默认配置训练
python train.py

# 命令行覆盖参数
python train.py train.epochs=100 train.lr=5e-4 train.emb_dim=256
```

---

## 📈 评估（说话人验证）

### 完整评估

```bash
python verify.py \
    --val_meta processed/cn_celeb2/val_meta.jsonl \
    --ckpt outputs/best.pt \
    --out_dir outputs_eval
```

**输出文件**：
- `roc.png`、`det.png`、`score_hist.png`
- `tsne.png`（说话人聚类可视化）
- `metrics.txt`（EER、Recall@K 等指标）

---

## 🎯 单条音频对比（最常用场景）

```bash
python compare_two_wavs.py \
    --wav1 test1.wav \
    --wav2 test2.wav \
    --ckpt outputs/export/model.onnx   # 支持 ONNX
```

---

## 🛠️ 模型导出（部署）

```bash
# 一键导出 ONNX + MNN
python scripts/export.py \
    --ckpt outputs/best.pt \
    --out_dir outputs/deploy \
    --onnx --mnn
```

**支持的部署方式**：
- **ONNX Runtime**（Python / C++）
- **MNN**（移动端 / 边缘设备）
- **TensorRT**（高性能服务器）

---

## 🧠 模型概览

### 主干网络

- **ECAPA-TDNN**
  - Res2Net 风格时域卷积
  - Squeeze-and-Excitation（SE）
  - 注意力统计池化（Attentive Statistics Pooling）
- 嵌入维度：**192 / 256**

### 损失函数

- **AAM-Softmax（加性角度边距 Softmax）**
  - 增大说话人之间的角度边距
  - 仅在训练阶段使用

### 嵌入表示

- L2 归一化说话人嵌入
- 使用余弦相似度进行验证

---

## 📊 数据集

- **CN-Celeb**
  - ≈1000 名说话人
  - 录音条件高度多样
- 数据划分：
  - `train`：说话人不重叠的训练集
  - `val`：说话人不重叠的验证集
- 特征：
  - 80 维 log Mel 滤波器组
  - 16kHz 采样率

---

## 📌 推荐配置（6GB GPU）

```yaml
# configs/train.yaml
emb_dim: 256
channels: 512
lr: 1e-3
epochs: 80
crop_frames: 200          # 训练时裁剪长度
crop_frames_val: 400      # 验证时裁剪长度
num_crops: 6
p: 32
k: 4
```

---

## 🔮 未来改进计划

- [x] Hydra 配置管理
- [x] 并行预处理脚本
- [x] ONNX / MNN 导出
- [ ] 噪声 / RIR 数据增强

---

## 📜 开源协议

本项目采用 **Apache License 2.0** 开源协议。  
CN-Celeb 数据集遵循其原始许可协议和使用条款。

---

## 🙋 说明

本仓库主要用于：

- 学习说话人验证系统
- 科研复现与二次开发

**不是开箱即用的商用系统**。
