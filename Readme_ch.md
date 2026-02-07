
---

# 🎙️ Sound-Groove：基于 ECAPA-TDNN 的说话人验证模型


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/zzj-pro/CN_Celeb_v2)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)



<div align="center">

[中文](README_zh.md) | [English](README.md)

</div>

> 本项目实现了一个基于 **ECAPA-TDNN + AAM-Softmax** 的说话人验证（Speaker Verification）系统，
> 使用 **CN-Celeb** 数据集进行训练与评估。


---

## 📌 项目特点

* ✅ ECAPA-TDNN 主干网络（Res2Net + SE + ASP）
* ✅ AAM-Softmax 判别损失，增强类间角度间隔
* ✅ PK 采样（按说话人均衡采样）
* ✅ 面向验证任务的评估流程（EER / 分数分布 / t-SNE）
* ✅ 多裁剪平均（crop-avg）推理，提升验证稳定性
* ✅ 可在 **6GB 显存** 下稳定训练

---

## 📂 目录结构说明

```
sound-groove-model/
├── CN-Celeb_flac/          # 原始 CN-Celeb 数据集（FLAC/WAV）
│
├── processed/              # 预处理后的特征与索引
│   └── cn_celeb2/
│       ├── fbank_pt/       # 保存的 fbank 特征 (*.pt)
│       ├── train_fbank_list.txt
│       ├── val_meta.jsonl  # 验证集元信息（speaker, feature path）
│       └── spk2id.json
│
├── configs/
│   └── train_config.py     # 训练超参数配置
│
├── data/
│   ├── dataset.py          # 训练 / 验证数据集定义
│   ├── pk_sampler.py       # PK 采样器（按说话人）
│   └── ...
│
├── models/
│   └── ecapa.py            # ECAPA-TDNN 网络实现
│
├── loss/
│   └── aamsoftmax.py       # AAM-Softmax 损失函数
│
├── utils/
│   ├── meters.py           # 准确率 / 平均值统计
│   ├── seed.py             # 随机种子控制
│   ├── plot.py             # 训练曲线绘制
│   └── ...
│
├── outputs/                # 训练输出（模型、日志、曲线）
├── outputs_eval/           # 验证结果（EER、ROC、DET、t-SNE）
│
├── train.py                # 主训练脚本
├── verify_pairs.py         # 说话人对验证（EER 计算）
├── compare_two_wavs.py     # 两段语音相似度对比示例
├── split_pt.py / turn.py   # 工具 / 调试脚本
│
├── README.md
├── README_ch.md
└── LICENSE
```

---

## 🧠 模型结构说明

### 主干网络（Backbone）

* **ECAPA-TDNN**

  * 多尺度 Res2Net 时序卷积
  * Squeeze-and-Excitation（SE）模块
  * Attentive Statistics Pooling（ASP）
* 输出 embedding 维度：**192 / 256**

### 损失函数（Training only）

* **AAM-Softmax**

  * 在角度空间引入 margin
  * 强化说话人之间的判别边界

### 推理方式

* 输出 embedding 做 **L2 归一化**
* 使用 **余弦相似度（cosine similarity）** 做说话人验证

---

## 📊 数据集说明

* **CN-Celeb**

  * 约 1000 名说话人
  * 多场景、多设备、多说话风格
* 数据划分：

  * `train`：训练集（说话人不重叠）
  * `val`：验证集（说话人不重叠）
* 特征：

  * 80 维 Mel-filterbank
  * 采样率 16kHz

---

## 🔧 数据预处理流程

1. 音频转为 **16kHz 单声道**
2. 使用 `torchaudio.compliance.kaldi.fbank` 提取 fbank
3. 将特征保存为 `.pt` 文件
4. 生成索引文件：

   * `train_fbank_list.txt`
   * `val_meta.jsonl`

训练集列表格式：

```
<label> <absolute_path_to_fbank.pt>
```

---

## 🚀 模型训练

### 启动训练

```bash
python train.py
```

### 关键训练策略

* **PK 采样**

  * P 个说话人 × 每人 K 条语音
  * 示例：`P=32, K=4` → batch=128
* **随机裁剪**

  * 训练阶段裁剪约 2 秒（`crop_frames=200`）
* **AMP 混合精度训练**
* **梯度裁剪** 防止数值不稳定

---

## 📈 验证与评估（Speaker Verification）

### 评估指标

* **EER（Equal Error Rate）**：核心指标
* Same / Diff 分数分布
* t-SNE 可视化
* Recall@K（采样评估）

### 验证策略

* **长裁剪 + 多裁剪平均**

  * `crop_frames = 400`
  * `num_crops = 5~10`
* 多段 embedding 平均后再归一化

### 执行验证

```bash
python verify_pairs.py
```

输出结果位于：

```
outputs_eval/
├── roc.png
├── det.png
├── score_hist.png
└── tsne.png
```

---

## 🧪 实验效果（CN-Celeb）

* 训练分类准确率：**80%+**
* 验证集 EER（采样）：**约 20–25%**
* Same / Diff 分数分布明显分离
* t-SNE 中同一说话人呈现聚类结构

> 注意：ECAPA-TDNN **收敛较慢**，通常在 40–80 epoch 后 EER 才会明显下降。

---

## 🛠️ 推荐训练配置（6GB 显存）

```python
emb_dim = 256
P = 32
K = 4
crop_frames_train = 200
crop_frames_val = 400
num_crops_val = 10
margin = 0.30 → 0.35（后期）
scale = 30 → 35
epochs = 60–200
```

---

## ⚠️ 当前局限

* 验证 EER 对裁剪长度仍较敏感
* 尚未加入噪声 / 混响数据增强

---

## 🔮 后续改进方向

* [ ] SpecAugment（fbank 级别）
* [ ] 噪声 / RIR 混合增强
* [ ] Hard Negative Mining
* [ ] 动态 margin / scale 调度
* [ ] ONNX / TensorRT 推理部署

---

## 📜 许可证

本项目采用 **Apache License**。
CN-Celeb 数据集遵循其原始数据许可协议。

---

## 🙋 说明

本项目主要用于：

* 说话人识别 / 验证学习

**并非直接可商用系统**。

---