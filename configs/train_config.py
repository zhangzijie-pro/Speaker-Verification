from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 数据
    train_list: str = r"processed/cn_celeb2/train_fbank_list.txt"
    val_list: str   = r"processed/cn_celeb2/val_meta.jsonl"

    # 模型
    feat_dim: int = 80
    channels: int = 512
    emb_dim: int = 256  # 192

    # AAM-Softmax
    margin: float = 0.40
    scale: float = 40.0

    # 训练
    epochs: int = 150
    batch_size: int = 128
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 3e-5
    grad_clip: float = 5.0

    # 设备
    device: str = "cuda"
    amp: bool = True

    # 输出
    out_dir: str = "outputs"
    save_best: bool = True
