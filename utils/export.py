import os
import time
import json
import argparse
import subprocess
from pathlib import Path

import torch
import onnx

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from speaker_verification.models.ecapa import ECAPA_TDNN


class Config:
    def __init__(self):
        self.__start_time = time.time()
        self.min_memory = 1024 * 1024 * 1024  # 1GB

    def get_spend_time(self):
        seconds = int(time.time() - self.__start_time)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}h {m:02d}m {s:02d}s"

    def get_device(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        max_free = -1
        best_gpu = 0
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
            if free > max_free:
                max_free = free
                best_gpu = i

        free_mb = max_free / (1024 * 1024)
        if free_mb < 1024:
            print(f"⚠️ GPU {best_gpu} 剩余 {free_mb:.1f}MB → 使用 CPU")
            return torch.device("cpu")
        return torch.device(f"cuda:{best_gpu}")


class ModelExporter:
    """统一模型导出类"""
    def __init__(self, ckpt_path: str, out_dir: str = "outputs/export"):
        self.ckpt_path = Path(ckpt_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.config = Config()
        self.device = self.config.get_device()

        self.model = None
        self.head = None
        self.emb_dim = 256
        self.channels = 512

        print(f"[{self.__class__.__name__}] Device: {self.device}")

    def split_checkpoint(self, save_head: bool = True):
        """拆分 best.pt → model.pt + head.pt"""
        print("正在拆分 checkpoint...")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        model_state = ckpt["model"]
        head_state = ckpt.get("head", None)

        model_path = self.out_dir / "model.pt"
        torch.save(model_state, model_path)
        print(f"✓ model saved → {model_path}")

        if save_head and head_state is not None:
            head_path = self.out_dir / "head.pt"
            torch.save(head_state, head_path)
            print(f"✓ head saved → {head_path}")

        meta = {
            "emb_dim": self.emb_dim,
            "channels": self.channels,
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_ckpt": str(self.ckpt_path),
        }
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def load_model(self):
        print("加载 ECAPA-TDNN...")
        self.model = ECAPA_TDNN(
            in_channels=80,
            channels=self.channels,
            embd_dim=self.emb_dim
        ).to(self.device)

        state_dict = torch.load(self.out_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("✓ 模型加载完成")

    def export_onnx(self, dummy_batch: int = 1, dummy_frames: int = 400, opset: int = 17):
        self.load_model()

        dummy_input = torch.randn(dummy_batch, dummy_frames, 80).to(self.device)

        onnx_path = self.out_dir / "model.onnx"
        print(f"正在导出 ONNX → {onnx_path} (opset={opset})")

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["fbank"],
            output_names=["embedding"],
            dynamic_axes={
                "fbank": {0: "batch_size", 1: "time_frames"},
                "embedding": {0: "batch_size"}
            },
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX 导出成功！文件大小: {onnx_path.stat().st_size / (1024*1024):.1f} MB")

    def export_mnn(self, fp16: bool = True, optimize: int = 1):
        """导出 MNN（需先安装 mnn: pip install mnn）"""
        if not (self.out_dir / "model.onnx").exists():
            print("请先执行 --onnx")
            return

        mnn_path = self.out_dir / "model.mnn"
        fp16_flag = "--fp16" if fp16 else ""
        cmd = f"mnnconvert -f ONNX --modelFile {self.out_dir}/model.onnx " \
              f"--MNNModel {mnn_path} {fp16_flag} --optimizePrefer {optimize}"

        print(f"正在转换 MNN → {mnn_path}")
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"✓ MNN 导出成功！")
        except subprocess.CalledProcessError as e:
            print("MNN 转换失败（请确认已安装 mnn）")
            print(e.stderr.decode())


def main():
    parser = argparse.ArgumentParser(description="Speaker Verification 模型导出工具")
    
    parser.add_argument("--ckpt", type=str, required=True,
                        help="输入 checkpoint 路径 (best.pt)")
    
    parser.add_argument("--out_dir", type=str, default="outputs/export",
                        help="导出目录 (默认: outputs/export)")
    
    parser.add_argument("--split", action="store_true", default=True,
                        help="是否拆分 checkpoint (默认: True)")
    
    parser.add_argument("--onnx", action="store_true", default=True,
                        help="是否导出 ONNX (默认: True)")
    
    parser.add_argument("--mnn", action="store_true", default=False,
                        help="是否导出 MNN (需先 --onnx)")
    
    parser.add_argument("--dummy_batch", type=int, default=1,
                        help="ONNX dummy batch size")
    
    parser.add_argument("--dummy_frames", type=int, default=400,
                        help="ONNX dummy frames (约4秒音频)")
    
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="MNN 使用 FP16")

    args = parser.parse_args()

    exporter = ModelExporter(ckpt_path=args.ckpt, out_dir=args.out_dir)

    start_time = time.time()

    if args.split:
        exporter.split_checkpoint()

    if args.onnx:
        exporter.export_onnx(
            dummy_batch=args.dummy_batch,
            dummy_frames=args.dummy_frames,
            opset=args.opset
        )

    if args.mnn:
        exporter.export_mnn(fp16=args.fp16)

    print("=" * 60)
    print(f"导出完成！耗时: {exporter.config.get_spend_time()}")
    print(f"输出目录: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()