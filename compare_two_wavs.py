import argparse
from pathlib import Path

import torch

from speaker_verification.inference import load_sv, cosine_score

try:
    import onnxruntime as ort  # noqa: F401
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False
    print("⚠️ onnxruntime 未安装，ONNX 模式不可用。")
    print("   pip install onnxruntime  或  onnxruntime-gpu")


def main():
    parser = argparse.ArgumentParser(description="两个音频说话人对比（PyTorch / ONNX）")

    parser.add_argument("--wav1", type=str, required=True, help="第一个音频路径")
    parser.add_argument("--wav2", type=str, required=True, help="第二个音频路径")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/export/model.onnx",
        help="模型路径：.pt (PyTorch) 或 .onnx (ONNX)",
    )

    parser.add_argument(
        "--onnx",
        action="store_true",
        default=False,
        help="使用 ONNX 推理（默认自动根据文件后缀判断）",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="判断同一人的余弦相似度阈值（建议通过 verify.py 得到最佳阈值）",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="PyTorch 模式使用的设备（ONNX 默认按 onnxruntime providers）",
    )

    parser.add_argument("--num_crops", type=int, default=5, help="多 crop 平均的 crop 数")
    parser.add_argument("--crop_sec", type=float, default=3.0, help="每个 crop 的时长（秒）")

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    use_onnx = args.onnx or ckpt_path.suffix.lower() == ".onnx"

    print("=" * 70)
    print("🎙️  Speaker Verification - Two Wavs Comparison")
    print("=" * 70)
    print(f"Audio 1  : {args.wav1}")
    print(f"Audio 2  : {args.wav2}")
    print(f"Model    : {ckpt_path}  ({'ONNX' if use_onnx else 'PyTorch'})")
    print(f"Threshold: {args.threshold}")
    print(f"Crops    : {args.num_crops}  |  Crop_sec: {args.crop_sec}")
    print("=" * 70)

    if use_onnx and not _HAS_ONNX:
        raise ImportError("请先安装 onnxruntime: pip install onnxruntime 或 onnxruntime-gpu")

    backend_device = "cpu" if use_onnx else (args.device if torch.cuda.is_available() else "cpu")
    sv, meta = load_sv(str(ckpt_path), device=backend_device, use_onnx=use_onnx)

    if use_onnx:
        print(f"使用 ONNX Runtime 推理... providers={meta.get('providers')}")
    else:
        print(f"使用 PyTorch 推理... device={backend_device}")

    score = cosine_score(
        sv,
        args.wav1,
        args.wav2,
        num_crops=args.num_crops,
        crop_sec=args.crop_sec,
    )

    same = score >= args.threshold

    print(f"\n🔍 Cosine Similarity = {score:.4f}")
    print(f"   Threshold        = {args.threshold}")
    print(f"   → {'同一说话人' if same else '不同说话人'}")

    color = "\033[92m" if same else "\033[91m"
    print(f"\n{color}【最终判定】{'✅ 同一人' if same else '❌ 不同人'}\033[0m")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()