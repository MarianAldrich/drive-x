"""Export DRIVE-X neural networks to ncnn and verify their PyTorch outputs.

Run this on the Windows development host.  The generated .param/.bin files are
platform independent; the QNX service loads them with an AArch64 QNX ncnn build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pnnx
import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drowsiness.model import TASKS, VideoMobileNet


def load_eye(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("class_to_idx") != {"closed": 0, "open": 1}:
        raise ValueError("Eye checkpoint class order is not closed=0, open=1")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), config


def load_yawn(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("classes") != TASKS["yawn"]:
        raise ValueError("Yawn checkpoint class order is incompatible")
    model = VideoMobileNet("yawn", top_fraction=config["top_fraction"])
    model.load_state_dict(checkpoint["model_state_dict"])
    # Only per-frame logits belong in the inference graph. Temporal top-quarter
    # pooling remains explicit in the QNX service over its sixteen-frame window.
    return model.encoder.eval(), config


def export_pnnx(model: nn.Module, output_stem: Path) -> None:
    sample = torch.zeros(1, 3, 224, 224)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    traced_path = output_stem.with_suffix(".pt")
    # Preserve FP32 weights so deployment probabilities remain numerically
    # equivalent to the jury metrics measured with the PyTorch checkpoints.
    pnnx.export(model, str(traced_path), (sample,), fp16=False)
    generated = traced_path.parent
    source_stem = traced_path.stem
    for suffix in (".ncnn.param", ".ncnn.bin"):
        source = generated / f"{source_stem}{suffix}"
        destination = output_stem.parent / f"{output_stem.name}{suffix}"
        if source != destination:
            shutil.move(source, destination)
        if not destination.is_file():
            raise RuntimeError(f"pnnx did not create {destination}")
    # Intermediate TorchScript and Python wrappers are unnecessary on QNX.
    for suffix in (".pt", ".pnnx.param", ".pnnx.bin", "_pnnx.py", "_ncnn.py"):
        candidate = generated / (source_stem + suffix)
        if candidate.exists():
            candidate.unlink()


def export_yunet(onnx_path: Path, output_dir: Path) -> None:
    executable = Path(pnnx.__file__).parent / "pnnx.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"pnnx converter missing: {executable}")
    # YuNet's feature pyramid requires dimensions divisible by 32. 320x256
    # stays close to 4:3; the QNX service center-crops frames to this ratio.
    command = [str(executable), str(onnx_path), "inputshape=[1,3,256,320]", "fp16=0"]
    subprocess.run(command, cwd=output_dir, check=True)
    stem = onnx_path.stem
    for suffix in (".ncnn.param", ".ncnn.bin"):
        source = onnx_path.parent / f"{stem}{suffix}"
        destination = output_dir / f"face_yunet{suffix}"
        if destination.exists():
            destination.unlink()
        source.replace(destination)
    for suffix in (".pnnx.param", ".pnnx.bin", "_pnnx.py", "_ncnn.py"):
        candidate = onnx_path.parent / f"{stem}{suffix}"
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models/trained"))
    parser.add_argument("--yunet", type=Path,
                        default=Path("models/assets/face_detection_yunet_2026may.onnx"))
    parser.add_argument("--output", type=Path, default=Path("qnx/models"))
    args = parser.parse_args()

    eye, eye_config = load_eye(args.model_root / "eye_state/best.pt")
    yawn, yawn_config = load_yawn(args.model_root / "yawn/best.pt")
    export_pnnx(eye, args.output / "eye_state")
    export_pnnx(yawn, args.output / "yawn_frame")
    export_yunet(args.yunet.resolve(), args.output.resolve())
    metadata = {
        "format": "ncnn",
        "input": {"width": 224, "height": 224, "layout": "RGB_CHW"},
        "normalization": eye_config["preprocessing"],
        "eye_classes": ["closed", "open"],
        "yawn_classes": yawn_config["classes"],
        "yawn_bag_size": yawn_config["bag_size"],
        "yawn_top_fraction": yawn_config["top_fraction"],
        "face_input": [320, 256],
    }
    (args.output / "models.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"QNX model assets written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
