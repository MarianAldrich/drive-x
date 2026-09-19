"""Export and verify standard ONNX models for the QNX Runtime port."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
import cv2
import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drowsiness.model import TASKS, VideoMobileNet


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models" / "trained"
OUTPUT = ROOT / "qnx" / "models_onnx"


def export_and_verify(model: nn.Module, destination: Path) -> dict[str, float | int]:
    model.eval()
    torch.manual_seed(7)
    sample = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, sample, destination,
        input_names=["images"], output_names=["logits"],
        dynamic_axes=None, opset_version=17, dynamo=False,
    )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    with torch.no_grad():
        expected = model(sample).cpu().numpy()
    network = cv2.dnn.readNetFromONNX(str(destination))
    network.setInput(sample.numpy())
    actual = network.forward()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    return {
        "opset": next(item.version for item in graph.opset_import if item.domain == ""),
        "nodes": len(graph.graph.node),
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    eye_checkpoint = torch.load(CHECKPOINTS / "eye_state" / "best.pt",
                                map_location="cpu", weights_only=True)
    if eye_checkpoint["config"].get("class_to_idx") != {"closed": 0, "open": 1}:
        raise RuntimeError("Unexpected eye model class order")
    eye = models.mobilenet_v3_small(weights=None)
    eye.classifier[-1] = nn.Linear(eye.classifier[-1].in_features, 2)
    eye.load_state_dict(eye_checkpoint["model_state_dict"])

    yawn_checkpoint = torch.load(CHECKPOINTS / "yawn" / "best.pt",
                                 map_location="cpu", weights_only=True)
    if yawn_checkpoint["config"].get("classes") != TASKS["yawn"]:
        raise RuntimeError("Unexpected yawn model class order")
    yawn = VideoMobileNet("yawn", top_fraction=yawn_checkpoint["config"]["top_fraction"])
    yawn.load_state_dict(yawn_checkpoint["model_state_dict"])

    report = {
        "eye_state": export_and_verify(eye, OUTPUT / "eye_state.onnx"),
        "yawn_frame": export_and_verify(yawn.encoder, OUTPUT / "yawn_frame.onnx"),
    }
    face_source = ROOT / "models" / "assets" / "face_detection_yunet_2023mar.onnx"
    shutil.copy2(face_source, OUTPUT / "face_yunet.onnx")
    face = onnx.load(str(OUTPUT / "face_yunet.onnx"))
    onnx.checker.check_model(face)
    report["face_yunet"] = {
        "opset": next(item.version for item in face.opset_import if item.domain == ""),
        "nodes": len(face.graph.node),
        "max_absolute_error": 0.0,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
