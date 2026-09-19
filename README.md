# DRIVE-X — QNX Driver Drowsiness Monitoring

DRIVE-X is a Raspberry Pi 5 / QNX 8.0 driver-monitoring prototype. Windows provides a webcam JPEG stream only; QNX receives and decodes it, runs local ONNX Runtime inference, makes the vigilance decision, and supervises alert, GPS, and logging services.

## What is included

- Windows camera/dashboard and training source
- QNX C++ source for video receive, inference, decision, alert, GPS, logger, and supervisor
- Preconverted QNX ONNX Runtime `.ort` models for YuNet face, eye state, and yawn
- Trained PyTorch checkpoints and YuNet asset used for model export
- QNX ONNX Runtime build scripts and QNX platform patch
- Deployment/runbook documents, profiler evidence, final documentation, and presentation

## Architecture

`Windows webcam → JPEG/UDP 45556 → QNX video RX/libimg → RGB shared memory → ONNX inference → QNX decision → alert/GPS/logger`

## Validated results

- QNX ONNX Runtime 1.18.1 smoke tests passed for face, eye, and yawn models.
- Live QNX local inference: **237–256 ms/frame**.
- QNX JPEG decode: **7–11 ms** at 640×480.
- System Profiler trace: **5.000 s**, **1,197,822 events**, **0 dropped buffers**.
- `vigil_ort_inference`: **1.549 s** running time in the trace.

## Windows development setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python scripts\stream_webcam_udp.py --host QNX_PI_IPV4 --port 45556 --fps 1 --width 640 --height 480 --preview
```

## QNX deployment

The prebuilt QNX executable is intentionally not committed. Build it with the QNX SDP 8.0 ARM64 toolchain after cloning ONNX Runtime, applying `qnx/onnxruntime-qnx/qnx-platform.patch`, and following `qnx/onnxruntime-qnx/README.md`. Deploy the resulting `vigil_ort_inference`, the other `vigil_*` services, and `qnx/models_ort/` to the Pi bundle.

On QNX, start the local inference pipeline:

```sh
export VIGIL_LOCAL_INFERENCE=1
export VIGIL_INFERENCE_RUNTIME=ort
export VIGIL_GPS_DEVICE=/dev/null
sh ./run_demo.sh
```

See `docs/QNX_LOCAL_ML_DEPLOYMENT.md`, `docs/QNX_REALTIME_ASSIGNMENT.md`, and `deliverables/DRIVE-X_Detailed_Project_Documentation.docx` for the complete process.

## Repository hygiene

Datasets, local environments, build outputs, ONNX Runtime source/build directories, logs, and personal IDE state are excluded through `.gitignore`.
