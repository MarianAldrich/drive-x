# DRIVE-X

DRIVE-X is a Raspberry Pi 5 and QNX 8.0 driver-drowsiness monitoring prototype. A Windows computer sends webcam JPEG frames only; QNX receives and decodes the frames, runs local ONNX Runtime inference, makes vigilance decisions, and supervises alert, GPS, and logging services.

## Project structure

- `app/` — Windows dashboard application
- `configs/` — runtime and model configuration
- `docs/` — deployment, architecture, and real-time design documentation
- `drowsiness/` — detection and decision components
- `evidence/` — System Profiler captures
- `models/` — model assets and trained checkpoints
- `notebooks/` — model-development notebooks
- `qnx/` — QNX services, build files, ONNX Runtime support, and `.ort` models
- `scripts/` — camera streaming, model export, training, and utility scripts
- `tests/` — automated tests
- `deliverables/` — presentation and detailed project documentation

## Architecture

`Windows webcam → JPEG/UDP 45556 → QNX video receiver/libimg → RGB shared memory → ONNX inference → QNX decision → alert/GPS/logger`

## Measured results

- ONNX Runtime 1.18.1 smoke tests passed for face, eye, and yawn models on QNX ARM64.
- Live local QNX inference: 237–256 ms per frame.
- JPEG decoding: 7–11 ms at 640×480.
- System Profiler trace: 5.000 s, 1,197,822 events, and 0 dropped buffers.
- `vigil_ort_inference`: 1.549 s running time in the captured trace.

## Run locally

On Windows, stream a camera feed to the target:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python scripts\stream_webcam_udp.py --host QNX_PI_IPV4 --port 45556 --fps 1 --width 640 --height 480 --preview
```

On QNX, after building and deploying the services and `qnx/models_ort/`:

```sh
export VIGIL_LOCAL_INFERENCE=1
export VIGIL_INFERENCE_RUNTIME=ort
export VIGIL_GPS_DEVICE=/dev/null
sh ./run_demo.sh
```

For build and deployment details, see `qnx/onnxruntime-qnx/README.md` and `docs/QNX_LOCAL_ML_DEPLOYMENT.md`.
