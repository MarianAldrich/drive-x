# Experimental ONNX Runtime port for QNX 8 ARM64

This directory contains an isolated CPU port of ONNX Runtime **v1.18.1**.
It is a bring-up experiment, not yet a replacement for `vigil_inference`.
The existing pipeline uses its existing runtime until integration is tested.

## Design

- Cross-compile on Windows with QNX SDP 8 GCC 12.2.
- Minimal static runtime, CPU execution provider, baseline ARM64/NEON kernels.
- Convert standard ONNX models once on Windows into ORT format using the
  matching 1.18.1 runtime. Basic graph optimization avoids host-specific kernels.
- Run with one inference thread and no automatic CPU affinity. QNX scheduling
  and runmask integration can be added after numerical validation.
- No GPU backend, automatic CPU feature detection, or runtime stack unwinding.

## Sources and configuration

ONNX Runtime: `https://github.com/microsoft/onnxruntime`, tag `v1.18.1`.
Eigen: `https://gitlab.com/libeigen/eigen.git`, commit
`e7248b26a1ed53fa030c5c459f7ea095dfd276ac` (the commit pinned by this ORT release).
Place these at `qnx/third_party/onnxruntime` and `qnx/third_party/eigen-ort`.
Apply `qnx-platform.patch` to the ORT checkout if using a fresh checkout.

In a Windows **Command Prompt**, from the repository root:

```bat
call C:\Users\AldriQNX\qnx800\qnxsdp-env.bat
powershell -NoProfile -ExecutionPolicy Bypass -File qnx\onnxruntime-qnx\configure-minimal.ps1
set MAKEFLAGS=
"C:\Program Files\CodeBlocks\MinGW\bin\cmake.exe" --build qnx\third_party\onnxruntime\build-qnx-minimal --parallel 6
powershell -NoProfile -ExecutionPolicy Bypass -File qnx\onnxruntime-qnx\build-smoke.ps1
```

Paths to CMake and the SDK are specific to this workstation. Configuration
exports public Windows trusted root certificates to a local PEM file for CMake
downloads; TLS verification stays enabled. Eigen uses a pinned Git checkout
because the original archive no longer matched the pinned archive checksum.

## Models and reference vectors

`scripts/export_onnx_models.py` exports the trained eye and per-frame yawn models
and copies YuNet to `qnx/models_onnx`. This requires the existing training Python
environment. The yawn export represents the per-frame network; temporal
aggregation remains application logic.

Run `prepare_models.py` with Python containing ONNX Runtime 1.18.1 and NumPy:

```powershell
& qnx/onnxruntime-qnx/host-tools/python311/python.exe qnx/onnxruntime-qnx/prepare_models.py
```

The isolated host tools are under `host-tools/`; they do not replace the system
Python. The output `qnx/models_ort` contains models, deterministic float inputs,
reference outputs, and `verification.json`.

## Pi validation

Copy `ort_smoke` and the entire `qnx/models_ort` directory to a new test directory
on the QNX Pi. From that directory:

```sh
chmod 755 ort_smoke
./ort_smoke models_ort/eye_state
./ort_smoke models_ort/yawn_frame
./ort_smoke models_ort/face_yunet
```

Each run loads a model, executes a saved tensor, checks all outputs, and prints
`ORT result=PASS` or `FAIL` with `inference_us`. Comparison tolerance is
`0.002 + 0.002 * abs(reference)` per element to allow CPU floating-point variation.
Nonfinite values always fail. Exit status is zero only on numerical success.
This test proves model execution and numerical agreement on a synthetic input;
it does not measure detection accuracy or sustained video throughput.

Do not enable the replacement pipeline until all three tests pass on QNX.
Next integration steps are RGB preprocessing, YuNet output decoding, eye/mouth
crops, temporal yawn aggregation, and the existing decision IPC. Validate these
against identical recorded frames on Windows before using a live camera.
