# Jury Submission Manifest

## Start here

1. Read `README.md` for the project summary and reproducible run path.
2. Open `deliverables/DRIVE-X_Detailed_Project_Documentation.docx` for the design report.
3. Open `deliverables/DRIVE-X_Final_Presentation_Profiler_Complete.pptx` for the jury presentation.
4. View `evidence/` for the QNX System Profiler captures.

## Repository layout

| Folder | Contents |
|---|---|
| `qnx/` | QNX C++ source, headers, ONNX Runtime build scripts, and preconverted `.ort` models |
| `app/`, `drowsiness/` | Windows dashboard, protocol, and model-side source |
| `models/` | Required YuNet asset and trained eye/yawn checkpoints |
| `scripts/` | Camera streaming, emergency relay, model export, and training helpers |
| `docs/` | Deployment, CPU/scheduling, and demonstration runbooks |
| `evidence/` | QNX System Profiler images |
| `deliverables/` | Final presentation and complete project documentation |
| `notebooks/`, `tests/` | Reproducible training notebooks and test source |

## Intentionally excluded

Datasets, temporary logs, local Python environments, IDE state, QNX build binaries, ONNX Runtime source/build trees, and host tooling are excluded because they are generated locally or are machine-specific.
