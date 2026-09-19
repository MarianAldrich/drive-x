# QNX Edge AI Vision Project — Raspberry Pi 5

## 1. Project Overview

### Project Goal

Build a real-time edge AI vision application on a **Raspberry Pi 5 running QNX Neutrino RTOS**.

The final system should:

1. Capture an image or video frame.
2. Preprocess the frame.
3. Run an AI model locally on the Raspberry Pi 5 CPU.
4. Detect or classify the target condition.
5. Pass results between QNX processes using IPC.
6. Display the result.
7. Log detections and system performance.
8. Optionally trigger an alert or action.

### Target Architecture

```text
Camera / Image / Video
        |
        v
+----------------------+
| Camera/Input Process |
+----------------------+
        |
      QNX IPC
        |
        v
+----------------------+
|  AI Inference Process|
+----------------------+
        |
      QNX IPC
        |
        v
+----------------------+
|   Decision Process   |
+----------------------+
      |           |
      v           v
+-----------+ +-----------+
|    UI     | |  Logger   |
+-----------+ +-----------+
```

---

# 2. Recommended Development Strategy

Do **not** start with live camera integration.

Build the project in this order:

```text
QNX system
    |
    v
Static image
    |
    v
AI model inference
    |
    v
Video file
    |
    v
USB camera
    |
    v
QNX IPC architecture
    |
    v
UI + logging
    |
    v
Performance optimization
    |
    v
CSI Camera Module 2 integration
```

This prevents camera-driver issues from blocking the entire project.

---

# 3. Hardware

## Required Hardware

- Raspberry Pi 5
- Raspberry Pi 5 power supply
- microSD card
- Ethernet connection
- HDMI monitor
- USB keyboard
- Development computer
- USB storage device if required

## Camera Options

Preferred order:

1. USB UVC webcam
2. Image files
3. Video files
4. Raspberry Pi Camera Module 2 / IMX219

The CSI camera should not be the first dependency for the AI pipeline.

---

# 4. Software Requirements

## Raspberry Pi 5

- QNX Neutrino RTOS
- Raspberry Pi 5 QNX BSP
- QNX Momentics / SDP tools
- SSH
- Networking tools
- C/C++ runtime
- Image handling library
- AI inference runtime

## Development Machine

Recommended:

- Linux or Windows
- Python 3
- PyTorch or TensorFlow
- OpenCV
- ONNX
- ONNX Runtime
- Git
- CMake

Optional:

- Docker
- VS Code
- Jupyter Notebook

---

# 5. Repository Structure

Create the project repository like this:

```text
qnx-edge-ai/
|
|-- README.md
|
|-- docs/
|   |-- architecture.md
|   |-- setup.md
|   |-- qnx_setup.md
|   |-- ai_model.md
|   `-- demo.md
|
|-- models/
|   |-- original/
|   |-- exported/
|   `-- optimized/
|
|-- datasets/
|   |-- train/
|   |-- val/
|   `-- test/
|
|-- src/
|   |
|   |-- input/
|   |   |-- image_loader.cpp
|   |   |-- video_loader.cpp
|   |   `-- camera_capture.cpp
|   |
|   |-- inference/
|   |   |-- model_loader.cpp
|   |   |-- preprocess.cpp
|   |   |-- inference.cpp
|   |   `-- postprocess.cpp
|   |
|   |-- decision/
|   |   `-- decision_engine.cpp
|   |
|   |-- ipc/
|   |   |-- ipc_server.cpp
|   |   `-- ipc_client.cpp
|   |
|   |-- logger/
|   |   `-- logger.cpp
|   |
|   `-- ui/
|       `-- ui.cpp
|
|-- scripts/
|   |-- train.py
|   |-- export_onnx.py
|   |-- benchmark.py
|   `-- deploy.sh
|
|-- test_images/
|
|-- test_videos/
|
`-- results/
    |-- logs/
    |-- benchmarks/
    `-- screenshots/
```

---

# 6. Phase 1 — QNX Platform Bring-Up

## Objective

Get the Raspberry Pi 5 QNX environment stable before adding AI.

## Tasks

- [ ] Flash the QNX Raspberry Pi 5 image.
- [ ] Boot QNX successfully.
- [ ] Verify HDMI output.
- [ ] Verify keyboard input.
- [ ] Verify Ethernet.
- [ ] Configure IP networking.
- [ ] Verify SSH access.
- [ ] Verify file transfer from development PC.
- [ ] Verify compiler/toolchain deployment.
- [ ] Verify system date.
- [ ] Set timezone.
- [ ] Make timezone configuration persistent.
- [ ] Verify system time synchronization.
- [ ] Document all working commands.

## Target Result

You should be able to:

```text
Development PC
      |
      | SSH / SCP
      v
Raspberry Pi 5
      |
      v
QNX shell
```

## Deliverable

Create:

```text
docs/qnx_setup.md
```

Document:

- boot process
- IP address
- SSH command
- file copy command
- timezone setup
- time synchronization
- application execution
- environment variables

---

# 7. Phase 2 — Basic QNX Application

## Objective

Confirm that custom applications can run correctly.

Create a simple C++ program.

Example:

```cpp
#include <iostream>

int main()
{
    std::cout << "QNX Edge AI project started" << std::endl;
    return 0;
}
```

## Tasks

- [ ] Compile for Raspberry Pi 5 target architecture.
- [ ] Copy executable to QNX.
- [ ] Run executable.
- [ ] Verify output.
- [ ] Create a repeatable build process.

## Target

```text
QNX Edge AI project started
```

---

# 8. Phase 3 — Define the AI Use Case

Before training anything, define exactly what the system detects.

Possible examples:

- person detection
- helmet detection
- PPE detection
- vehicle detection
- driver monitoring
- defect detection
- occupancy detection
- gesture recognition
- object classification

## Define

```text
Project use case:

Input:

Output:

Number of classes:

Required accuracy:

Required FPS:

Required maximum latency:

Alert condition:
```

## Example

```text
Use case:
Helmet detection

Input:
Camera image

Classes:
helmet
no_helmet

Output:
Detected class + confidence

Target latency:
< 200 ms

Minimum target FPS:
5 FPS
```

---

# 9. Phase 4 — Dataset Preparation

## Objective

Create a clean dataset for transfer learning.

Recommended split:

```text
70% training
20% validation
10% testing
```

## Tasks

- [ ] Collect images.
- [ ] Remove duplicate or unusable images.
- [ ] Label images.
- [ ] Split dataset.
- [ ] Confirm class balance.
- [ ] Create augmentation strategy.

Possible augmentation:

- horizontal flip
- brightness changes
- contrast changes
- crop
- scale
- slight rotation

Avoid unrealistic augmentation.

---

# 10. Phase 5 — Select the AI Model

Since the Raspberry Pi 5 will perform CPU inference, choose a lightweight model.

## Classification

Recommended candidates:

- MobileNetV3
- MobileNetV2
- EfficientNet-Lite
- ResNet18

## Object Detection

Recommended candidates:

- SSD MobileNet
- lightweight YOLO nano model
- lightweight EfficientDet variant

## Model Selection Criteria

Evaluate:

- accuracy
- model size
- CPU inference time
- memory consumption
- ease of export
- QNX runtime compatibility

Do not choose the largest model simply because it has higher accuracy.

---

# 11. Phase 6 — Train Using Transfer Learning

Do training on the development PC or cloud machine.

Do **not** train on the Raspberry Pi 5.

Workflow:

```text
Pretrained Model
      |
      v
Replace final layer
      |
      v
Train on custom dataset
      |
      v
Validate
      |
      v
Test
      |
      v
Export model
```

## Tasks

- [ ] Load pretrained model.
- [ ] Replace output layer.
- [ ] Freeze appropriate layers initially.
- [ ] Train.
- [ ] Validate.
- [ ] Fine-tune.
- [ ] Save best checkpoint.
- [ ] Evaluate on test dataset.

## Record

```text
Training accuracy:
Validation accuracy:
Test accuracy:
Precision:
Recall:
F1 score:
Model size:
```

---

# 12. Phase 7 — Export the Model

Preferred portable format:

```text
ONNX
```

Workflow:

```text
PyTorch / TensorFlow
        |
        v
      ONNX
        |
        v
Inference Runtime
```

## Tasks

- [ ] Export the trained model.
- [ ] Run ONNX checker.
- [ ] Verify input tensor shape.
- [ ] Verify output tensor shape.
- [ ] Compare original model output with ONNX output.
- [ ] Save exported model under:

```text
models/exported/
```

---

# 13. Phase 8 — Verify AI Inference on Development PC

Before touching QNX inference, make sure the exported model works.

Pipeline:

```text
test.jpg
   |
   v
image loading
   |
   v
resize
   |
   v
normalize
   |
   v
ONNX inference
   |
   v
postprocessing
   |
   v
result
```

Expected output example:

```text
Class: Helmet
Confidence: 94.2%
Inference: 18 ms
```

## Tasks

- [ ] Load ONNX model.
- [ ] Load test image.
- [ ] Preprocess image.
- [ ] Run inference.
- [ ] Postprocess output.
- [ ] Print result.
- [ ] Verify accuracy against expected result.

---

# 14. Phase 9 — Select QNX AI Runtime

This is one of the main technical checkpoints.

Investigate the following options:

## Option 1

ONNX Runtime for QNX / ARM64 if supported or buildable.

## Option 2

TensorFlow Lite if supported or portable to the QNX environment.

## Option 3

Other inference SDK/runtime available in your QNX environment.

## Validation Requirements

The runtime must support:

- ARM64 / AArch64
- CPU inference
- required model operators
- QNX compilation
- C or C++ API

## Deliverable

Document:

```text
Runtime selected:
Version:
Build method:
Required libraries:
Model format:
Known limitations:
```

---

# 15. Phase 10 — Static Image Inference on QNX

This is the first major project milestone.

Do not integrate a camera yet.

Target:

```text
test.jpg
   |
   v
Raspberry Pi 5
   |
   v
QNX
   |
   v
AI model
   |
   v
prediction
```

## Application Structure

```cpp
int main()
{
    load_model();

    auto image = load_image("test.jpg");

    auto input = preprocess(image);

    auto output = run_inference(input);

    auto result = postprocess(output);

    print_result(result);

    return 0;
}
```

## Expected Output

```text
Model loaded successfully

Input:
test.jpg

Detected:
Helmet

Confidence:
93.8%

Inference time:
105 ms
```

## Tasks

- [ ] Model loads.
- [ ] Image loads.
- [ ] Preprocessing works.
- [ ] Inference executes.
- [ ] Result is correct.
- [ ] Timing is measured.

---

# 16. Phase 11 — Benchmark QNX Inference

Measure performance early.

Record:

- model loading time
- preprocessing time
- inference time
- postprocessing time
- total latency
- FPS
- CPU usage
- RAM usage
- temperature
- model size

## Benchmark Table

| Metric | Result |
|---|---:|
| Model size | |
| Input resolution | |
| Preprocessing | |
| Inference | |
| Postprocessing | |
| Total latency | |
| FPS | |
| RAM | |
| CPU | |

## Initial Target

For a hackathon prototype:

```text
FPS:
5+ FPS

Inference:
< 200 ms

Model size:
< 50 MB

Resolution:
224x224 to 640x640 depending on model
```

These are engineering targets, not hard guarantees.

---

# 17. Phase 12 — Video File Input

Before a physical camera, test using a prerecorded video.

Pipeline:

```text
test.mp4
   |
   v
decode frame
   |
   v
AI inference
   |
   v
result
```

## Tasks

- [ ] Open video.
- [ ] Extract frame.
- [ ] Run inference.
- [ ] Process multiple frames.
- [ ] Measure FPS.
- [ ] Skip frames if necessary.

Possible strategy:

```text
process every frame
```

or:

```text
process every second / third frame
```

depending on inference speed.

---

# 18. Phase 13 — USB Camera Integration

Use a USB UVC camera before returning to CSI.

Target:

```text
USB Webcam
    |
    v
Raspberry Pi 5
    |
    v
QNX camera/input layer
    |
    v
frame
```

## Tasks

- [ ] Connect camera.
- [ ] Verify USB detection.
- [ ] Verify camera device.
- [ ] Capture single frame.
- [ ] Save captured image.
- [ ] Confirm image quality.
- [ ] Capture continuous frames.
- [ ] Feed frames into inference engine.

Do not proceed until single-frame capture works reliably.

---

# 19. Phase 14 — Live AI Inference

Combine camera and AI.

Main loop:

```cpp
while (running)
{
    Frame frame = capture_frame();

    Tensor input = preprocess(frame);

    Output output = inference(input);

    Result result = postprocess(output);

    display_result(result);

    log_result(result);
}
```

## Target Output

```text
Frame: 1512
Detection: Person
Confidence: 96%
Inference: 92 ms
FPS: 9.8
```

---

# 20. Phase 15 — QNX Process Architecture

Once basic live inference works, split the application into processes.

Recommended structure:

```text
+-------------------+
| camera_service    |
+-------------------+
          |
          | QNX IPC
          v
+-------------------+
| inference_service |
+-------------------+
          |
          | QNX IPC
          v
+-------------------+
| decision_service  |
+-------------------+
      |        |
      |        |
      v        v
+---------+ +---------+
|   UI    | | logger  |
+---------+ +---------+
```

Benefits:

- fault isolation
- modular design
- easier debugging
- clearer QNX-specific architecture
- process restart capability
- real-time prioritization

---

# 21. Phase 16 — QNX IPC

Use QNX native IPC/message passing where practical.

Example messages:

## Camera to Inference

```text
FRAME_READY
frame_id
timestamp
buffer_reference
width
height
format
```

## Inference to Decision

```text
INFERENCE_RESULT
frame_id
class_id
confidence
inference_time
timestamp
```

## Decision to Logger/UI

```text
EVENT
event_type
severity
timestamp
details
```

## Tasks

- [ ] Implement IPC prototype.
- [ ] Send simple text message.
- [ ] Send structured message.
- [ ] Add error handling.
- [ ] Add timestamps.
- [ ] Measure IPC latency.

---

# 22. Phase 17 — Decision Engine

The decision engine converts model output into useful actions.

Example:

```text
if no_helmet detected
AND confidence > 0.80
THEN
    create alert
    update UI
    write event log
```

## Important

Do not trigger events on every noisy frame.

Add:

- confidence threshold
- consecutive-frame confirmation
- cooldown period

Example:

```text
Detection threshold:
0.80

Required consecutive detections:
3

Alert cooldown:
5 seconds
```

---

# 23. Phase 18 — Logging

Store both detection and performance data.

Recommended CSV format:

```text
timestamp,event,class,confidence,inference_ms,fps
2026-09-16T20:30:15,DETECTION,person,0.947,104,8.9
```

Save logs under:

```text
results/logs/
```

## Log

- timestamp
- class
- confidence
- event type
- latency
- FPS
- errors

---

# 24. Phase 19 — User Interface

Keep the UI simple.

Example:

```text
+------------------------------------------+
|         QNX EDGE AI MONITOR              |
+------------------------------------------+
| Camera        ONLINE                     |
| Model         RUNNING                    |
|                                          |
| Detection     PERSON                     |
| Confidence    94.2%                      |
|                                          |
| Inference     105 ms                     |
| FPS           9.4                        |
|                                          |
| System        NORMAL                     |
+------------------------------------------+
```

Optional live features:

- camera preview
- bounding boxes
- confidence values
- FPS
- CPU usage
- event count
- alert state

---

# 25. Phase 20 — Performance Optimization

Only optimize after the complete pipeline works.

Recommended order:

1. Reduce image resolution.
2. Use a smaller model.
3. Reduce unnecessary frame copies.
4. Pre-allocate buffers.
5. Optimize image preprocessing.
6. Add threading where safe.
7. Adjust QNX scheduling priorities.
8. Add CPU affinity if useful.
9. Quantize model.
10. Skip frames when needed.

Possible model optimization:

```text
FP32
 |
 v
FP16
 |
 v
INT8
```

Benchmark after every major optimization.

---

# 26. Phase 21 — Reliability Tests

Test failure conditions.

## Camera Failure

What happens if the camera disconnects?

Expected:

```text
camera_service detects failure
logger records event
system retries connection
AI service remains alive
```

## AI Failure

What happens if model loading fails?

## IPC Failure

What happens if one process crashes?

## Invalid Frame

What happens if image data is corrupted?

## Tasks

- [ ] Camera disconnect test.
- [ ] Camera reconnect test.
- [ ] Inference service restart.
- [ ] Invalid image test.
- [ ] High CPU usage test.
- [ ] Long-duration test.

---

# 27. Phase 22 — CSI Camera Module 2 / IMX219

Only return to this after the rest of the system works.

Your known camera path includes:

```text
Raspberry Pi Camera Module 2
IMX219
CSI
QNX camera/sensor service
```

Debug it as an isolated subsystem.

Do not modify the working inference pipeline while debugging CSI.

Target first:

```text
IMX219
  |
  v
single frame capture
  |
  v
save image
```

Only after that:

```text
IMX219
  |
  v
continuous frames
  |
  v
AI inference
```

If CSI cannot be stabilized within the hackathon schedule, keep the USB webcam as the demonstration camera.

---

# 28. Phase 23 — Final Demo Architecture

Final target:

```text
                  Raspberry Pi 5
                 QNX Neutrino RTOS

             +--------------------+
Camera ----> |   camera_service   |
             +--------------------+
                       |
                     QNX IPC
                       |
                       v
             +--------------------+
             | inference_service  |
             +--------------------+
                       |
                     QNX IPC
                       |
                       v
             +--------------------+
             |  decision_service  |
             +--------------------+
                  |            |
                  |            |
                  v            v
            +----------+  +----------+
            |    UI    |  |  Logger  |
            +----------+  +----------+
```

---

# 29. Final Demo Flow

The hackathon demonstration should be simple and reliable.

## Demo Sequence

1. Boot Raspberry Pi 5.
2. Show QNX running.
3. Start project services.
4. Show camera online.
5. Place target object/person in camera view.
6. Show AI detection.
7. Show confidence value.
8. Show inference time/FPS.
9. Show generated event.
10. Show event in log.
11. Explain QNX process separation and IPC.
12. Demonstrate failure isolation if time allows.

---

# 30. Suggested Project Milestones

## Milestone 1 — Platform Ready

- [ ] QNX boots.
- [ ] Network works.
- [ ] SSH works.
- [ ] Deployment works.
- [ ] Time configuration works.

## Milestone 2 — AI Model Ready

- [ ] Dataset prepared.
- [ ] Transfer learning complete.
- [ ] Model evaluated.
- [ ] ONNX export works.

## Milestone 3 — QNX AI Ready

- [ ] Runtime works.
- [ ] Static image inference works.
- [ ] Benchmark captured.

## Milestone 4 — Input Pipeline Ready

- [ ] Video works.
- [ ] USB camera works.
- [ ] Continuous frames work.

## Milestone 5 — Live AI Ready

- [ ] Camera + inference works.
- [ ] FPS measured.
- [ ] Detection stable.

## Milestone 6 — QNX Architecture Ready

- [ ] Separate processes.
- [ ] QNX IPC works.
- [ ] Decision engine works.
- [ ] Logger works.

## Milestone 7 — Demo Ready

- [ ] UI works.
- [ ] Automatic launch works.
- [ ] Logs work.
- [ ] Recovery behavior tested.
- [ ] Presentation/demo flow rehearsed.

---

# 31. Suggested Day-by-Day Build Plan

## Day 1 — Platform

- QNX boot
- network
- SSH
- deployment
- environment configuration

## Day 2 — AI Training

- finalize use case
- dataset preparation
- transfer learning
- model testing

## Day 3 — Model Export

- ONNX export
- PC inference
- preprocessing verification
- output verification

## Day 4 — QNX Inference

- inference runtime
- static image
- benchmark

## Day 5 — Input Pipeline

- video input
- USB camera
- continuous frames

## Day 6 — Live AI

- camera + model
- thresholds
- performance measurements

## Day 7 — QNX IPC

- camera service
- inference service
- decision service
- IPC

## Day 8 — UI and Logging

- UI
- event logging
- timestamps
- status reporting

## Day 9 — Optimization

- model size
- resolution
- quantization
- buffers
- process priorities

## Day 10 — Final Integration

- boot-to-demo
- reliability testing
- demo script
- results collection
- screenshots/videos

Adjust the schedule to the actual hackathon duration.

---

# 32. Priority System

## Must Have

- QNX on Raspberry Pi 5
- AI inference
- static image support
- camera or video input
- live detection
- measurable inference performance
- result logging

## Should Have

- QNX IPC
- independent services
- live UI
- USB camera
- event decision logic

## Nice to Have

- IMX219 CSI camera
- INT8 optimization
- automatic process recovery
- sophisticated GUI
- cloud connectivity
- dashboard

Do not sacrifice the working demo for a nice-to-have feature.

---

# 33. Definition of Done

The project is complete when:

```text
Raspberry Pi 5 boots QNX
        |
        v
AI services start
        |
        v
image/video/camera input received
        |
        v
model inference runs locally
        |
        v
detection/classification produced
        |
        v
result passed through QNX architecture
        |
        +----> UI
        |
        +----> logger
        |
        +----> optional alert
```

The final demo should be reproducible after a reboot.

---

# 34. Final Presentation Story

A concise technical description:

> This project implements a real-time edge AI vision system on a Raspberry Pi 5 running QNX Neutrino RTOS. A lightweight transfer-learned AI model performs local CPU inference without requiring cloud processing or a discrete GPU. The application is divided into independent QNX processes for image capture, inference, decision logic, visualization, and logging. QNX message passing connects the services while maintaining process isolation and enabling a modular real-time architecture.

---

# 35. Immediate Next Actions

Start with these tasks only.

## Step 1

Create the repository:

```bash
mkdir qnx-edge-ai
cd qnx-edge-ai
```

## Step 2

Create:

```text
docs/
models/
datasets/
src/
scripts/
test_images/
test_videos/
results/
```

## Step 3

Confirm the Raspberry Pi 5 QNX environment:

- [ ] boot
- [ ] networking
- [ ] SSH
- [ ] file copy
- [ ] compilation
- [ ] execution

## Step 4

Finalize the exact AI use case.

## Step 5

Before using any camera, prove:

```text
test.jpg -> model -> prediction
```

on the development PC.

## Step 6

Port the same inference pipeline to QNX.

---

# 36. Core Rule for the Project

Always debug one layer at a time.

```text
Hardware
   |
Operating System
   |
Input
   |
Preprocessing
   |
Inference
   |
Postprocessing
   |
IPC
   |
Decision Logic
   |
UI / Logging
```

Never debug several layers simultaneously.

The first important goal is **not** live CSI camera AI.

The first important goal is:

```text
QNX + Raspberry Pi 5 + test image + AI model = correct inference
```

Once that works, add one subsystem at a time.
