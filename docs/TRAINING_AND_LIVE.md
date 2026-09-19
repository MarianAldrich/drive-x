# Colab training and Windows live detection

## Current state

The live dashboard runs locally at **http://127.0.0.1:8765** when `app/server.py`
is running. Camera preview and face/mouth tracking work without trained models.
Drowsiness/yawn scores remain unavailable until real training checkpoints are installed.

The standalone notebook is `notebooks/Train_Driver_Drowsiness.ipynb`. The coding
assistant cannot enter your authenticated Colab GPU runtime. **Full dataset
training has not yet run, and no real-data accuracy is reported.**

## 1. Start the GPU training run

1. YawDD is configured to come from your uploaded folder
   `1hRWkRD1xKQ0reWvM5o9OSGs1oPhCVSov`. Keep `YawDD.rar.gz` directly in that
   folder and use the Google account with access to it. No public sharing is needed.
2. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload
   notebook**, and select `notebooks/Train_Driver_Drowsiness.ipynb` from this project.
3. Select **Runtime → Change runtime type → T4 GPU**, or another CUDA GPU.
4. Choose **Runtime → Run all** and authorize the Drive mount and authenticated
   dataset read in your browser. The archive is copied to Colab's local storage;
   your original remains in its existing folder. Checkpoints and crop caches are
   still written to `My Drive/DriverDrowsiness`.

If using the already shared Colab notebook, replace its **first code cell** with
the complete contents of `notebooks/Colab_Setup.py`. Its remaining cells work
unchanged. Updating this local project does not modify that remote notebook.
The new setup cell verifies archive size and its Drive MD5 checksum when provided.
It reuses a valid local copy or the existing completed crop cache on reruns.
An interrupted source archive download restarts when the cell is rerun.

### Fix for `download_assets.py returned non-zero exit status 1`

An older notebook version staged YuNet as a file ending in `.download`. OpenCV
selects its model reader from the extension and therefore could not recognize it
as ONNX. In an already-open Colab session, add a temporary code cell immediately
before the failed dependency/setup cell, paste all of `notebooks/Colab_YuNet_Fix.py`,
and run it once. Then rerun the failed dependency/setup cell and continue downward.
This does not require remounting Drive or repeating completed dataset work.

Newly generated copies of `Train_Driver_Drowsiness.ipynb` already contain the
permanent fix and print the underlying asset-setup traceback if another error occurs.

The notebook embeds the project training code. It trains YawDD first, downloads
the official MRL Eyes archive (~342 MB), then trains the eye-state model. Together
with the supplied YawDD archive, source data is approximately 5.7 GB. This replaces
the 119.5 GB UTA branch after its public Drive archives hit download quotas.

If the official MRL server stalls in Colab, upload the existing local
`dataset/mrlEyes_2018_01.zip` to `My Drive/DriverDrowsiness/mrlEyes_2018_01.zip`.
The small-dataset training cell prefers that Drive copy and validates the ZIP before
preparation. Interrupt a stalled download, upload the file, and rerun the cell.

Eye training uses `workers=0` in Colab because multiprocessing DataLoader workers
can stall before the first epoch in some runtimes. If there is no epoch output for
more than 20 minutes, interrupt the cell, confirm CUDA with
`torch.cuda.is_available()`, and rerun the latest small-dataset cell. Each interrupted
Drive run is renamed rather than overwritten.

If interrupted, rerun the notebook. Completed crop archives are restored from
`DriverDrowsiness/prepared_v1`; training resumes from the last completed epoch.
An unfinished source archive download or crop preparation restarts that archive.
If interrupted before the first epoch checkpoint, rename the incomplete
`DriverDrowsiness/training/<task>` folder before restarting that task.
Changing batch size, bag size, seed, or other resume-critical options requires a
new run folder; this prevents silently changing the training experiment.

All source videos are used for preparation, with frames sampled at 1 FPS. Training
samples 16 frames across each training video per epoch; validation/test sampling
is deterministic. This is a video-level baseline, not training on every frame.

## 2. Install the trained artifacts on Windows

At completion, the notebook saves and downloads `trained_models.zip`. Extract it
into `models/trained` so the layout is:

```text
models/trained/
  eye_state/
    best.pt
    config.json
    test_metrics.json
    dataset_report.json  # optional provenance report
    history.json
  yawn/
    best.pt
    config.json
    test_metrics.json
    dataset_report.json  # optional provenance report
    history.json
```

The persistent Drive run folders additionally contain `last.pt` with optimizer,
scheduler, AMP scaler, and random-generator states, plus `status.json`. `best.pt`
is selected only using validation macro F1. The held-out test results are computed
after selection. The confusion matrix uses true classes as rows and predicted
classes as columns. Precision, recall, F1, support, and accuracy are recorded.

The notebook also attempts ONNX export of each frame encoder and compares its
outputs against PyTorch. Only successfully verified ONNX files are bundled.
ONNX failure does not prevent using the PyTorch checkpoints on Windows.
Video pooling and preprocessing specifications accompany the encoder in JSON;
an encoder alone does not implement the complete temporal decision pipeline.

## 3. Run the live interface

From the project root:

```powershell
python -m pip install -r requirements.txt
python scripts/download_assets.py
python app/server.py
```

Open **http://127.0.0.1:8765** in Chrome or Edge and click **Start camera**.
Allow the browser to use your camera. `start_live.ps1` is an alternative launcher
once dependencies are installed. If the server is already running, use its
existing page instead of starting another copy on the same port.

The page shows the live feed, face/mouth crop outlines, alert/low-vigilance/drowsy
evidence, a separate yawn score, relative head movement, inference time, model
status, and session events.
Sound alerts are optional. Model files can be installed while the page is open;
click **Reload models** afterward. No random model is substituted for a missing
or incompatible checkpoint.

One frame per second enters a rolling 8-frame model window. The first prediction
therefore needs approximately 8 seconds of continuous face tracking. Both detected
eyes are classified and averaged. The rolling closed-eye proportion must reach
0.65 for two seconds before an audible/visual break alert.
These initial thresholds are engineering defaults, not calibrated risk thresholds.
A yawn is shown separately and does not by itself trigger a drowsiness alarm.

No face, multiple faces, a large face-position jump, stopped camera, missing models,
and connection gaps reset or suspend prediction. Tracking uses box overlap, not
person recognition: a person change in exactly the same position cannot be reliably
detected. Keep one driver in view and stop/restart between drivers.

Head tracking uses YuNet's eye, nose, and mouth landmarks and needs about five
seconds to learn the driver's neutral pose at session start. Sustained sideways,
vertical, or tilted displacement contributes a low-vigilance cue. It does not
require another trained model or dataset. Head movement alone does not trigger the
high-severity break alarm; that alarm remains tied to sustained eye closure.

The server binds only to loopback. Browser frames are sent to the Python process on
this PC, processed in memory, and not saved. Session events are browser-local.
If camera permission is denied, check the browser and Windows camera permissions.
Another program may also be holding the camera open.

## Dataset and model design

| Branch | Input | Classes | Training supervision | Split |
|---|---|---|---|---|
| Eye state | Individual eye crops | closed, open | MRL image labels | Subjects split approximately 70/20/10 |
| Yawn | Mouth crops | no_yawn, yawn | YawDD video bags; top-quarter positive log-odds | Subjects split approximately 70/20/10 |

Both encoders are ImageNet-initialized **MobileNetV3-Small**, with a replaced
classification layer. The shared YuNet locator and crop geometry are used in
both preparation and inference. Crops are resized to 256×256, converted to RGB,
center-cropped to 224×224, and ImageNet-normalized. Only training uses horizontal
flips and mild brightness/contrast augmentation. The backbone is fine-tuned at
one tenth of the classifier learning rate.

The [MRL Eyes dataset](https://mrl.cs.vsb.cz/eyedataset.html) contains 84,898
eye crops from 37 subjects with explicit open/closed labels. Subject-disjoint
splits prevent the same person appearing in training and test data.

The local YawDD archive contains 349 AVI videos. Its 320 Mirror videos have usable
video labels: 205 no-yawn and 115 containing yawning, across 90 subject identifiers.
Talking-and-yawning clips are positive bags; normal/talking/singing clips are
negative bags. The 29 Dash videos have mixed actions without supported video-level
labels and are excluded. No individual frame in a positive video is automatically
declared a yawn. Preparation records exclusions and face-detection failures.

Weak YawDD video labels limit what can be inferred from individual frames. MRL uses
infrared eye crops, while a normal webcam provides RGB crops; this domain shift must
be evaluated on separate annotated driver recordings. The live alert is a temporal
fusion rule over model evidence, not a directly trained drowsiness probability.

## Verification performed here

- Python compilation and JavaScript syntax checks passed.
- Integration tests cover label parsing, subject leakage, MIL pooling, training,
  checkpoint resume, runtime warmup, face-loss reset, HTTP errors, and notebook syntax.
- The dashboard rendered successfully in headless Edge.
- A real YawDD video produced 13 mouth crops with zero missed samples; crop geometry
  was visually checked.
- Physical webcam operation, full GPU training, real-data accuracy, and ONNX parity
  remain unverified until the Colab run and live-camera test complete.

```powershell
python -m unittest discover -s tests -v
```

Implementation: `drowsiness/` contains the shared model, crops and runtime;
`scripts/prepare_video_dataset.py`, `scripts/colab_prepare.py`, and
`scripts/train_video.py` handle training; `app/` contains the Windows web interface.
After editing training code, regenerate the self-contained notebook with
`python scripts/build_colab_notebook.py`.

References: [YawDD source](https://doi.org/10.21227/e1qm-hb90),
[MobileNetV3-Small API](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html),
[OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
