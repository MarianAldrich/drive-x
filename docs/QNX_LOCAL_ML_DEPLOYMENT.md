# DRIVE-X: ML inference entirely on QNX

In this mode Windows captures and JPEG-compresses the webcam only. QNX receives
the video, decodes it with `libimg`, runs YuNet and both MobileNetV3 networks with
ncnn, calculates the landmark-based head cue, and sends evidence to the QNX
decision service through native synchronous IPC.

## Runtime pipeline

```text
Windows webcam -> JPEG/UDP 45556 -> vigil_video_rx -> /vigil_rgb_frames
  -> vigil_inference (YuNet + eye MobileNet + yawn MobileNet + head landmarks)
  -> MsgSend -> vigil_decision -> logger / alert / GPS
```

The models sample at 1 Hz to match training and the existing temporal logic.
The video receiver can still accept 5 FPS so it always selects a recent frame.

## Deploy

Copy the updated bundle from Windows PowerShell:

```powershell
scp -O -o KexAlgorithms=curve25519-sha256@libssh.org -o Ciphers=aes128-ctr -o MACs=hmac-sha2-256 -r `
  "C:\Users\AldriQNX\Documents\QNX chatgpt\results\qnx_bundle" `
  "qnxuser@192.168.137.97:/data/home/qnxuser/drive-x-ml"
```

On QNX:

```sh
su
cd /data/home/qnxuser/drive-x-ml
chmod 755 run_demo.sh build/vigil_*
mqueue
export VIGIL_GPS_DEVICE=/dev/ser4
export VIGIL_ALERT_HOST=192.168.137.1
export VIGIL_ALERT_PORT=45557
./run_demo.sh
```

The expected startup lines include `stage=video_rx` and
`stage=inference ... runtime=ncnn`. Keep this QNX terminal running.

On Windows, close the browser dashboard if it owns the webcam, then stream video:

```powershell
python scripts\stream_webcam_udp.py --host 192.168.137.97 --port 45556 `
  --fps 5 --width 640 --height 480 --quality 65 --preview
```

The QNX terminal then prints one `INFERENCE` record per second with face count,
eye probability, yawn probability, head pose, inference time, and QNX decision
state. `logs/vigil_events.csv` remains the authoritative decision log.

## Scheduling assignment

| Service | CPU | Policy | Priority / budget |
|---|---:|---|---|
| JPEG receive/decode | 0 | SPORADIC | 25/10, 40 ms per 100 ms |
| QNX ML inference | 3 | SPORADIC | 40/15, 150 ms per 200 ms |
| Evidence ingest | 1 | FIFO | 60 |
| Decision | 2 | FIFO | 58 |
| Emergency alert | 3 | FIFO | 62 |
| Supervisor | 3 | FIFO | 45 |

The inference workload is background work with an explicit CPU budget. Emergency
and supervisor threads can preempt it. The decision path remains isolated on CPU2.

## Acceptance evidence

Do not claim QNX-local ML until the live terminal shows `INFERENCE` records.
Record the first 30 seconds and report median and p99 `inference_ms`. Confirm that
`vigil_events.csv` has `deadline_missed=0` while `vigil_stress` runs.
