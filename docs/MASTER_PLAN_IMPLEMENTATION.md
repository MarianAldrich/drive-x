# DRIVE-X master-plan implementation

## Implemented and buildable

- Existing MobileNetV3 checkpoints, classes, preprocessing, and weights are unchanged.
- Windows camera frames can be sent as chunked JPEG over UDP 45556.
- `vigil_video_rx` validates frame CRC, decodes with QNX `libimg`, and publishes
  RGB888 into the two-slot `/vigil_rgb_frames` shared-memory buffer.
- `vigil_frame_monitor` independently verifies frame metadata and RGB CRC.
- The existing evidence path retains UDP validation, QNX native synchronous IPC,
  temporal decision processing, stale-input protection, nonblocking logging,
  shared telemetry, deadline measurement, process supervision, and recovery RTO.
- Prolonged drowsiness is outside the model and is configurable with
  `VIGIL_EMERGENCY_MS` (default 10000 ms after entering `DROWSY`).
- `vigil_gps` parses checksum-valid `$GPRMC`/`$GNRMC` data from a configurable UART
  and publishes a seqlock-protected GPS snapshot.
- `vigil_alert` consumes a compact POSIX queue event at FIFO priority 62, combines
  it with the latest GPS fix, logs it, and optionally sends it to a UDP HTTPS/SMS
  gateway on Windows.
- The dashboard shows a QNX-state-driven simulated vehicle speed and emergency state.

## QNX-local inference boundary

The installed Raspberry Pi QNX image includes `libimg` but does not contain ONNX
Runtime, TensorFlow Lite, OpenCV DNN, or another runtime capable of executing the
existing MobileNetV3 checkpoints. Therefore RGB acquisition and the QNX real-time
control pipeline are complete, but claiming that the trained neural network runs on
QNX would be false until a compatible ARM64 QNX runtime is added and measured.

The shared RGB interface is the stable boundary for that final integration. A future
runtime adapter must read `/vigil_rgb_frames`, reproduce the existing preprocessing,
execute exported unchanged weights, and send only its small result to
`vigil_decision`. It must be validated against the Windows/PyTorch outputs before the
Windows evidence publisher is removed.

## Configuration

Before starting the services on QNX:

```sh
export VIGIL_GPS_DEVICE=/dev/ser4
export VIGIL_EMERGENCY_MS=5000
export VIGIL_ALERT_HOST=192.168.137.1
export VIGIL_ALERT_PORT=45557
```

Use 5000 ms for a short project demonstration. Use a validated safety requirement for
real deployment. If the GPS device is absent, `vigil_gps` remains alive, reports that
it is waiting, and the alert record explicitly says `gps=NO_FIX`.

Start the Windows emergency gateway before the QNX services:

```powershell
python scripts/emergency_relay.py --port 45557
```

Without credentials it prints and logs the message. To send an actual SMS, configure
`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and
`DRIVEX_EMERGENCY_NUMBER` in that PowerShell session. Credentials and phone numbers
are never compiled into the QNX application.

## Demonstration sequence

1. Start `mqueue`, then `sh ./run_demo.sh` on QNX.
2. Start `vigil_video_rx` and the Windows JPEG sender; run `vigil_frame_monitor`.
3. Start DRIVE-X with `--qnx-host` and begin live detection.
4. Sustain a QNX `DROWSY` decision until the configurable emergency timer expires.
5. Show `EMERGENCY ACTIVE`, the GPS coordinates, `vigil_emergency.csv`, the gateway
   message, and the dashboard speed falling toward zero.
6. Apply CPU stress and capture the priorities, affinities, sporadic budgets, and
   decision execution in Momentics System Profiler.

