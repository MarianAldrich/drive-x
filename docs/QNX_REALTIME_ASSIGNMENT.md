# DRIVE-X QNX real-time assignment

## Runtime pipeline

| Stage | QNX process | Core | Scheduling | Budget / deadline | Communication |
|---|---|---:|---|---|---|
| Emergency alert | `vigil_alert` | 3 | `SCHED_FIFO 62` | event-driven; highest application priority | POSIX emergency queue, UDP gateway |
| Ethernet validation | `vigil_ingest` | 1 | `SCHED_FIFO 60` | forwards each valid sample immediately | UDP in, native `MsgSend` out |
| Temporal safety decision | `vigil_decision` | 2 | `SCHED_FIFO 58` | 10 ms receive-to-decision deadline | `MsgReceive/MsgReply` |
| Supervision | `vigil_supervisor` | 3 | `SCHED_FIFO 45` | 500 ms heartbeat, 2 s failure threshold | heartbeats/process monitoring |
| Face/eye/yawn/head inference | `vigil_inference` | 3 | `SCHED_SPORADIC 40/15` | 150 ms budget every 200 ms; 1 Hz sample | RGB shared buffer, native `MsgSend` |
| NEO-7M location | `vigil_gps` | 3 | `SCHED_SPORADIC 35/12` | 5 ms budget every 100 ms | UART NMEA, shared GPS snapshot |
| CSV/console logging | `vigil_logger` | 3 | `SCHED_SPORADIC 20/8` | 5 ms budget every 100 ms | nonblocking POSIX queue |
| JPEG/RGB input | `vigil_video_rx` | 0 | `SCHED_SPORADIC 25/10` | 40 ms budget every 100 ms | UDP JPEG, double RGB shared buffer |
| Controlled load | `vigil_stress` | 3 | `SCHED_SPORADIC 10/4` | 80 ms budget every 100 ms | none |
| QNX/network housekeeping | OS | 0 | system | isolated from application work | network stack |

The critical path uses native synchronous QNX IPC, which provides bounded handoff
and priority inheritance. Disk output is outside that path. The decision process
uses `mq_send(..., O_NONBLOCK)`; a full queue increments the drop counter instead
of delaying a decision.

`SCHED_SPORADIC` enforces execution budgets for noncritical work. When a process
uses its high-priority budget it falls to its configured low priority until the
budget is replenished. Ingest and decision instead get isolated cores and fixed
priorities. Every decision compares its QNX monotonic completion time with
`receive_ns + 10 ms`; the CSV records each miss and shared telemetry holds the
cumulative count. This is fixed-priority scheduling with deadline monitoring,
not EDF.

The decision process writes a pointer-free snapshot to `/dev/shmem/vigil_telemetry`.
A generation counter implements a seqlock so readers cannot accept a torn update.
`vigil_monitor` demonstrates a separate process reading this shared buffer.

The video receiver publishes decoded RGB888 frames to `/dev/shmem/vigil_rgb_frames`
using two fixed slots and a generation seqlock. `vigil_frame_monitor` verifies the
latest frame metadata and pixel CRC without copying frames through a message queue.
The NEO-7M reader independently publishes `/dev/shmem/vigil_gps`. After continuous
`DROWSY` state for `VIGIL_EMERGENCY_MS`, the decision task sends a compact event to
`/dev/mqueue/vigil_emergency`; the FIFO-62 alert task reads the latest GPS fix and
emits the emergency record without putting network or disk I/O on the decision path.

## Build and deploy

In the QNX SDP command prompt on Windows:

```bat
cd /d C:\VIGIL-QNX_QNX_Source
make clean all
scp build/vigil_* root@PI_IP:/data/home/root/drive-x/
scp -r models root@PI_IP:/data/home/root/drive-x/
scp run_demo.sh root@PI_IP:/data/home/root/drive-x/
```

On the Pi, as root:

```sh
cd /data/home/root/drive-x
chmod +x build/vigil_* run_demo.sh
mqueue
./run_demo.sh
```

If `mqueue` reports that it is already running, continue. The services deliberately
fail startup if QNX refuses their priority, affinity, or sporadic budget; fix the
process abilities or run as root rather than silently losing real-time behavior.

## Demonstrate each jury feature

Use another Pi terminal:

```sh
# Shared buffer snapshot and measured deadline count
./build/vigil_monitor

# Decoded RGB double-buffer snapshot
./build/vigil_frame_monitor

# Show POSIX queue and QNX shared-memory objects
ls -l /dev/mqueue/vigil_log /dev/shmem/vigil_telemetry

# Budgeted noncritical CPU load
./build/vigil_stress --cpu 30

# Three-second input outage; SENSOR_LOST must occur within two seconds
./build/vigil_stress --udp-loss

# Supervisor restart and RTO measurement
./build/vigil_stress --kill-decision

# Timing evidence
tail -20 logs/vigil_events.csv
cat logs/vigil_recovery.csv
tail -20 logs/vigil_emergency.csv
```

Start the Windows sensor with:

```powershell
.\start_live.ps1 -QnxHost PI_IP
```

In Momentics System Profiler capture `vigil_video_rx`, `vigil_inference`,
`vigil_ingest`, `vigil_decision`, `vigil_supervisor`, `vigil_logger`, and
`vigil_stress`. Show their run masks,
FIFO/sporadic policies, effective priorities, decision wakeups, and CPU3 load.

## Acceptance evidence

- `vigil_events.csv`: filter `deadline_missed=1`; target zero and p99 below 10 ms.
- `vigil_monitor`: deadline-miss and queue-drop totals should remain zero.
- UDP loss: console changes to `SENSOR_LOST` within two seconds and the temporal
  window is cleared.
- Decision kill: `vigil_recovery.csv` reports RTO below 2,000,000 microseconds.
- CPU stress: ingest and decision continue on CPUs 1 and 2 while budgeted stress
  remains on CPU 3.

Adaptive Partitioning Scheduler is not claimed here because it depends on the Pi
image being booted with the APS scheduler. The implemented CPU budgets use QNX
`SCHED_SPORADIC`, which is present in QNX SDP 8 and can be shown directly in the
profiler.
