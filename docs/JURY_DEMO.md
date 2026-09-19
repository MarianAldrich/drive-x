# VIGIL-QNX jury demo runbook

VIGIL-QNX is a hybrid prototype. Windows performs camera inference; QNX owns the
time-bounded decision, stale-data protection, logging, and recovery. Do not claim
that neural-network inference runs on QNX in this build.

## 1. Build and deploy from Momentics

Import `qnx/` as a QNX C++ make project, select the Raspberry Pi 5 ARM64 target,
and build with the SDP environment active:

```sh
cd qnx
make clean all
```

Deploy the four files from `qnx/build/` to `/data/home/root/vigil/build/`, copy
`qnx/run_demo.sh` to `/data/home/root/vigil/`, and make it executable. The target
account must have permission to set FIFO priorities. The programs fail at startup
if priority or run-mask configuration fails, rather than silently running with a
different scheduling policy.

Core and priority assignment:

| Component | Core | Policy/priority |
|---|---:|---:|
| QNX/network housekeeping | 0 | system |
| `vigil_ingest` | 1 | `SCHED_FIFO 60` |
| `vigil_decision` | 2 | `SCHED_FIFO 58` |
| `vigil_supervisor` | 3 | `SCHED_FIFO 45` |
| logger thread | 3 | `SCHED_FIFO 20` |
| stress utility | 3 | `SCHED_FIFO 10` |

Check for HAM before the demo with `which ham`. The supervisor reports whether it
is installed and always provides the tested fallback restart path.

## 2. Start QNX and Windows

On QNX:

```sh
cd /data/home/root/vigil
chmod +x run_demo.sh build/vigil_*
./run_demo.sh
```

Open UDP port 45555 between the Ethernet hosts. On Windows, install the trained
checkpoints beneath `models/trained/`, then start the sensor with the Pi address:

```powershell
python app/server.py --qnx-host QNX_PI_IPV4 --qnx-port 45555
```

Open `http://127.0.0.1:8765`. QNX LINK changes to Connected after the first valid
reply. QNX latency is receive-to-decision time measured only with the QNX monotonic
clock. Packet age and UDP round-trip are Windows freshness diagnostics; they are
not mixed into the deterministic latency claim.

## 3. Record the four experiments

Run each experiment for at least 60 valid evidence samples and retain its CSV.

1. **Baseline:** normal live operation over Ethernet.
2. **Stress:** `./build/vigil_stress --cpu 60` while live evidence continues.
3. **UDP loss:** `./build/vigil_stress --udp-loss`; verify `SENSOR_LOST` within two seconds and no stale alert.
4. **Recovery:** `./build/vigil_stress --kill-decision`; verify supervisor output reports `decision_rto_us` below 2,000,000.

After each experiment, rename `logs/vigil_events.csv` before the next run. Copy the
files to Windows and calculate presentation statistics:

```powershell
python scripts/analyze_qnx_log.py results/qnx/baseline.csv --output results/qnx/baseline.json
python scripts/analyze_qnx_log.py results/qnx/stress.csv --output results/qnx/stress.json
```

The target is p99 below 10,000 us with zero missed 10 ms deadlines. Report the
observed result even if a target is missed.

## 4. Momentics System Profiler proof

Create a System Profiler launch for the Raspberry Pi target and capture the
baseline, CPU-stress, UDP-loss, and process-restart intervals in one trace. The
mandatory screenshot must visibly include:

- the QNX target and timeline;
- `vigil_ingest`, `vigil_decision`, `vigil_supervisor`, and logger activity;
- CPU 1/2/3 placement;
- FIFO priorities 60/58/45/20;
- the stress interval and continued decision execution.

Save the trace and screenshot under `results/qnx/profiler/`. Replace the placeholder
on slide 6 only after this real capture exists.

## 5. Twenty-minute demonstration

1. Show QNX running and the Momentics target connection.
2. Start the supervisor and show its service PIDs.
3. Start the Windows camera; wait for the 8-sample window.
4. Demonstrate alert, low-vigilance, and sustained-closure states.
5. Run CPU stress and point to stable QNX latency.
6. Inject UDP loss and show `SENSOR_LOST` without a stale drowsiness alert.
7. Kill the decision service and show automatic restart plus measured RTO.
8. Open the saved profiler trace and the baseline/stress statistics.

The YawDD classifier uses video-level weak supervision. Describe its score as yawn
evidence, not a calibrated medical or safety probability.
