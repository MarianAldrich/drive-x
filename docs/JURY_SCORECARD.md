# DRIVE-X jury scorecard and evidence checklist

This document maps every scored category to evidence that can be shown during the
20-minute demonstration. Do not claim a measured value until the Raspberry Pi run
has produced the corresponding CSV or Momentics trace.

| Criterion | Marks | Evidence in DRIVE-X | Live proof |
|---|---:|---|---|
| QNX RTOS Architecture & Usage | 30 | Three native processes, QNX channels and messages, monotonic timing, CPU run masks, `SCHED_FIFO`, supervisor and isolated logger | Show named processes/threads, CPU placement and priorities in Momentics; explain each failure boundary |
| Real-Time Performance & Stability | 20 | Receive-to-decision measurement, 10 ms deadline counter, two-second stale-input state, nonblocking logger, stress and kill utilities | Baseline, CPU stress, UDP interruption and forced decision termination; show p99 and RTO from real logs |
| Innovation & Use-Case Relevance | 10 | Windows AI sensor solves unavailable Pi camera/runtime; QNX remains the deterministic safety control plane | Demonstrate one Ethernet cable carrying timestamped evidence and QNX decisions returning to the DRIVE-X UI |
| Code Quality & Documentation | 20 | Fixed/versioned 40-byte protocol, network byte order, CRC32, malformed/duplicate rejection, modular services, warnings enabled, automated Python tests and deployment runbook | Open the source tree, protocol test output, `JURY_DEMO.md`, CSV analyzer and this traceable checklist |
| Demo & Presentation | 20 | Seven-slide jury-template deck, scripted fault sequence, honest model/test claims and saved profiler trace | Execute the rehearsed flow without editing code; finish with the real profiler screenshot and measured results |

## Evidence to collect on the Pi

- [ ] QNX services compile without errors for `gcc_ntoaarch64le`.
- [ ] `vigil_ingest`, `vigil_decision` and `vigil_supervisor` start successfully.
- [ ] DRIVE-X reports `QNX LINK: Connected`.
- [ ] Baseline CSV contains at least 60 valid decisions.
- [ ] Stress CSV contains at least 60 valid decisions while `vigil_stress --cpu 60` runs.
- [ ] Baseline and stressed p99 values are calculated with `analyze_qnx_log.py`.
- [ ] UDP loss enters `SENSOR_LOST` within two seconds and clears temporal state.
- [ ] Forced decision termination records detection, ready heartbeat and RTO.
- [ ] Momentics screenshot visibly shows thread names, CPU lanes and FIFO priorities.
- [ ] Slide 6 placeholders are replaced only with the real measurements and trace.

## Twenty-minute scoring flow

1. **2 minutes — relevance:** explain fatigue risk and why stale evidence is unsafe.
2. **4 minutes — QNX architecture:** show process boundaries, native IPC, CPU affinity and priorities.
3. **4 minutes — normal operation:** start DRIVE-X, build the window and demonstrate eye, yawn and head cues.
4. **4 minutes — stability:** apply CPU stress, interrupt UDP and show the fail-safe state.
5. **3 minutes — recovery:** terminate the decision service and show restart plus measured RTO.
6. **2 minutes — proof:** open the profiler trace and baseline/stress statistics.
7. **1 minute — conclusion:** explain production path and the deterministic safety-boundary vision.

## Claims that need careful wording

- Eye-state macro-F1 is 95.32% on the held-out MRL split.
- Yawn macro-F1 is 86.15% at the YawDD video-bag level under weak supervision.
- These component metrics are not an end-to-end live-system accuracy claim.
- Windows performs neural inference in this prototype; QNX performs deterministic
  temporal decisions, freshness enforcement, timing, logging and recovery.
