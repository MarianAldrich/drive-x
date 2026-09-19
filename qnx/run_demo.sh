#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# Start from one known process set so stale UDP sockets and QNX pathname
# services cannot collide with this run.
sh ./stop_demo.sh

mkdir -p logs
cd logs
export VIGIL_FACE_MODE=${VIGIL_FACE_MODE:-fixed}
runtime=${VIGIL_INFERENCE_RUNTIME:-ncnn}
case "$runtime" in
    ncnn)
        inference=../build/vigil_inference
        export VIGIL_MODEL_ROOT=${VIGIL_MODEL_ROOT:-../models}
        ;;
    ort)
        inference=../build/vigil_ort_inference
        export VIGIL_MODEL_ROOT=${VIGIL_MODEL_ROOT:-../models_ort}
        ;;
    *) echo "ERROR: VIGIL_INFERENCE_RUNTIME must be ncnn or ort" >&2; exit 2 ;;
esac
if [ ! -d /dev/mqueue ]; then
    echo "ERROR: QNX POSIX message queue manager is not running. Start it as root with: mqueue" >&2
    exit 1
fi
# Remove data left by a previous complete demo. Supervised logger restarts do not unlink it.
rm -f /dev/mqueue/vigil_log
rm -f /dev/mqueue/vigil_emergency
../build/vigil_supervisor ../build/vigil_logger ../build/vigil_decision \
    ../build/vigil_ingest ../build/vigil_gps ../build/vigil_alert \
    ../build/vigil_video_rx "$inference"
