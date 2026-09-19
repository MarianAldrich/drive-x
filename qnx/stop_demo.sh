#!/bin/sh

# Stop every DRIVE-X process by executable name. The -f option handles
# duplicates left by interrupted launches without prompting.
for service in vigil_supervisor vigil_inference vigil_video_rx vigil_ingest \
               vigil_decision vigil_logger vigil_gps vigil_alert vigil_monitor \
               vigil_frame_monitor vigil_stress
do
    slay -f -s SIGTERM "$service" >/dev/null 2>&1 || true
done

sleep 1

# Force-stop only processes that ignored the graceful request.
for service in vigil_supervisor vigil_inference vigil_video_rx vigil_ingest \
               vigil_decision vigil_logger vigil_gps vigil_alert vigil_monitor \
               vigil_frame_monitor vigil_stress
do
    slay -f -s SIGKILL "$service" >/dev/null 2>&1 || true
done

rm -f /tmp/vigil_*.hb /tmp/vigil_*.pid /tmp/vigil_unavailable /tmp/vigil_last_rto_us
rm -f /dev/mqueue/vigil_log /dev/mqueue/vigil_emergency
echo "DRIVE-X services stopped and stale runtime files removed"
