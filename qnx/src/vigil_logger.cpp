#include "vigil_common.h"
#include "vigil_protocol.h"
#include "vigil_runtime.h"
#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }
static const char *state_name(uint32_t state) {
    static const char *names[] = {"SENSOR_LOST", "WARMING_UP", "ALERT",
        "LOW_VIGILANCE", "CHECKING", "DROWSY"};
    return state < 6 ? names[state] : "UNKNOWN";
}

int main(void) {
    if (vigil_configure_sporadic(3, 20, 8, 5, 100) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    struct mq_attr a; memset(&a, 0, sizeof(a));
    a.mq_maxmsg = 64; a.mq_msgsize = sizeof(vigil_log_record_t);
    mqd_t queue = mq_open(VIGIL_LOG_QUEUE, O_RDONLY | O_CREAT, 0660, &a);
    if (queue == (mqd_t)-1) { perror("mq_open logger (is mqueue running?)"); return EXIT_FAILURE; }
    FILE *file = fopen("vigil_events.csv", "a+");
    if (!file) { perror("vigil_events.csv"); mq_close(queue); return EXIT_FAILURE; }
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0)
        fprintf(file, "sequence,receive_ns,decision_ns,latency_us,deadline_us,deadline_missed,state,flags,eye_closed,yawn,outage,restart_event,rto_us,queue_dropped\n");
    fprintf(stdout, "PIPELINE stage=logger CPU3 policy=SPORADIC priority=20/8 budget_ms=5 period_ms=100 queue=%s\n", VIGIL_LOG_QUEUE);
    fflush(stdout); vigil_heartbeat("/tmp/vigil_logger.hb");
    while (running) {
        struct timespec deadline; clock_gettime(CLOCK_REALTIME, &deadline); deadline.tv_sec += 1;
        vigil_log_record_t r = {};
        ssize_t received = mq_timedreceive(queue, reinterpret_cast<char *>(&r), sizeof(r), NULL, &deadline);
        if (received == -1) {
            if (errno == ETIMEDOUT || errno == EINTR) {
                vigil_heartbeat("/tmp/vigil_logger.hb"); continue;
            }
            perror("mq_timedreceive"); break;
        }
        fprintf(file, "%u,%llu,%llu,%u,%u,%u,%s,%u,%.6f,%.6f,%u,%u,%u,%u\n",
            r.sequence, (unsigned long long)r.receive_ns, (unsigned long long)r.decision_ns,
            r.latency_us, VIGIL_DECISION_DEADLINE_US, r.deadline_missed, state_name(r.state),
            r.flags, r.eye, r.yawn, r.outage, r.restart_event, r.rto_us, r.queue_dropped);
        fflush(file);
        fprintf(stdout, "DECISION seq=%u state=%s latency_us=%u deadline=%s eye=%.3f yawn=%.3f\n",
            r.sequence, state_name(r.state), r.latency_us,
            r.deadline_missed ? "MISS" : "PASS", r.eye, r.yawn);
        fflush(stdout); vigil_heartbeat("/tmp/vigil_logger.hb");
    }
    /* Keep the queue object alive so a supervised logger restart preserves senders. */
    fclose(file); mq_close(queue); unlink("/tmp/vigil_logger.hb");
    return EXIT_SUCCESS;
}
