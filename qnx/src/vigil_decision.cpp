#include "vigil_common.h"
#include "vigil_emergency.h"
#include "vigil_protocol.h"
#include "vigil_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/mman.h>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }

static uint32_t read_rto(void) {
    FILE *file = fopen("/tmp/vigil_last_rto_us", "r"); unsigned value = 0;
    if (file) { fscanf(file, "%u", &value); fclose(file); unlink("/tmp/vigil_last_rto_us"); }
    return value;
}

static mqd_t open_log_queue(void) {
    struct mq_attr a; memset(&a, 0, sizeof(a));
    a.mq_maxmsg = 64; a.mq_msgsize = sizeof(vigil_log_record_t);
    return mq_open(VIGIL_LOG_QUEUE, O_WRONLY | O_NONBLOCK | O_CREAT, 0660, &a);
}

static mqd_t open_emergency_queue(void) {
    struct mq_attr a; memset(&a, 0, sizeof(a));
    a.mq_maxmsg = 16; a.mq_msgsize = sizeof(vigil_emergency_record_t);
    return mq_open(VIGIL_EMERGENCY_QUEUE, O_WRONLY | O_NONBLOCK | O_CREAT, 0660, &a);
}

static uint32_t emergency_threshold_ms(void) {
    const char *text = getenv("VIGIL_EMERGENCY_MS");
    if (!text || !text[0]) return 10000U;
    char *end = NULL; unsigned long value = strtoul(text, &end, 10);
    return end && *end == '\0' && value >= 1000UL && value <= 3600000UL ?
           (uint32_t)value : 10000U;
}

static void publish_emergency(mqd_t queue, uint32_t kind, uint32_t sequence,
                              uint32_t state, uint32_t threshold_ms, uint64_t now) {
    if (queue == (mqd_t)-1) return;
    vigil_emergency_record_t record = {kind, sequence, state, threshold_ms, now};
    if (mq_send(queue, reinterpret_cast<const char *>(&record), sizeof(record), 10) == -1 &&
        errno != EAGAIN)
        perror("mq_send emergency");
}

static vigil_shared_telemetry_t *open_telemetry(void) {
    int fd = shm_open(VIGIL_SHARED_MEMORY, O_RDWR | O_CREAT, 0660);
    if (fd == -1) { perror("shm_open telemetry"); return NULL; }
    if (ftruncate(fd, sizeof(vigil_shared_telemetry_t)) == -1) {
        perror("ftruncate telemetry"); close(fd); return NULL;
    }
    void *mapping = mmap(NULL, sizeof(vigil_shared_telemetry_t), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap telemetry"); return NULL; }
    vigil_shared_telemetry_t *t = static_cast<vigil_shared_telemetry_t *>(mapping);
    memset(t, 0, sizeof(*t)); t->version = VIGIL_SHARED_VERSION; t->writer_pid = (uint32_t)getpid();
    return t;
}

static void publish_telemetry(vigil_shared_telemetry_t *t, const vigil_log_record_t &r,
                              uint32_t missed_total, uint32_t dropped_total) {
    if (!t) return;
    uint32_t generation = t->generation;
    t->generation = generation + 1U; __sync_synchronize();
    t->version = VIGIL_SHARED_VERSION; t->state = r.state; t->sequence = r.sequence;
    t->receive_ns = r.receive_ns; t->decision_ns = r.decision_ns;
    t->latency_us = r.latency_us; t->deadline_missed_total = missed_total;
    t->queue_dropped_total = dropped_total; t->flags = r.flags;
    t->eye = r.eye; t->yawn = r.yawn; t->writer_pid = (uint32_t)getpid();
    __sync_synchronize(); t->generation = generation + 2U;
}

int main(void) {
    if (vigil_configure_thread(2, 58) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    name_attach_t *attachment = name_attach(NULL, VIGIL_DECISION_SERVICE, 0);
    if (!attachment) { perror("name_attach"); return EXIT_FAILURE; }
    FILE *pidfile = fopen("/tmp/vigil_decision.pid", "w");
    if (pidfile) { fprintf(pidfile, "%d\n", getpid()); fclose(pidfile); }
    mqd_t log_queue = open_log_queue();
    if (log_queue == (mqd_t)-1) perror("mq_open decision logger");
    mqd_t emergency_queue = open_emergency_queue();
    if (emergency_queue == (mqd_t)-1) perror("mq_open decision emergency");
    vigil_shared_telemetry_t *telemetry = open_telemetry();
    constexpr unsigned EYE_WINDOW = 8;
    float eyes[EYE_WINDOW] = {}; unsigned count = 0, cursor = 0;
    uint64_t last_packet_ns = 0, high_since_ns = 0, drowsy_since_ns = 0;
    const uint32_t emergency_ms = emergency_threshold_ms();
    int emergency_active = 0;
    uint32_t state = VIGIL_SENSOR_LOST, prior_state = VIGIL_SENSOR_LOST;
    uint32_t missed_total = 0, dropped_total = 0;
    fprintf(stdout, "PIPELINE stage=decision CPU2 policy=FIFO priority=58 deadline_us=%u "
                    "emergency_after_ms=%u queue=%s\n",
            VIGIL_DECISION_DEADLINE_US, emergency_ms, VIGIL_EMERGENCY_QUEUE); fflush(stdout);
    vigil_heartbeat("/tmp/vigil_decision.hb");

    while (running) {
        uint64_t receive_timeout = 250000000ULL;
        TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL, &receive_timeout, NULL);
        vigil_ipc_message_t message = {};
        int rcvid = MsgReceive(attachment->chid, &message, sizeof(message), NULL);
        uint64_t now = vigil_now_ns();
        if (rcvid == -1) {
            if (errno == ETIMEDOUT || errno == EINTR) {
                if (last_packet_ns && now - last_packet_ns >= 2000000000ULL) {
                    state = VIGIL_SENSOR_LOST; count = cursor = 0; high_since_ns = 0;
                    drowsy_since_ns = 0;
                    if (emergency_active) {
                        emergency_active = 0;
                        publish_emergency(emergency_queue, VIGIL_EMERGENCY_CLEAR,
                                          0, state, emergency_ms, now);
                    }
                }
                vigil_heartbeat("/tmp/vigil_decision.hb"); continue;
            }
            perror("MsgReceive"); continue;
        }
        if (rcvid == 0) continue;
        if (message.kind == 1) {
            last_packet_ns = message.receive_ns;
            if (message.flags & VIGIL_SENDER_RESET) { count = cursor = 0; high_since_ns = 0; }
            if ((message.flags & (VIGIL_MODEL_READY | VIGIL_FACE_VALID)) !=
                (VIGIL_MODEL_READY | VIGIL_FACE_VALID)) {
                count = cursor = 0; high_since_ns = 0; state = VIGIL_WARMING_UP;
            } else {
                eyes[cursor] = message.eye_closed; cursor = (cursor + 1) % EYE_WINDOW;
                if (count < EYE_WINDOW) ++count;
                if (count < EYE_WINDOW) state = VIGIL_WARMING_UP;
                else {
                    float total = 0.f; for (unsigned i = 0; i < EYE_WINDOW; ++i) total += eyes[i];
                    float closed = total / EYE_WINDOW;
                    if (closed >= .65f) {
                        if (!high_since_ns) high_since_ns = now;
                        state = now - high_since_ns >= 2000000000ULL ? VIGIL_DROWSY : VIGIL_CHECKING;
                    } else {
                        high_since_ns = 0;
                        state = closed >= .35f || message.yawn >= .45f ||
                                (message.flags & VIGIL_HEAD_CUE) ? VIGIL_LOW_VIGILANCE : VIGIL_ALERT;
                    }
                }
            }
        } else if (!last_packet_ns || now - last_packet_ns >= 2000000000ULL) {
            state = VIGIL_SENSOR_LOST; count = cursor = 0; high_since_ns = 0;
        }
        if (state == VIGIL_DROWSY) {
            if (!drowsy_since_ns) drowsy_since_ns = now;
            if (!emergency_active && now - drowsy_since_ns >= (uint64_t)emergency_ms * 1000000ULL) {
                emergency_active = 1;
                publish_emergency(emergency_queue, VIGIL_EMERGENCY_ACTIVATE,
                                  message.sequence, state, emergency_ms, now);
            }
        } else {
            drowsy_since_ns = 0;
            if (emergency_active) {
                emergency_active = 0;
                publish_emergency(emergency_queue, VIGIL_EMERGENCY_CLEAR,
                                  message.sequence, state, emergency_ms, now);
            }
        }
        /* Capture completion after temporal classification, not merely after receive. */
        now = vigil_now_ns();
        uint32_t latency_us = message.receive_ns <= now ?
                              (uint32_t)((now - message.receive_ns) / 1000ULL) : 0;
        uint32_t deadline_missed = latency_us > VIGIL_DECISION_DEADLINE_US;
        missed_total += deadline_missed;
        uint32_t rto_us = read_rto();
        uint32_t output_flags = message.flags |
            (emergency_active ? (uint32_t)VIGIL_EMERGENCY_ACTIVE : 0U);
        vigil_ipc_reply_t reply = {message.sequence, message.receive_ns, latency_us, state,
                                   output_flags, rto_us};
        MsgReply(rcvid, EOK, &reply, sizeof(reply));
        vigil_heartbeat("/tmp/vigil_decision.hb");
        if (message.kind == 1 || state != prior_state) {
            vigil_log_record_t record = {message.sequence, state, output_flags, latency_us,
                deadline_missed, rto_us, message.receive_ns, now, message.eye_closed, message.yawn,
                state == VIGIL_SENSOR_LOST, rto_us != 0, dropped_total, 0};
            if (log_queue == (mqd_t)-1) log_queue = open_log_queue();
            if (log_queue == (mqd_t)-1 ||
                mq_send(log_queue, reinterpret_cast<const char *>(&record), sizeof(record), 1) == -1)
                ++dropped_total;
            publish_telemetry(telemetry, record, missed_total, dropped_total);
            prior_state = state;
        }
    }
    if (log_queue != (mqd_t)-1) mq_close(log_queue);
    if (emergency_queue != (mqd_t)-1) mq_close(emergency_queue);
    if (telemetry) munmap(telemetry, sizeof(*telemetry));
    name_detach(attachment, 0); unlink("/tmp/vigil_decision.pid"); unlink("/tmp/vigil_decision.hb");
    return EXIT_SUCCESS;
}
