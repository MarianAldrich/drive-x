#ifndef VIGIL_RUNTIME_H
#define VIGIL_RUNTIME_H

#include <stdint.h>

#define VIGIL_LOG_QUEUE "/vigil_log"
#define VIGIL_SHARED_MEMORY "/vigil_telemetry"
#define VIGIL_SHARED_VERSION 1U
#define VIGIL_DECISION_DEADLINE_US 10000U

typedef struct {
    uint32_t sequence;
    uint32_t state;
    uint32_t flags;
    uint32_t latency_us;
    uint32_t deadline_missed;
    uint32_t rto_us;
    uint64_t receive_ns;
    uint64_t decision_ns;
    float eye;
    float yawn;
    uint32_t outage;
    uint32_t restart_event;
    uint32_t queue_dropped;
    uint32_t reserved;
} vigil_log_record_t;

/* Pointer-free fixed-size snapshot. generation is a seqlock: odd while writing. */
typedef struct {
    volatile uint32_t generation;
    uint32_t version;
    uint32_t state;
    uint32_t sequence;
    uint64_t receive_ns;
    uint64_t decision_ns;
    uint32_t latency_us;
    uint32_t deadline_missed_total;
    uint32_t queue_dropped_total;
    uint32_t flags;
    float eye;
    float yawn;
    uint32_t writer_pid;
    uint32_t reserved[7];
} vigil_shared_telemetry_t;

#endif
