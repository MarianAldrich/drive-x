#ifndef VIGIL_EMERGENCY_H
#define VIGIL_EMERGENCY_H

#include <stdint.h>

#define VIGIL_EMERGENCY_QUEUE "/vigil_emergency"
#define VIGIL_GPS_SHM "/vigil_gps"
#define VIGIL_GPS_VERSION 1U
#define VIGIL_EMERGENCY_ACTIVATE 1U
#define VIGIL_EMERGENCY_CLEAR 2U

typedef struct {
    uint32_t kind;
    uint32_t sequence;
    uint32_t state;
    uint32_t threshold_ms;
    uint64_t event_ns;
} vigil_emergency_record_t;

/* Coordinates are signed decimal degrees multiplied by 10,000,000. */
typedef struct {
    volatile uint32_t generation;
    uint32_t version;
    uint32_t fix_valid;
    int32_t latitude_e7;
    int32_t longitude_e7;
    uint32_t utc_hhmmss;
    uint32_t sentences_valid;
    uint32_t sentences_rejected;
    uint64_t update_ns;
    uint32_t writer_pid;
    uint32_t reserved[5];
} vigil_gps_snapshot_t;

#endif
