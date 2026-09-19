#ifndef VIGIL_PROTOCOL_H
#define VIGIL_PROTOCOL_H

#include <stdint.h>

#define VIGIL_MAGIC 0x5649474cU
#define VIGIL_VERSION 1U
#define VIGIL_TYPE_EVIDENCE 1U
#define VIGIL_TYPE_STATUS 2U
#define VIGIL_UDP_PORT 45555
#define VIGIL_DECISION_SERVICE "vigil_decision"

enum vigil_flags {
    VIGIL_MODEL_READY = 1U << 0,
    VIGIL_FACE_VALID = 1U << 1,
    VIGIL_HEAD_CUE = 1U << 2,
    VIGIL_SENDER_RESET = 1U << 3,
    VIGIL_EMERGENCY_ACTIVE = 1U << 4,
    VIGIL_GPS_FIX_VALID = 1U << 5
};

enum vigil_state {
    VIGIL_SENSOR_LOST = 0,
    VIGIL_WARMING_UP = 1,
    VIGIL_ALERT = 2,
    VIGIL_LOW_VIGILANCE = 3,
    VIGIL_CHECKING = 4,
    VIGIL_DROWSY = 5
};

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint8_t version;
    uint8_t type;
    uint16_t length;
    uint32_t sequence;
    uint64_t sender_ns;
    uint32_t eye_closed_bits;
    uint32_t yawn_bits;
    uint32_t inference_us;
    uint32_t flags;
    uint32_t crc32;
} vigil_evidence_wire_t;

typedef struct {
    uint32_t magic;
    uint8_t version;
    uint8_t type;
    uint16_t length;
    uint32_t sequence;
    uint64_t qnx_receive_ns;
    uint32_t decision_us;
    uint32_t state;
    uint32_t flags;
    uint32_t rto_us;
    uint32_t crc32;
} vigil_status_wire_t;
#pragma pack(pop)

typedef struct {
    uint16_t kind; /* 1=evidence, 2=timer tick */
    uint16_t reserved;
    uint32_t sequence;
    uint64_t receive_ns;
    float eye_closed;
    float yawn;
    uint32_t inference_us;
    uint32_t flags;
} vigil_ipc_message_t;

typedef struct {
    uint32_t sequence;
    uint64_t receive_ns;
    uint32_t decision_us;
    uint32_t state;
    uint32_t flags;
    uint32_t rto_us;
} vigil_ipc_reply_t;

#ifdef __cplusplus
static_assert(sizeof(vigil_evidence_wire_t) == 40, "evidence wire size changed");
static_assert(sizeof(vigil_status_wire_t) == 40, "status wire size changed");
#endif

#endif
