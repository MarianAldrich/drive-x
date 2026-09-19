#ifndef VIGIL_FRAME_BUFFER_H
#define VIGIL_FRAME_BUFFER_H

#include <stdint.h>

#define VIGIL_FRAME_SHM "/vigil_rgb_frames"
#define VIGIL_FRAME_VERSION 1U
#define VIGIL_FRAME_SLOTS 2U
#define VIGIL_FRAME_MAX_WIDTH 640U
#define VIGIL_FRAME_MAX_HEIGHT 480U
#define VIGIL_FRAME_CHANNELS 3U
#define VIGIL_FRAME_SLOT_BYTES \
    (VIGIL_FRAME_MAX_WIDTH * VIGIL_FRAME_MAX_HEIGHT * VIGIL_FRAME_CHANNELS)

/* The generation counter is odd while a producer is writing. Consumers retry
 * if it changes while they copy the selected slot. */
typedef struct {
    volatile uint32_t generation;
    uint32_t version;
    uint32_t active_slot;
    uint32_t frame_id;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_format; /* 1 = packed RGB888 */
    uint64_t capture_ns;
    uint64_t decoded_ns;
    uint32_t data_bytes;
    uint32_t reserved[5];
    unsigned char slots[VIGIL_FRAME_SLOTS][VIGIL_FRAME_SLOT_BYTES];
} vigil_frame_buffer_t;

#endif
