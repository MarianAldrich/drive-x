#ifndef VIGIL_VIDEO_PROTOCOL_H
#define VIGIL_VIDEO_PROTOCOL_H

#include <stdint.h>

#define VIGIL_VIDEO_MAGIC 0x44585646U /* DXVF */
#define VIGIL_VIDEO_VERSION 1U
#define VIGIL_VIDEO_TYPE_JPEG 1U
#define VIGIL_VIDEO_PORT 45556
#define VIGIL_VIDEO_CHUNK_BYTES 1200U
#define VIGIL_VIDEO_MAX_BYTES (512U * 1024U)
#define VIGIL_VIDEO_MAX_CHUNKS ((VIGIL_VIDEO_MAX_BYTES + VIGIL_VIDEO_CHUNK_BYTES - 1U) / VIGIL_VIDEO_CHUNK_BYTES)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint8_t version;
    uint8_t type;
    uint16_t header_bytes;
    uint32_t frame_id;
    uint64_t capture_ns;
    uint16_t chunk_index;
    uint16_t chunk_count;
    uint32_t frame_bytes;
    uint16_t payload_bytes;
    uint16_t width;
    uint16_t height;
    uint16_t reserved;
    uint32_t frame_crc32;
} vigil_video_chunk_t;
#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(vigil_video_chunk_t) == 40, "video chunk header size changed");
#endif

#endif
