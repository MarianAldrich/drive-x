#ifndef VIGIL_PROTOCOL_HELPERS_H
#define VIGIL_PROTOCOL_HELPERS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t vigil_crc32(const void *, size_t);
uint64_t vigil_ntoh64(uint64_t);
uint64_t vigil_hton64(uint64_t);
float vigil_ntoh_float(uint32_t);
uint32_t vigil_hton_float(float);
#ifdef __cplusplus
}
#endif
#endif
