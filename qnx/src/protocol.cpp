#include "vigil_protocol.h"

#include <arpa/inet.h>
#include <string.h>

extern "C" uint32_t vigil_crc32(const void *data, size_t length) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (uint32_t)-(int32_t)(crc & 1U));
    }
    return crc ^ 0xffffffffU;
}

extern "C" uint64_t vigil_ntoh64(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)ntohl((uint32_t)value) << 32) | ntohl((uint32_t)(value >> 32));
#else
    return value;
#endif
}

extern "C" uint64_t vigil_hton64(uint64_t value) { return vigil_ntoh64(value); }

extern "C" float vigil_ntoh_float(uint32_t value) {
    value = ntohl(value);
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

extern "C" uint32_t vigil_hton_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return htonl(bits);
}
