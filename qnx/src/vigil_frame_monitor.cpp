#include "protocol_helpers.h"
#include "vigil_frame_buffer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    int fd = shm_open(VIGIL_FRAME_SHM, O_RDONLY, 0);
    if (fd == -1) { perror("shm_open (start vigil_video_rx first)"); return EXIT_FAILURE; }
    void *mapping = mmap(NULL, sizeof(vigil_frame_buffer_t), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap"); return EXIT_FAILURE; }
    volatile vigil_frame_buffer_t *shared =
        static_cast<volatile vigil_frame_buffer_t *>(mapping);
    uint32_t before, after, version, slot, bytes, frame_id, width, height, stride;
    uint64_t capture_ns, decoded_ns;
    uint32_t crc;
    for (;;) {
        before = shared->generation;
        if (before & 1U) continue;
        __sync_synchronize();
        version = shared->version; slot = shared->active_slot;
        bytes = shared->data_bytes; frame_id = shared->frame_id;
        width = shared->width; height = shared->height; stride = shared->stride;
        capture_ns = shared->capture_ns; decoded_ns = shared->decoded_ns;
        if (slot >= VIGIL_FRAME_SLOTS || bytes > VIGIL_FRAME_SLOT_BYTES) {
            fprintf(stderr, "invalid shared frame metadata\n");
            munmap(mapping, sizeof(vigil_frame_buffer_t)); return EXIT_FAILURE;
        }
        crc = vigil_crc32(const_cast<const unsigned char *>(shared->slots[slot]), bytes);
        __sync_synchronize(); after = shared->generation;
        if (before == after && !(after & 1U)) break;
    }
    printf("RGB_SHARED version=%u generation=%u frame=%u slot=%u size=%ux%u "
           "stride=%u bytes=%u crc=%08x capture_ns=%llu decoded_ns=%llu\n",
           version, after, frame_id, slot, width, height, stride, bytes, crc,
           (unsigned long long)capture_ns, (unsigned long long)decoded_ns);
    munmap(mapping, sizeof(vigil_frame_buffer_t));
    return version == VIGIL_FRAME_VERSION ? EXIT_SUCCESS : EXIT_FAILURE;
}
