#include "protocol_helpers.h"
#include "vigil_common.h"
#include "vigil_frame_buffer.h"
#include "vigil_video_protocol.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <img/img.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <vector>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }

struct frame_state {
    uint32_t id;
    uint64_t capture_ns;
    uint64_t assembly_started_ns;
    uint32_t expected_crc;
    uint16_t width, height, chunk_count;
    uint32_t received_count;
    std::vector<unsigned char> bytes;
    std::vector<unsigned char> received;
};

static void reset_frame(frame_state &frame, uint32_t id, uint64_t capture_ns,
                        uint32_t crc, uint16_t width, uint16_t height,
                        uint16_t chunks, uint32_t bytes) {
    frame.id = id; frame.capture_ns = capture_ns; frame.assembly_started_ns = vigil_now_ns();
    frame.expected_crc = crc;
    frame.width = width; frame.height = height; frame.chunk_count = chunks;
    frame.received_count = 0; frame.bytes.assign(bytes, 0); frame.received.assign(chunks, 0);
}

static int save_frame(const char *path, const frame_state &frame) {
    char temporary[512];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", path) >= (int)sizeof(temporary)) {
        errno = ENAMETOOLONG; return -1;
    }
    FILE *file = fopen(temporary, "wb");
    if (!file) return -1;
    size_t written = fwrite(frame.bytes.data(), 1, frame.bytes.size(), file);
    int error = ferror(file); fclose(file);
    if (error || written != frame.bytes.size()) { unlink(temporary); errno = EIO; return -1; }
    if (rename(temporary, path) == -1) { unlink(temporary); return -1; }
    return 0;
}

static vigil_frame_buffer_t *open_frame_buffer(void) {
    int fd = shm_open(VIGIL_FRAME_SHM, O_RDWR | O_CREAT, 0660);
    if (fd == -1) { perror("shm_open RGB frame buffer"); return NULL; }
    if (ftruncate(fd, sizeof(vigil_frame_buffer_t)) == -1) {
        perror("ftruncate RGB frame buffer"); close(fd); return NULL;
    }
    void *mapping = mmap(NULL, sizeof(vigil_frame_buffer_t), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap RGB frame buffer"); return NULL; }
    vigil_frame_buffer_t *buffer = static_cast<vigil_frame_buffer_t *>(mapping);
    memset(buffer, 0, sizeof(*buffer));
    buffer->version = VIGIL_FRAME_VERSION;
    return buffer;
}

static int publish_rgb(vigil_frame_buffer_t *buffer, uint32_t frame_id,
                       uint64_t capture_ns, const img_t &image) {
    if (!buffer || image.w > VIGIL_FRAME_MAX_WIDTH || image.h > VIGIL_FRAME_MAX_HEIGHT ||
        image.access.direct.stride > VIGIL_FRAME_MAX_WIDTH * VIGIL_FRAME_CHANNELS)
        return -1;
    const uint32_t bytes = image.access.direct.stride * image.h;
    if (bytes > VIGIL_FRAME_SLOT_BYTES) return -1;
    const uint32_t slot = (buffer->active_slot + 1U) % VIGIL_FRAME_SLOTS;
    uint32_t generation = buffer->generation;
    buffer->generation = generation + 1U;
    __sync_synchronize();
    memcpy(buffer->slots[slot], image.access.direct.data, bytes);
    buffer->version = VIGIL_FRAME_VERSION;
    buffer->active_slot = slot;
    buffer->frame_id = frame_id;
    buffer->width = image.w;
    buffer->height = image.h;
    buffer->stride = image.access.direct.stride;
    buffer->pixel_format = 1U;
    buffer->capture_ns = capture_ns;
    buffer->decoded_ns = vigil_now_ns();
    buffer->data_bytes = bytes;
    __sync_synchronize();
    buffer->generation = generation + 2U;
    return 0;
}

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : VIGIL_VIDEO_PORT;
    const char *output = argc > 2 ? argv[2] : "/tmp/drive_x_latest.jpg";
    if (port < 1 || port > 65535 || vigil_configure_sporadic(0, 25, 10, 40, 100) == -1)
        return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    img_lib_t image_library = NULL;
    int image_result = img_lib_attach(&image_library);
    if (image_result != IMG_ERR_OK) {
        fprintf(stderr, "img_lib_attach failed: %d (check /etc/system/config/img.conf)\n",
                image_result);
        return EXIT_FAILURE;
    }
    vigil_frame_buffer_t *frame_buffer = open_frame_buffer();
    if (!frame_buffer) { img_lib_detach(image_library); return EXIT_FAILURE; }
    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd == -1) {
        perror("socket"); munmap(frame_buffer, sizeof(*frame_buffer));
        img_lib_detach(image_library); return EXIT_FAILURE;
    }
    struct timeval timeout = {1, 0};
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    int reuse = 1; setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address; memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET; address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(socket_fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) == -1) {
        perror("bind video UDP"); close(socket_fd); munmap(frame_buffer, sizeof(*frame_buffer));
        img_lib_detach(image_library);
        return EXIT_FAILURE;
    }
    fprintf(stdout, "VIDEO_RX UDP=%d CPU0 policy=SPORADIC priority=25/10 budget_ms=40 "
                    "period_ms=100 output=%s shared=%s slots=2\n",
            port, output, VIGIL_FRAME_SHM); fflush(stdout);
    vigil_heartbeat("/tmp/vigil_video_rx.hb");

    frame_state frame = {};
    bool have_frame = false, seen_id = false;
    uint32_t latest_id = 0;
    uint64_t completed = 0, decoded = 0, decode_failed = 0;
    uint64_t dropped = 0, rejected = 0, last_report = vigil_now_ns();
    unsigned char datagram[sizeof(vigil_video_chunk_t) + VIGIL_VIDEO_CHUNK_BYTES];
    while (running) {
        ssize_t length = recvfrom(socket_fd, datagram, sizeof(datagram), 0, NULL, NULL);
        if (length == -1) {
            if (errno == EAGAIN || errno == ETIMEDOUT || errno == EINTR) {
                vigil_heartbeat("/tmp/vigil_video_rx.hb"); continue;
            }
            perror("recvfrom video"); break;
        }
        if ((size_t)length < sizeof(vigil_video_chunk_t)) { ++rejected; continue; }
        vigil_video_chunk_t wire; memcpy(&wire, datagram, sizeof(wire));
        uint32_t magic = ntohl(wire.magic), frame_id = ntohl(wire.frame_id);
        uint16_t header_bytes = ntohs(wire.header_bytes), index = ntohs(wire.chunk_index);
        uint16_t chunks = ntohs(wire.chunk_count), payload = ntohs(wire.payload_bytes);
        uint32_t frame_bytes = ntohl(wire.frame_bytes), crc = ntohl(wire.frame_crc32);
        uint16_t width = ntohs(wire.width), height = ntohs(wire.height);
        uint64_t capture_ns = vigil_ntoh64(wire.capture_ns);
        if (magic != VIGIL_VIDEO_MAGIC || wire.version != VIGIL_VIDEO_VERSION ||
            wire.type != VIGIL_VIDEO_TYPE_JPEG || header_bytes != sizeof(wire) ||
            chunks == 0 || chunks > VIGIL_VIDEO_MAX_CHUNKS || index >= chunks ||
            frame_bytes == 0 || frame_bytes > VIGIL_VIDEO_MAX_BYTES ||
            payload == 0 || payload > VIGIL_VIDEO_CHUNK_BYTES ||
            (size_t)length != sizeof(wire) + payload || width == 0 || height == 0 ||
            (uint32_t)index * VIGIL_VIDEO_CHUNK_BYTES + payload > frame_bytes) {
            ++rejected; continue;
        }
        if (!have_frame || frame_id != frame.id) {
            if (seen_id && (int32_t)(frame_id - latest_id) <= 0) {
                ++rejected; continue;
            }
            if (have_frame && frame.received_count != frame.chunk_count) ++dropped;
            reset_frame(frame, frame_id, capture_ns, crc, width, height, chunks, frame_bytes);
            have_frame = true; seen_id = true; latest_id = frame_id;
        }
        if (chunks != frame.chunk_count || frame_bytes != frame.bytes.size() ||
            crc != frame.expected_crc || width != frame.width || height != frame.height) {
            ++rejected; continue;
        }
        if (!frame.received[index]) {
            memcpy(frame.bytes.data() + (size_t)index * VIGIL_VIDEO_CHUNK_BYTES,
                   datagram + sizeof(wire), payload);
            frame.received[index] = 1; ++frame.received_count;
        }
        if (frame.received_count == frame.chunk_count) {
            if (vigil_crc32(frame.bytes.data(), frame.bytes.size()) != frame.expected_crc) {
                ++rejected;
            } else if (save_frame(output, frame) == -1) {
                perror("save JPEG"); close(socket_fd); munmap(frame_buffer, sizeof(*frame_buffer));
                img_lib_detach(image_library);
                return EXIT_FAILURE;
            } else {
                ++completed;
                uint64_t decode_started = vigil_now_ns();
                img_t image; memset(&image, 0, sizeof(image));
                image.format = IMG_FMT_RGB888;
                image.flags = IMG_FORMAT;
                int result = img_load_file(image_library, output, NULL, &image);
                uint64_t decode_finished = vigil_now_ns();
                uint32_t decode_us = (uint32_t)((decode_finished - decode_started) / 1000ULL);
                unsigned decoded_width = image.w, decoded_height = image.h;
                unsigned decoded_stride = 0; uint32_t rgb_crc = 0;
                if (result == IMG_ERR_OK && (image.flags & IMG_DIRECT) &&
                    (image.flags & IMG_W) && (image.flags & IMG_H) &&
                    (image.flags & IMG_FORMAT) && image.format == IMG_FMT_RGB888 &&
                    image.access.direct.data != NULL) {
                    decoded_stride = image.access.direct.stride;
                    rgb_crc = vigil_crc32(image.access.direct.data,
                                          (size_t)decoded_stride * decoded_height);
                    if (publish_rgb(frame_buffer, frame.id, frame.capture_ns, image) == 0)
                        ++decoded;
                    else {
                        ++decode_failed;
                        fprintf(stderr, "RGB_PUBLISH_FAIL frame=%u dimensions=%ux%u stride=%u\n",
                                frame.id, decoded_width, decoded_height, decoded_stride);
                    }
                    free(image.access.direct.data);
                } else {
                    ++decode_failed;
                    fprintf(stderr, "JPEG_DECODE_FAIL frame=%u result=%d flags=0x%x format=0x%x\n",
                            frame.id, result, image.flags, (unsigned)image.format);
                }
                uint64_t now = vigil_now_ns();
                if (now - last_report >= 1000000000ULL) {
                    double assembly_ms = (now - frame.assembly_started_ns) / 1000000.0;
                    fprintf(stdout, "VIDEO frame=%u jpeg_bytes=%u wire=%ux%u decoded=%ux%u "
                                    "stride=%u decode_us=%u rgb_crc=%08x assembly_ms=%.1f "
                                    "complete=%llu decoded_total=%llu decode_failed=%llu "
                                    "dropped=%llu rejected=%llu\n",
                            frame.id, frame_bytes, frame.width, frame.height,
                            decoded_width, decoded_height, decoded_stride, decode_us, rgb_crc,
                            assembly_ms, (unsigned long long)completed,
                            (unsigned long long)decoded, (unsigned long long)decode_failed,
                            (unsigned long long)dropped, (unsigned long long)rejected);
                    fflush(stdout); last_report = now;
                }
            }
            have_frame = false;
        }
        vigil_heartbeat("/tmp/vigil_video_rx.hb");
    }
    close(socket_fd); munmap(frame_buffer, sizeof(*frame_buffer));
    img_lib_detach(image_library);
    unlink("/tmp/vigil_video_rx.hb");
    fprintf(stdout, "VIDEO_RX stopped complete=%llu decoded=%llu decode_failed=%llu "
                    "dropped=%llu rejected=%llu\n",
            (unsigned long long)completed, (unsigned long long)decoded,
            (unsigned long long)decode_failed, (unsigned long long)dropped,
            (unsigned long long)rejected);
    return EXIT_SUCCESS;
}
