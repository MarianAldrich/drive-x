#include "vigil_common.h"
#include "vigil_emergency.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <mqueue.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }

static int gps_snapshot(volatile vigil_gps_snapshot_t *shared, vigil_gps_snapshot_t *copy) {
    if (!shared) return 0;
    uint32_t before, after;
    for (;;) {
        before = shared->generation;
        if (before & 1U) continue;
        __sync_synchronize();
        memcpy(copy, const_cast<const vigil_gps_snapshot_t *>(shared), sizeof(*copy));
        __sync_synchronize(); after = shared->generation;
        if (before == after && !(after & 1U)) break;
    }
    return copy->version == VIGIL_GPS_VERSION && copy->fix_valid;
}

static volatile vigil_gps_snapshot_t *map_gps(void) {
    int fd = shm_open(VIGIL_GPS_SHM, O_RDONLY, 0);
    if (fd == -1) return NULL;
    void *mapping = mmap(NULL, sizeof(vigil_gps_snapshot_t), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return mapping == MAP_FAILED ? NULL :
        static_cast<volatile vigil_gps_snapshot_t *>(mapping);
}

static int send_to_relay(const char *host, const char *port, const char *message) {
    if (!host || !host[0]) return 0;
    struct addrinfo hints; memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_DGRAM;
    struct addrinfo *addresses = NULL;
    int result = getaddrinfo(host, port, &hints, &addresses);
    if (result != 0) { fprintf(stderr, "alert getaddrinfo: %s\n", gai_strerror(result)); return -1; }
    int socket_fd = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
    if (socket_fd == -1) { perror("alert socket"); freeaddrinfo(addresses); return -1; }
    ssize_t sent = sendto(socket_fd, message, strlen(message), 0,
                          addresses->ai_addr, addresses->ai_addrlen);
    close(socket_fd); freeaddrinfo(addresses);
    return sent == (ssize_t)strlen(message) ? 1 : -1;
}

int main(void) {
    if (vigil_configure_thread(3, 62) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    struct mq_attr attributes; memset(&attributes, 0, sizeof(attributes));
    attributes.mq_maxmsg = 16; attributes.mq_msgsize = sizeof(vigil_emergency_record_t);
    mqd_t queue = mq_open(VIGIL_EMERGENCY_QUEUE, O_RDONLY | O_NONBLOCK | O_CREAT,
                          0660, &attributes);
    if (queue == (mqd_t)-1) { perror("mq_open emergency"); return EXIT_FAILURE; }
    volatile vigil_gps_snapshot_t *gps = map_gps();
    const char *relay_host = getenv("VIGIL_ALERT_HOST");
    const char *relay_port = getenv("VIGIL_ALERT_PORT");
    if (!relay_port || !relay_port[0]) relay_port = "45557";
    FILE *log = fopen("vigil_emergency.csv", "a+");
    if (!log) { perror("vigil_emergency.csv"); mq_close(queue); return EXIT_FAILURE; }
    fseek(log, 0, SEEK_END);
    if (ftell(log) == 0)
        fprintf(log, "event_ns,sequence,event,gps_valid,latitude,longitude,gps_utc,relay_result\n");
    fprintf(stdout, "PIPELINE stage=alert CPU3 policy=FIFO priority=62 queue=%s relay=%s:%s\n",
            VIGIL_EMERGENCY_QUEUE, relay_host ? relay_host : "disabled", relay_port);
    fflush(stdout); vigil_heartbeat("/tmp/vigil_alert.hb");

    while (running) {
        vigil_emergency_record_t event = {};
        ssize_t amount = mq_receive(queue, reinterpret_cast<char *>(&event), sizeof(event), NULL);
        if (amount == -1) {
            if (errno == EAGAIN || errno == EINTR) {
                if (!gps) gps = map_gps();
                vigil_heartbeat("/tmp/vigil_alert.hb"); delay(50); continue;
            }
            perror("mq_receive emergency"); break;
        }
        vigil_gps_snapshot_t location = {}; int valid = gps_snapshot(gps, &location);
        double latitude = location.latitude_e7 / 10000000.0;
        double longitude = location.longitude_e7 / 10000000.0;
        if (event.kind == VIGIL_EMERGENCY_CLEAR) {
            fprintf(stdout, "EMERGENCY cleared seq=%u\n", event.sequence); fflush(stdout);
            fprintf(log, "%llu,%u,CLEAR,%d,%.7f,%.7f,%06u,0\n",
                    (unsigned long long)event.event_ns, event.sequence, valid,
                    latitude, longitude, location.utc_hhmmss); fflush(log);
            continue;
        }
        char message[512];
        snprintf(message, sizeof(message),
                 "{\"type\":\"DROWSINESS_ALERT\",\"sequence\":%u,\"event_ns\":%llu,"
                 "\"gps_valid\":%s,\"latitude\":%.7f,\"longitude\":%.7f,"
                 "\"gps_time\":\"%06u\",\"message\":\"Prolonged driver drowsiness detected. Please check the driver.\"}",
                 event.sequence, (unsigned long long)event.event_ns,
                 valid ? "true" : "false", latitude, longitude, location.utc_hhmmss);
        int relay_result = send_to_relay(relay_host, relay_port, message);
        fprintf(stdout, "EMERGENCY ACTIVE seq=%u gps=%s lat=%.7f lon=%.7f relay=%s\n",
                event.sequence, valid ? "VALID" : "NO_FIX", latitude, longitude,
                relay_result > 0 ? "SENT" : relay_result == 0 ? "DISABLED" : "FAILED");
        fprintf(stdout, "%s\n", message); fflush(stdout);
        fprintf(log, "%llu,%u,ACTIVATE,%d,%.7f,%.7f,%06u,%d\n",
                (unsigned long long)event.event_ns, event.sequence, valid,
                latitude, longitude, location.utc_hhmmss, relay_result); fflush(log);
        vigil_heartbeat("/tmp/vigil_alert.hb");
    }
    if (gps) munmap(const_cast<vigil_gps_snapshot_t *>(gps), sizeof(vigil_gps_snapshot_t));
    fclose(log); mq_close(queue); unlink("/tmp/vigil_alert.hb");
    return EXIT_SUCCESS;
}
