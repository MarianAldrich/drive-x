#include "vigil_common.h"
#include "vigil_protocol.h"
#include "vigil_runtime.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static const char *state_name(uint32_t state) {
    static const char *names[] = {"SENSOR_LOST", "WARMING_UP", "ALERT",
        "LOW_VIGILANCE", "CHECKING", "DROWSY"};
    return state < 6 ? names[state] : "UNKNOWN";
}

int main(void) {
    int fd = shm_open(VIGIL_SHARED_MEMORY, O_RDONLY, 0);
    if (fd == -1) { perror("shm_open (start services first)"); return EXIT_FAILURE; }
    void *mapping = mmap(NULL, sizeof(vigil_shared_telemetry_t), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap"); return EXIT_FAILURE; }
    volatile vigil_shared_telemetry_t *shared =
        static_cast<volatile vigil_shared_telemetry_t *>(mapping);
    vigil_shared_telemetry_t copy; uint32_t before, after;
    for (;;) {
        before = shared->generation;
        if (before & 1U) continue;
        __sync_synchronize();
        memcpy(&copy, const_cast<const vigil_shared_telemetry_t *>(shared), sizeof(copy));
        __sync_synchronize(); after = shared->generation;
        if (before == after && !(after & 1U)) break;
    }
    printf("SHARED_BUFFER version=%u writer_pid=%u seq=%u state=%s emergency=%s "
           "latency_us=%u deadline_misses=%u queue_drops=%u eye=%.3f yawn=%.3f age_ms=%llu\n",
           copy.version, copy.writer_pid, copy.sequence, state_name(copy.state),
           (copy.flags & VIGIL_EMERGENCY_ACTIVE) ? "ACTIVE" : "standby", copy.latency_us,
           copy.deadline_missed_total, copy.queue_dropped_total, copy.eye, copy.yawn,
           (unsigned long long)((vigil_now_ns() - copy.decision_ns) / 1000000ULL));
    munmap(mapping, sizeof(vigil_shared_telemetry_t));
    return copy.version == VIGIL_SHARED_VERSION ? EXIT_SUCCESS : EXIT_FAILURE;
}
