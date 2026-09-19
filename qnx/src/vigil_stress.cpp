#include "vigil_common.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pid_t read_pid(const char *path) {
    FILE *file = fopen(path, "r"); int pid = -1;
    if (file) { fscanf(file, "%d", &pid); fclose(file); }
    return (pid_t)pid;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--kill-decision") == 0) {
        pid_t pid = read_pid("/tmp/vigil_decision.pid");
        if (pid <= 0 || kill(pid, SIGKILL) == -1) { perror("kill decision"); return EXIT_FAILURE; }
        printf("Injected decision process failure pid=%d\n", pid); return EXIT_SUCCESS;
    }
    if (argc >= 2 && strcmp(argv[1], "--udp-loss") == 0) {
        pid_t pid = read_pid("/tmp/vigil_ingest.pid");
        if (pid <= 0 || kill(pid, SIGUSR1) == -1) { perror("signal ingest"); return EXIT_FAILURE; }
        printf("Injected 3-second UDP receive blackout pid=%d\n", pid); return EXIT_SUCCESS;
    }
    int seconds = argc >= 3 && strcmp(argv[1], "--cpu") == 0 ? atoi(argv[2]) : 30;
    /* CPU3 load is capped at 80 ms per 100 ms replenishment period. */
    if (seconds < 1 || vigil_configure_sporadic(3, 10, 4, 80, 100) == -1)
        return EXIT_FAILURE;
    uint64_t end = vigil_now_ns() + (uint64_t)seconds * 1000000000ULL;
    volatile uint64_t value = 1;
    while (vigil_now_ns() < end) value = value * 1664525U + 1013904223U;
    printf("CPU3 stress complete seconds=%d checksum=%llu\n", seconds, (unsigned long long)value);
    return EXIT_SUCCESS;
}
