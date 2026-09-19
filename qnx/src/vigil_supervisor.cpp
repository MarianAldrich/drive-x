#include "vigil_common.h"

#include <errno.h>
#include <process.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/dispatch.h>
#include <sys/wait.h>

extern char **environ;
static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }

struct child_service {
    const char *name, *path, *heartbeat;
    pid_t pid;
    uint64_t launched_ns;
    uint64_t timeout_ns;
    uint64_t last_heartbeat_ns;
    int enabled;
};

static uint64_t heartbeat_value(const char *path) {
    FILE *file = fopen(path, "r"); unsigned long long value = 0;
    if (file) { fscanf(file, "%llu", &value); fclose(file); }
    return (uint64_t)value;
}

static pid_t launch(child_service *service, int record_rto) {
    unlink(service->heartbeat);
    uint64_t detected = vigil_now_ns();
    if (record_rto && strcmp(service->name, "decision") == 0) {
        FILE *unavailable = fopen("/tmp/vigil_unavailable", "w");
        if (unavailable) { fprintf(unavailable, "%llu\n", (unsigned long long)detected); fclose(unavailable); }
        fprintf(stdout, "SYSTEM_UNAVAILABLE reason=decision_restart\n"); fflush(stdout);
    }
    pid_t pid = spawnl(P_NOWAIT, service->path, service->path, NULL);
    if (pid == -1) { perror(service->name); return -1; }
    service->pid = pid; service->launched_ns = detected; service->last_heartbeat_ns = 0;
    fprintf(stdout, "%s launched pid=%d\n", service->name, pid); fflush(stdout);
    if (record_rto && strcmp(service->name, "decision") == 0) {
        uint64_t deadline = detected + 5000000000ULL, ready = 0;
        while (running && vigil_now_ns() < deadline) {
            ready = heartbeat_value(service->heartbeat);
            if (ready >= detected) break;
            delay(20);
        }
        uint32_t rto_us = ready >= detected ? (uint32_t)((ready - detected) / 1000ULL) : 0xffffffffU;
        FILE *file = fopen("/tmp/vigil_last_rto_us", "w");
        if (file) { fprintf(file, "%u\n", rto_us); fclose(file); }
        if (ready >= detected) unlink("/tmp/vigil_unavailable");
        FILE *recovery = fopen("vigil_recovery.csv", "a+");
        if (recovery) {
            fseek(recovery, 0, SEEK_END);
            if (ftell(recovery) == 0) fprintf(recovery, "detected_ns,ready_ns,rto_us,result\n");
            fprintf(recovery, "%llu,%llu,%u,%s\n", (unsigned long long)detected,
                    (unsigned long long)ready, rto_us, rto_us < 2000000U ? "PASS" : "FAIL");
            fclose(recovery);
        }
        fprintf(stdout, "RECOVERY decision_rto_us=%u target_us=2000000 result=%s\n", rto_us,
                rto_us < 2000000U ? "PASS" : "FAIL"); fflush(stdout);
    }
    return pid;
}

int main(int argc, char **argv) {
    if (vigil_configure_thread(3, 45) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    /* Prevent two supervisors from spawning duplicate UDP and IPC services. */
    name_attach_t *guard = name_attach(NULL, "drive_x_supervisor", 0);
    if (!guard) {
        fprintf(stderr, "ERROR: DRIVE-X supervisor is already running (%s)\n",
                strerror(errno));
        return EXIT_FAILURE;
    }
    FILE *supervisor_pid = fopen("/tmp/vigil_supervisor.pid", "w");
    if (supervisor_pid) { fprintf(supervisor_pid, "%d\n", getpid()); fclose(supervisor_pid); }
    const char *logger_path = argc > 1 ? argv[1] : "./vigil_logger";
    const char *decision_path = argc > 2 ? argv[2] : "./vigil_decision";
    const char *ingest_path = argc > 3 ? argv[3] : "./vigil_ingest";
    const char *gps_path = argc > 4 ? argv[4] : "./vigil_gps";
    const char *alert_path = argc > 5 ? argv[5] : "./vigil_alert";
    const char *video_path = argc > 6 ? argv[6] : "./vigil_video_rx";
    const char *inference_path = argc > 7 ? argv[7] : "./vigil_inference";
    const char *local_text = getenv("VIGIL_LOCAL_INFERENCE");
    const int local_inference = !local_text || strcmp(local_text, "0") != 0;
    const uint64_t normal_timeout = 4000000000ULL;
    child_service services[] = {{"logger", logger_path, "/tmp/vigil_logger.hb", -1, 0, normal_timeout, 0, 1},
                                {"decision", decision_path, "/tmp/vigil_decision.hb", -1, 0, normal_timeout, 0, 1},
                                {"ingest", ingest_path, "/tmp/vigil_ingest.hb", -1, 0, normal_timeout, 0, 1},
                                {"gps", gps_path, "/tmp/vigil_gps.hb", -1, 0, 6000000000ULL, 0, 1},
                                {"alert", alert_path, "/tmp/vigil_alert.hb", -1, 0, normal_timeout, 0, 1},
                                {"video_rx", video_path, "/tmp/vigil_video_rx.hb", -1, 0, 6000000000ULL, 0, local_inference},
                                /* First ncnn execution initializes kernels and allocators. */
                                {"inference", inference_path, "/tmp/vigil_inference.hb", -1, 0, 60000000000ULL, 0, local_inference}};
    const unsigned service_count = sizeof(services) / sizeof(services[0]);
    unlink("/tmp/vigil_last_rto_us");
    fprintf(stdout, "HAM check: %s; supervisor fallback active\n",
            access("/sbin/ham", X_OK) == 0 || access("/usr/sbin/ham", X_OK) == 0 ? "available" : "not found");
    fprintf(stdout, "PIPELINE alert=CPU3/FIFO62 ingest=CPU1/FIFO60 decision=CPU2/FIFO58 "
                    "supervisor=CPU3/FIFO45 gps=CPU3/SPORADIC35-12 "
                    "logger=CPU3/SPORADIC20-8\n");
    fprintf(stdout, "PIPELINE inference_location=%s\n",
            local_inference ? "QNX" : "WINDOWS_EVIDENCE_UDP");
    launch(&services[0], 0); delay(200);
    launch(&services[3], 0); delay(200); launch(&services[4], 0); delay(200);
    launch(&services[1], 0); delay(300); launch(&services[2], 0);
    if (local_inference) {
        delay(200); launch(&services[5], 0); delay(500); launch(&services[6], 0);
    }

    while (running) {
        uint64_t now = vigil_now_ns();
        for (unsigned i = 0; i < service_count; ++i) {
            if (!services[i].enabled) continue;
            int status = 0; pid_t result = waitpid(services[i].pid, &status, WNOHANG);
            uint64_t heartbeat = heartbeat_value(services[i].heartbeat);
            /* fopen("w") briefly exposes an empty heartbeat file. Preserve the
               last complete value instead of treating that instant as failure. */
            if (heartbeat) services[i].last_heartbeat_ns = heartbeat;
            uint64_t reference = services[i].last_heartbeat_ns ?
                                 services[i].last_heartbeat_ns : services[i].launched_ns;
            int stale = reference && now > reference && now - reference > services[i].timeout_ns;
            if (result == services[i].pid || stale) {
                fprintf(stdout, "FAULT service=%s reason=%s\n", services[i].name,
                        stale ? "heartbeat_timeout" : "process_exit"); fflush(stdout);
                if (stale) { kill(services[i].pid, SIGKILL); waitpid(services[i].pid, &status, 0); }
                delay(500);
                launch(&services[i], 1);
            }
        }
        delay(100);
    }
    for (unsigned i = 0; i < service_count; ++i)
        if (services[i].pid > 0) kill(services[i].pid, SIGTERM);
    for (unsigned i = 0; i < service_count; ++i)
        if (services[i].pid > 0) waitpid(services[i].pid, NULL, 0);
    unlink("/tmp/vigil_supervisor.pid");
    name_detach(guard, 0);
    return EXIT_SUCCESS;
}
