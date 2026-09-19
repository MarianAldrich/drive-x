#ifndef VIGIL_COMMON_H
#define VIGIL_COMMON_H

#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/neutrino.h>

static inline uint64_t vigil_now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static inline int vigil_configure_thread(unsigned cpu, int priority) {
    uint64_t runmask = 1ULL << cpu;
    /* QNX 8 RUNMASK takes the value cast to void*, not a pointer to it. */
    if (ThreadCtl(_NTO_TCTL_RUNMASK, (void *)(uintptr_t)runmask) == -1) {
        perror("ThreadCtl RUNMASK");
        return -1;
    }
    struct sched_param parameters;
    memset(&parameters, 0, sizeof(parameters));
    parameters.sched_priority = priority;
    if (sched_setscheduler(0, SCHED_FIFO, &parameters) == -1) {
        perror("sched_setscheduler SCHED_FIFO");
        return -1;
    }
    return 0;
}

static inline void vigil_ms_to_timespec(unsigned ms, struct timespec *value) {
    value->tv_sec = (time_t)(ms / 1000U);
    value->tv_nsec = (long)(ms % 1000U) * 1000000L;
}

/* QNX sporadic scheduling gives background work an explicit execution budget. */
static inline int vigil_configure_sporadic(unsigned cpu, int high_priority,
                                           int low_priority, unsigned budget_ms,
                                           unsigned period_ms) {
    if (budget_ms == 0 || budget_ms >= period_ms || high_priority <= low_priority) {
        errno = EINVAL;
        return -1;
    }
    uint64_t runmask = 1ULL << cpu;
    if (ThreadCtl(_NTO_TCTL_RUNMASK, (void *)(uintptr_t)runmask) == -1) {
        perror("ThreadCtl RUNMASK");
        return -1;
    }
    struct sched_param parameters;
    memset(&parameters, 0, sizeof(parameters));
    parameters.sched_priority = high_priority;
    parameters.sched_ss_low_priority = low_priority;
    parameters.sched_ss_max_repl = 8;
    vigil_ms_to_timespec(budget_ms, &parameters.sched_ss_init_budget);
    vigil_ms_to_timespec(period_ms, &parameters.sched_ss_repl_period);
    if (sched_setscheduler(0, SCHED_SPORADIC, &parameters) == -1) {
        perror("sched_setscheduler SCHED_SPORADIC");
        return -1;
    }
    return 0;
}

static inline void vigil_heartbeat(const char *path) {
    static uint64_t last = 0;
    uint64_t now = vigil_now_ns();
    if (last && now - last < 500000000ULL) return;
    last = now;
    FILE *file = fopen(path, "w");
    if (file) {
        fprintf(file, "%llu\n", (unsigned long long)now);
        fclose(file);
    }
}

#endif
