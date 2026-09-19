#include "vigil_common.h"
#include "vigil_emergency.h"

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <termios.h>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }

static int configure_uart(int fd) {
    struct termios options;
    if (tcgetattr(fd, &options) == -1) return -1;
    options.c_iflag = 0; options.c_oflag = 0; options.c_lflag = 0;
    options.c_cflag = CLOCAL | CREAD | CS8;
    options.c_cc[VMIN] = 0; options.c_cc[VTIME] = 2;
    cfsetispeed(&options, B9600); cfsetospeed(&options, B9600);
    return tcsetattr(fd, TCSANOW, &options);
}

static int checksum_valid(const char *line) {
    if (!line || line[0] != '$') return 0;
    const char *star = strchr(line, '*');
    if (!star || !isxdigit((unsigned char)star[1]) || !isxdigit((unsigned char)star[2])) return 0;
    unsigned checksum = 0;
    for (const char *p = line + 1; p < star; ++p) checksum ^= (unsigned char)*p;
    char expected_text[3] = {star[1], star[2], '\0'};
    return checksum == strtoul(expected_text, NULL, 16);
}

static int32_t coordinate_e7(const char *text, char hemisphere, int degree_digits) {
    if (!text || (int)strlen(text) <= degree_digits) return 0;
    char degrees_text[4] = {};
    memcpy(degrees_text, text, (size_t)degree_digits);
    double degrees = atof(degrees_text);
    double minutes = atof(text + degree_digits);
    double decimal = degrees + minutes / 60.0;
    if (hemisphere == 'S' || hemisphere == 'W') decimal = -decimal;
    return (int32_t)llround(decimal * 10000000.0);
}

static int parse_rmc(char *line, int32_t *latitude_e7, int32_t *longitude_e7,
                     uint32_t *utc_hhmmss) {
    if (!checksum_valid(line)) return 0;
    char *fields[12] = {}; unsigned count = 0;
    char *save = NULL;
    for (char *token = strtok_r(line, ",", &save); token && count < 12;
         token = strtok_r(NULL, ",", &save)) fields[count++] = token;
    if (count < 7 || (strcmp(fields[0], "$GPRMC") && strcmp(fields[0], "$GNRMC")) ||
        strcmp(fields[2], "A") || !fields[3][0] || !fields[4][0] ||
        !fields[5][0] || !fields[6][0]) return 0;
    *utc_hhmmss = (uint32_t)strtoul(fields[1], NULL, 10);
    *latitude_e7 = coordinate_e7(fields[3], fields[4][0], 2);
    *longitude_e7 = coordinate_e7(fields[5], fields[6][0], 3);
    return *latitude_e7 != 0 || *longitude_e7 != 0;
}

static vigil_gps_snapshot_t *open_gps_shared(void) {
    int fd = shm_open(VIGIL_GPS_SHM, O_RDWR | O_CREAT, 0660);
    if (fd == -1) { perror("shm_open GPS"); return NULL; }
    if (ftruncate(fd, sizeof(vigil_gps_snapshot_t)) == -1) {
        perror("ftruncate GPS"); close(fd); return NULL;
    }
    void *mapping = mmap(NULL, sizeof(vigil_gps_snapshot_t), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap GPS"); return NULL; }
    vigil_gps_snapshot_t *gps = static_cast<vigil_gps_snapshot_t *>(mapping);
    memset(gps, 0, sizeof(*gps)); gps->version = VIGIL_GPS_VERSION;
    gps->writer_pid = (uint32_t)getpid();
    return gps;
}

static void publish_fix(vigil_gps_snapshot_t *gps, int valid, int32_t latitude,
                        int32_t longitude, uint32_t utc, uint32_t accepted,
                        uint32_t rejected) {
    uint32_t generation = gps->generation;
    gps->generation = generation + 1U; __sync_synchronize();
    gps->version = VIGIL_GPS_VERSION; gps->fix_valid = valid ? 1U : 0U;
    if (valid) {
        gps->latitude_e7 = latitude; gps->longitude_e7 = longitude;
        gps->utc_hhmmss = utc; gps->update_ns = vigil_now_ns();
    }
    gps->sentences_valid = accepted; gps->sentences_rejected = rejected;
    gps->writer_pid = (uint32_t)getpid();
    __sync_synchronize(); gps->generation = generation + 2U;
}

int main(int argc, char **argv) {
    const char *configured = getenv("VIGIL_GPS_DEVICE");
    const char *device = argc > 1 ? argv[1] :
                         configured && configured[0] ? configured : "/dev/ser4";
    if (vigil_configure_sporadic(3, 35, 12, 5, 100) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    vigil_gps_snapshot_t *gps = open_gps_shared();
    if (!gps) return EXIT_FAILURE;
    fprintf(stdout, "PIPELINE stage=gps device=%s CPU3 policy=SPORADIC priority=35/12 "
                    "budget_ms=5 period_ms=100 shared=%s\n", device, VIGIL_GPS_SHM);
    fflush(stdout);

    int fd = -1;
    while (running && fd == -1) {
        fd = open(device, O_RDONLY | O_NONBLOCK);
        if (fd == -1) {
            fprintf(stderr, "GPS waiting for %s: %s\n", device, strerror(errno));
            vigil_heartbeat("/tmp/vigil_gps.hb"); delay(1000);
        }
    }
    if (!running) { munmap(gps, sizeof(*gps)); return EXIT_SUCCESS; }
    if (isatty(fd) && configure_uart(fd) == -1) {
        perror("configure GPS UART"); close(fd); munmap(gps, sizeof(*gps));
        return EXIT_FAILURE;
    }

    char line[256] = {}; size_t used = 0; uint32_t accepted = 0, rejected = 0;
    while (running) {
        vigil_heartbeat("/tmp/vigil_gps.hb");
        char input[128]; ssize_t amount = read(fd, input, sizeof(input));
        if (amount > 0) {
            for (ssize_t i = 0; i < amount; ++i) {
                char ch = input[i];
                if (ch == '\n') {
                    line[used] = '\0';
                    while (used && (line[used - 1] == '\r' || line[used - 1] == '\n'))
                        line[--used] = '\0';
                    char parsing[256]; memcpy(parsing, line, used + 1);
                    int32_t latitude = 0, longitude = 0; uint32_t utc = 0;
                    if (parse_rmc(parsing, &latitude, &longitude, &utc)) {
                        ++accepted;
                        publish_fix(gps, 1, latitude, longitude, utc, accepted, rejected);
                        fprintf(stdout, "GPS fix=VALID lat=%.7f lon=%.7f utc=%06u accepted=%u rejected=%u\n",
                                latitude / 10000000.0, longitude / 10000000.0,
                                utc, accepted, rejected); fflush(stdout);
                    } else if (used) {
                        ++rejected; publish_fix(gps, gps->fix_valid, gps->latitude_e7,
                                                gps->longitude_e7, gps->utc_hhmmss,
                                                accepted, rejected);
                    }
                    used = 0;
                } else if (used + 1 < sizeof(line)) line[used++] = ch;
                else { used = 0; ++rejected; }
            }
        } else if (amount == 0 && !isatty(fd)) {
            lseek(fd, 0, SEEK_SET); delay(200);
        } else if (amount == 0) {
            /* UART read timeouts are normal between one-Hz NMEA bursts. */
            delay(20);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            perror("read GPS"); break;
        } else delay(20);
    }
    munmap(gps, sizeof(*gps)); close(fd); unlink("/tmp/vigil_gps.hb");
    return EXIT_SUCCESS;
}
