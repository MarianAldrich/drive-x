#include "vigil_common.h"
#include "vigil_protocol.h"
#include "protocol_helpers.h"

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/socket.h>

static volatile sig_atomic_t running = 1;
static volatile sig_atomic_t simulate_loss = 0;

static void signal_handler(int signal) {
    if (signal == SIGUSR1) simulate_loss = 15; /* 15 x 200 ms receive cycles = 3 s */
    else running = 0;
}

static int connect_decision(void) {
    int connection;
    while (running && (connection = name_open(VIGIL_DECISION_SERVICE, 0)) == -1)
        delay(100);
    return running ? connection : -1;
}

static int valid_packet(const vigil_evidence_wire_t *packet) {
    if (ntohl(packet->magic) != VIGIL_MAGIC || packet->version != VIGIL_VERSION ||
        packet->type != VIGIL_TYPE_EVIDENCE || ntohs(packet->length) != sizeof(*packet)) return 0;
    uint32_t expected = vigil_crc32(packet, sizeof(*packet) - sizeof(packet->crc32));
    if (ntohl(packet->crc32) != expected) return 0;
    float eye = vigil_ntoh_float(packet->eye_closed_bits);
    float yawn = vigil_ntoh_float(packet->yawn_bits);
    return isfinite(eye) && eye >= 0.f && eye <= 1.f && isfinite(yawn) && yawn >= 0.f && yawn <= 1.f;
}

static vigil_status_wire_t status_wire(const vigil_ipc_reply_t &reply) {
    vigil_status_wire_t packet = {};
    packet.magic = htonl(VIGIL_MAGIC);
    packet.version = VIGIL_VERSION;
    packet.type = VIGIL_TYPE_STATUS;
    packet.length = htons(sizeof(packet));
    packet.sequence = htonl(reply.sequence);
    packet.qnx_receive_ns = vigil_hton64(reply.receive_ns);
    packet.decision_us = htonl(reply.decision_us);
    packet.state = htonl(reply.state);
    packet.flags = htonl(reply.flags);
    packet.rto_us = htonl(reply.rto_us);
    packet.crc32 = htonl(vigil_crc32(&packet, sizeof(packet) - sizeof(packet.crc32)));
    return packet;
}

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : VIGIL_UDP_PORT;
    if (port < 1 || port > 65535 || vigil_configure_thread(1, 60) == -1) return EXIT_FAILURE;
    signal(SIGINT, signal_handler); signal(SIGTERM, signal_handler); signal(SIGUSR1, signal_handler);
    FILE *pidfile = fopen("/tmp/vigil_ingest.pid", "w");
    if (pidfile) { fprintf(pidfile, "%d\n", getpid()); fclose(pidfile); }

    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd == -1) { perror("socket"); return EXIT_FAILURE; }
    struct timeval timeout = {0, 200000};
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address = {};
    address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_ANY); address.sin_port = htons(port);
    if (bind(socket_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == -1) {
        perror("bind"); return EXIT_FAILURE;
    }
    int decision = connect_decision();
    uint32_t last_sequence = 0;
    int have_sequence = 0, have_sender = 0;
    sockaddr_in sender = {}; socklen_t sender_length = sizeof(sender);
    unsigned long rejected = 0;
    fprintf(stdout, "PIPELINE stage=ingest UDP=%d CPU1 policy=FIFO priority=60 IPC=MsgSend\n",
            port); fflush(stdout);

    while (running) {
        vigil_evidence_wire_t wire = {};
        sockaddr_in candidate = {}; socklen_t candidate_length = sizeof(candidate);
        ssize_t received = recvfrom(socket_fd, &wire, sizeof(wire), 0,
                                    reinterpret_cast<sockaddr *>(&candidate), &candidate_length);
        if (simulate_loss > 0) {
            /* Simulate transport loss while proving the ingest service is alive. */
            vigil_heartbeat("/tmp/vigil_ingest.hb");
            --simulate_loss;
            continue;
        }

        vigil_ipc_message_t message = {};
        if (received == (ssize_t)sizeof(wire) && valid_packet(&wire)) {
            uint32_t sequence = ntohl(wire.sequence), flags = ntohl(wire.flags);
            if (have_sequence && !(flags & VIGIL_SENDER_RESET) && (int32_t)(sequence - last_sequence) <= 0) {
                ++rejected; continue;
            }
            sender = candidate; sender_length = candidate_length; have_sender = 1;
            last_sequence = sequence; have_sequence = 1;
            message.kind = 1; message.sequence = sequence; message.receive_ns = vigil_now_ns();
            message.eye_closed = vigil_ntoh_float(wire.eye_closed_bits);
            message.yawn = vigil_ntoh_float(wire.yawn_bits);
            message.inference_us = ntohl(wire.inference_us); message.flags = flags;
        } else if (received >= 0) {
            ++rejected; continue;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            message.kind = 2; message.sequence = last_sequence; message.receive_ns = vigil_now_ns();
        } else if (errno == EINTR) continue;
        else { perror("recvfrom"); continue; }

        if (decision == -1) decision = connect_decision();
        vigil_ipc_reply_t reply = {};
        if (decision != -1 && MsgSend(decision, &message, sizeof(message), &reply, sizeof(reply)) == -1) {
            name_close(decision); decision = connect_decision();
            if (decision == -1 || MsgSend(decision, &message, sizeof(message), &reply, sizeof(reply)) == -1) continue;
        }
        if (have_sender && (message.kind == 1 || reply.state == VIGIL_SENSOR_LOST)) {
            vigil_status_wire_t status = status_wire(reply);
            sendto(socket_fd, &status, sizeof(status), 0, reinterpret_cast<sockaddr *>(&sender), sender_length);
        }
        vigil_heartbeat("/tmp/vigil_ingest.hb");
    }
    fprintf(stdout, "vigil_ingest stopped; rejected=%lu\n", rejected);
    if (decision != -1) name_close(decision);
    close(socket_fd); unlink("/tmp/vigil_ingest.pid");
    return EXIT_SUCCESS;
}
