#include "vigil_common.h"
#include "vigil_frame_buffer.h"
#include "vigil_protocol.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <net.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/mman.h>
#include <vector>

static volatile sig_atomic_t running = 1;
static void stop_handler(int) { running = 0; }
static volatile sig_atomic_t inference_phase = 0;
static void fatal_handler(int signal_number) {
    const char *signal_name = signal_number == SIGABRT ? "SIGABRT" :
                              signal_number == SIGSEGV ? "SIGSEGV" :
                              signal_number == SIGILL ? "SIGILL" : "UNKNOWN";
    const char *phase = inference_phase == 1 ? "snapshot" :
                        inference_phase == 2 ? "face_ncnn" :
                        inference_phase == 3 ? "eye_ncnn" :
                        inference_phase == 4 ? "yawn_ncnn" :
                        inference_phase == 5 ? "decision_ipc" : "unknown";
    char message[128];
    int length = snprintf(message, sizeof(message),
                          "INFERENCE_FATAL signal=%s phase=%s\n", signal_name, phase);
    if (length < 0) length = 0;
    if (length > (int)sizeof(message)) length = sizeof(message);
    write(STDERR_FILENO, message, strlen(message));
    _exit(128 + signal_number);
}

struct frame_copy {
    uint32_t id, width, height, stride;
    uint64_t capture_ns;
    std::vector<unsigned char> rgb;
};

struct point { float x, y; };
struct face {
    float x, y, w, h, score;
    point landmark[5];
};
struct box { int x, y, w, h; };

static bool snapshot(const vigil_frame_buffer_t *shared, frame_copy &frame) {
    for (int retry = 0; retry < 4; ++retry) {
        uint32_t before = shared->generation;
        if (before & 1U || shared->version != VIGIL_FRAME_VERSION) { delay(1); continue; }
        __sync_synchronize();
        uint32_t slot = shared->active_slot, bytes = shared->data_bytes;
        if (slot >= VIGIL_FRAME_SLOTS || !bytes || bytes > VIGIL_FRAME_SLOT_BYTES) return false;
        frame.id = shared->frame_id; frame.width = shared->width; frame.height = shared->height;
        frame.stride = shared->stride; frame.capture_ns = shared->capture_ns;
        frame.rgb.resize(bytes); memcpy(frame.rgb.data(), shared->slots[slot], bytes);
        __sync_synchronize();
        if (before == shared->generation && !(before & 1U)) return true;
    }
    return false;
}

static float overlap(const face &a, const face &b) {
    float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w), y2 = std::min(a.y + a.h, b.y + b.h);
    float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    return intersection / std::max(1.f, a.w * a.h + b.w * b.h - intersection);
}

static std::vector<face> detect_faces(ncnn::Net &net, const frame_copy &frame) {
    const int target_w = 320, target_h = 256;
    float wanted = (float)target_w / target_h;
    int crop_w = (int)frame.width, crop_h = (int)frame.height;
    if ((float)crop_w / crop_h > wanted) crop_w = (int)(crop_h * wanted);
    else crop_h = (int)(crop_w / wanted);
    int crop_x = ((int)frame.width - crop_w) / 2, crop_y = ((int)frame.height - crop_h) / 2;
    ncnn::Mat input = ncnn::Mat::from_pixels_roi_resize(frame.rgb.data(),
        ncnn::Mat::PIXEL_RGB2BGR, frame.width, frame.height, frame.stride,
        crop_x, crop_y, crop_w, crop_h, target_w, target_h);
    ncnn::Extractor extractor = net.create_extractor();
    /* MobileNetV3 has split squeeze/excitation branches. Keep intermediate
       tensors alive to avoid allocator recycling faults on the QNX backend. */
    extractor.set_light_mode(false);
    extractor.input("in0", input);
    ncnn::Mat cls[3], obj[3], bbox[3], kps[3];
    for (int i = 0; i < 3; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "out%d", i); if (extractor.extract(name, cls[i])) return {};
        snprintf(name, sizeof(name), "out%d", i + 3); if (extractor.extract(name, obj[i])) return {};
        snprintf(name, sizeof(name), "out%d", i + 6); if (extractor.extract(name, bbox[i])) return {};
        snprintf(name, sizeof(name), "out%d", i + 9); if (extractor.extract(name, kps[i])) return {};
    }
    std::vector<face> candidates;
    const int strides[3] = {8, 16, 32};
    for (int level = 0; level < 3; ++level) {
        int stride = strides[level], grid_w = target_w / stride;
        int anchors = cls[level].h > 1 ? cls[level].h : cls[level].w;
        for (int index = 0; index < anchors; ++index) {
            const float *class_row = cls[level].h > 1 ? cls[level].row(index) : (const float *)cls[level];
            const float *object_row = obj[level].h > 1 ? obj[level].row(index) : (const float *)obj[level];
            float score = sqrtf(std::max(0.f, class_row[0] * object_row[0]));
            if (score < .60f) continue;
            int gx = index % grid_w, gy = index / grid_w;
            const float *b = bbox[level].row(index), *p = kps[level].row(index);
            float cx = (gx + b[0]) * stride, cy = (gy + b[1]) * stride;
            float width = expf(b[2]) * stride, height = expf(b[3]) * stride;
            face found = {(cx - width / 2) * crop_w / target_w + crop_x,
                          (cy - height / 2) * crop_h / target_h + crop_y,
                          width * crop_w / target_w, height * crop_h / target_h, score, {}};
            for (int k = 0; k < 5; ++k) {
                found.landmark[k].x = (gx + p[k * 2]) * stride * crop_w / target_w + crop_x;
                found.landmark[k].y = (gy + p[k * 2 + 1]) * stride * crop_h / target_h + crop_y;
            }
            candidates.push_back(found);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const face &a, const face &b) {
        return a.score > b.score;
    });
    std::vector<face> kept;
    for (const face &candidate : candidates) {
        bool suppress = false;
        for (const face &chosen : kept) if (overlap(candidate, chosen) > .30f) { suppress = true; break; }
        if (!suppress) kept.push_back(candidate);
        if (kept.size() >= 8) break;
    }
    return kept;
}

/* Reliable demo fallback for a dashboard camera mounted in a fixed position.
   The driver keeps their face in the central guide; trained eye/yawn networks
   still execute on QNX. Approximate landmarks preserve the existing crop and
   temporal-decision pipeline without invoking the failing YuNet graph. */
static std::vector<face> fixed_camera_face(const frame_copy &frame) {
    const float w = frame.width * .52f;
    const float h = frame.height * .78f;
    const float x = (frame.width - w) * .5f;
    const float y = frame.height * .08f;
    face f = {x, y, w, h, 1.f, {}};
    f.landmark[0] = {x + w * .32f, y + h * .36f};
    f.landmark[1] = {x + w * .68f, y + h * .36f};
    f.landmark[2] = {x + w * .50f, y + h * .54f};
    f.landmark[3] = {x + w * .38f, y + h * .73f};
    f.landmark[4] = {x + w * .62f, y + h * .73f};
    return {f};
}

static box bounded(float x, float y, float w, float h, const frame_copy &frame) {
    int x1 = std::max(0, (int)x), y1 = std::max(0, (int)y);
    int x2 = std::min((int)frame.width, (int)(x + w));
    int y2 = std::min((int)frame.height, (int)(y + h));
    return {x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1)};
}

static box center_224(box value, const frame_copy &frame) {
    // Windows first makes a 256x256 ROI and then center-crops 224x224.
    float margin_x = value.w * .0625f, margin_y = value.h * .0625f;
    return bounded(value.x + margin_x, value.y + margin_y,
                   value.w - 2 * margin_x, value.h - 2 * margin_y, frame);
}

static bool classify(ncnn::Net &net, const frame_copy &frame, box roi, float logits[2]) {
    roi = center_224(roi, frame);
    if (roi.w < 8 || roi.h < 8) return false;
    ncnn::Mat input = ncnn::Mat::from_pixels_roi_resize(frame.rgb.data(), ncnn::Mat::PIXEL_RGB,
        frame.width, frame.height, frame.stride, roi.x, roi.y, roi.w, roi.h, 224, 224);
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float norm[3] = {1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f};
    input.substract_mean_normalize(mean, norm);
    ncnn::Extractor extractor = net.create_extractor(); extractor.input("in0", input);
    ncnn::Mat output;
    if (extractor.extract("out0", output) || output.total() < 2) return false;
    logits[0] = output[0]; logits[1] = output[1]; return true;
}

static float probability(float a, float b) {
    float maximum = std::max(a, b), ea = expf(a - maximum), eb = expf(b - maximum);
    return eb / (ea + eb);
}

struct head_tracker {
    std::vector<float> calibration[3]; float neutral[3] = {}; bool ready = false;
    std::deque<bool> history;
    bool update(const face &value, float &yaw, float &pitch, float &roll) {
        const point &right = value.landmark[0], &left = value.landmark[1];
        const point &nose = value.landmark[2], &rm = value.landmark[3], &lm = value.landmark[4];
        float ex = (right.x + left.x) / 2, ey = (right.y + left.y) / 2;
        float mx = (rm.x + lm.x) / 2, my = (rm.y + lm.y) / 2;
        float eye_distance = hypotf(right.x - left.x, right.y - left.y);
        float face_height = hypotf(ex - mx, ey - my);
        if (eye_distance < 4 || face_height < 4) return false;
        float current[3] = {(nose.x - ex) / eye_distance, (nose.y - ey) / face_height,
                            atan2f(left.y - right.y, left.x - right.x) * 57.2957795f};
        if (!ready) {
            for (int i = 0; i < 3; ++i) calibration[i].push_back(current[i]);
            if (calibration[0].size() >= 10) {
                for (int i = 0; i < 3; ++i) {
                    std::sort(calibration[i].begin(), calibration[i].end());
                    neutral[i] = calibration[i][calibration[i].size() / 2];
                }
                ready = true;
            }
            yaw = pitch = roll = 0; return false;
        }
        yaw = current[0] - neutral[0]; pitch = current[1] - neutral[1];
        roll = fmodf(current[2] - neutral[2] + 540.f, 360.f) - 180.f;
        bool deviated = fabsf(yaw) >= .18f || fabsf(pitch) >= .15f || fabsf(roll) >= 12.f;
        history.push_back(deviated); if (history.size() > 6) history.pop_front();
        return history.size() == 6 && std::count(history.begin(), history.end(), true) >= 4;
    }
};

static int load_net(ncnn::Net &net, const char *root, const char *stem) {
    char param[512], model[512];
    snprintf(param, sizeof(param), "%s/%s.ncnn.param", root, stem);
    snprintf(model, sizeof(model), "%s/%s.ncnn.bin", root, stem);
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_int8_inference = false;
    return net.load_param(param) || net.load_model(model) ? -1 : 0;
}

int main(int argc, char **argv) {
    const char *configured_root = getenv("VIGIL_MODEL_ROOT");
    const char *model_root = argc > 1 ? argv[1] :
                             (configured_root && configured_root[0] ? configured_root : "./models");
    if (vigil_configure_sporadic(3, 40, 15, 150, 200) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    signal(SIGABRT, fatal_handler); signal(SIGSEGV, fatal_handler); signal(SIGILL, fatal_handler);
    const char *face_mode_text = getenv("VIGIL_FACE_MODE");
    const bool fixed_face_mode = face_mode_text && strcmp(face_mode_text, "fixed") == 0;
    ncnn::Net face_net, eye_net, yawn_net;
    if ((!fixed_face_mode && load_net(face_net, model_root, "face_yunet")) ||
        load_net(eye_net, model_root, "eye_state") ||
        load_net(yawn_net, model_root, "yawn_frame")) {
        fprintf(stderr, "Unable to load ncnn models from %s\n", model_root); return EXIT_FAILURE;
    }
    int fd = shm_open(VIGIL_FRAME_SHM, O_RDONLY, 0);
    if (fd == -1) { perror("shm_open RGB frames"); return EXIT_FAILURE; }
    void *mapping = mmap(NULL, sizeof(vigil_frame_buffer_t), PROT_READ, MAP_SHARED, fd, 0); close(fd);
    if (mapping == MAP_FAILED) { perror("mmap RGB frames"); return EXIT_FAILURE; }
    const vigil_frame_buffer_t *shared = static_cast<const vigil_frame_buffer_t *>(mapping);
    int decision = -1; uint32_t sequence = 0, last_frame = 0; uint64_t last_sample = 0;
    std::deque<std::pair<float, float>> yawn_window; int yawn_streak = 0;
    head_tracker head;
    fprintf(stdout, "PIPELINE stage=inference shared=%s CPU3 policy=SPORADIC priority=40/15 "
                    "budget_ms=150 period_ms=200 runtime=ncnn models=%s face_mode=%s\n",
                    VIGIL_FRAME_SHM, model_root, fixed_face_mode ? "fixed" : "yunet");
    fflush(stdout);
    while (running) {
        frame_copy frame;
        inference_phase = 1;
        if (!snapshot(shared, frame) || frame.id == last_frame ||
            (last_sample && vigil_now_ns() - last_sample < 1000000000ULL)) {
            vigil_heartbeat("/tmp/vigil_inference.hb"); delay(10); continue;
        }
        last_frame = frame.id; last_sample = vigil_now_ns(); uint64_t started = last_sample;
        inference_phase = 2;
        fprintf(stderr, "INFERENCE_TRACE frame=%u phase=%s\n", frame.id,
                fixed_face_mode ? "face_fixed" : "face_ncnn"); fflush(stderr);
        vigil_heartbeat("/tmp/vigil_inference.hb");
        std::vector<face> faces = fixed_face_mode ? fixed_camera_face(frame) :
                                                   detect_faces(face_net, frame);
        vigil_heartbeat("/tmp/vigil_inference.hb");
        float eye_closed = 0, yawn_probability = 0, yaw = 0, pitch = 0, roll = 0;
        uint32_t flags = VIGIL_MODEL_READY;
        if (faces.size() == 1) {
            const face &f = faces[0]; flags |= VIGIL_FACE_VALID;
            float eye_logits[2][2]; bool eyes_ok = true;
            for (int i = 0; i < 2; ++i) {
                inference_phase = 3;
                point e = f.landmark[i]; box roi = bounded(e.x - .18f*f.w, e.y - .12f*f.h,
                                                            .36f*f.w, .24f*f.h, frame);
                eyes_ok &= classify(eye_net, frame, roi, eye_logits[i]);
                vigil_heartbeat("/tmp/vigil_inference.hb");
            }
            if (eyes_ok) eye_closed = (probability(eye_logits[0][1], eye_logits[0][0]) +
                                       probability(eye_logits[1][1], eye_logits[1][0])) / 2;
            point rm = f.landmark[3], lm = f.landmark[4];
            float mx = (rm.x + lm.x) / 2, my = (rm.y + lm.y) / 2;
            float mouth_width = std::max(hypotf(rm.x-lm.x, rm.y-lm.y) * 1.9f, .55f*f.w);
            box mouth = bounded(mx-mouth_width/2, my-.17f*f.h, mouth_width, .44f*f.h, frame);
            float yawn_logits[2];
            inference_phase = 4;
            if (classify(yawn_net, frame, mouth, yawn_logits)) {
                yawn_window.emplace_back(yawn_logits[0], yawn_logits[1]);
                if (yawn_window.size() > 16) yawn_window.pop_front();
                if (yawn_window.size() >= 8) {
                    std::vector<float> evidence;
                    for (const auto &item : yawn_window) evidence.push_back(item.second-item.first);
                    std::sort(evidence.begin(), evidence.end(), std::greater<float>());
                    unsigned count = (evidence.size()+3)/4; float total = 0;
                    for (unsigned i=0; i<count; ++i) total += evidence[i];
                    yawn_probability = 1.f/(1.f+expf(-total/count));
                    yawn_streak = yawn_probability >= .45f ? yawn_streak+1 : 0;
                    if (yawn_streak < 2) yawn_probability = 0;
                }
            }
            vigil_heartbeat("/tmp/vigil_inference.hb");
            if (head.update(f, yaw, pitch, roll)) flags |= VIGIL_HEAD_CUE;
        } else {
            yawn_window.clear(); yawn_streak = 0; head = head_tracker();
        }
        ++sequence; if (sequence == 1) flags |= VIGIL_SENDER_RESET;
        vigil_ipc_message_t message = {1, 0, sequence, vigil_now_ns(), eye_closed,
            yawn_probability, (uint32_t)((vigil_now_ns()-started)/1000ULL), flags};
        inference_phase = 5;
        while (running && decision == -1) { decision = name_open(VIGIL_DECISION_SERVICE, 0); if (decision == -1) delay(100); }
        vigil_ipc_reply_t reply = {};
        if (decision != -1 && MsgSend(decision, &message, sizeof(message), &reply, sizeof(reply)) == -1) {
            name_close(decision); decision = -1;
        } else {
            fprintf(stdout, "INFERENCE frame=%u seq=%u faces=%u eye=%.3f yawn=%.3f "
                    "head=%u yaw=%.3f pitch=%.3f roll=%.1f inference_ms=%.1f state=%u\n",
                    frame.id, sequence, (unsigned)faces.size(), eye_closed, yawn_probability,
                    !!(flags & VIGIL_HEAD_CUE), yaw, pitch, roll, message.inference_us/1000.f, reply.state);
            fflush(stdout);
        }
        vigil_heartbeat("/tmp/vigil_inference.hb");
        inference_phase = 0;
    }
    if (decision != -1) name_close(decision);
    munmap(mapping, sizeof(vigil_frame_buffer_t)); unlink("/tmp/vigil_inference.hb");
    return EXIT_SUCCESS;
}
