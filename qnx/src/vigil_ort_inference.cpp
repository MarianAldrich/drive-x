#include "vigil_common.h"
#include "vigil_frame_buffer.h"
#include "vigil_protocol.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/mman.h>
#include <vector>

static volatile sig_atomic_t running = 1;
static volatile sig_atomic_t inference_phase = 0;
static void stop_handler(int) { running = 0; }
static void fatal_handler(int signal_number) {
    const char *phase = inference_phase == 1 ? "snapshot" :
                        inference_phase == 2 ? "face_ort" :
                        inference_phase == 3 ? "eye_ort" :
                        inference_phase == 4 ? "yawn_ort" :
                        inference_phase == 5 ? "decision_ipc" : "unknown";
    char text[128];
    snprintf(text, sizeof(text), "INFERENCE_FATAL signal=%d phase=%s\n", signal_number, phase);
    write(STDERR_FILENO, text, strlen(text));
    _exit(128 + signal_number);
}

struct frame_copy { uint32_t id, width, height, stride; std::vector<unsigned char> rgb; };
struct point { float x, y; };
struct face { float x, y, w, h, score; point landmark[5]; };
struct box { int x, y, w, h; };

static bool snapshot(const vigil_frame_buffer_t *shared, frame_copy &frame) {
    for (int retry = 0; retry < 4; ++retry) {
        uint32_t before = shared->generation;
        if ((before & 1U) || shared->version != VIGIL_FRAME_VERSION) { delay(1); continue; }
        __sync_synchronize();
        uint32_t slot = shared->active_slot, bytes = shared->data_bytes;
        if (slot >= VIGIL_FRAME_SLOTS || !bytes || bytes > VIGIL_FRAME_SLOT_BYTES) return false;
        frame.id = shared->frame_id; frame.width = shared->width; frame.height = shared->height;
        frame.stride = shared->stride; frame.rgb.resize(bytes);
        memcpy(frame.rgb.data(), shared->slots[slot], bytes);
        __sync_synchronize();
        if (before == shared->generation && !(before & 1U)) return true;
    }
    return false;
}

static box bounded(float x, float y, float w, float h, const frame_copy &frame) {
    int x1 = std::max(0, (int)x), y1 = std::max(0, (int)y);
    int x2 = std::min((int)frame.width, (int)(x + w));
    int y2 = std::min((int)frame.height, (int)(y + h));
    return {x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1)};
}

static box center_224(box value, const frame_copy &frame) {
    return bounded(value.x + value.w * .0625f, value.y + value.h * .0625f,
                   value.w * .875f, value.h * .875f, frame);
}

static float overlap(const face &a, const face &b) {
    float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w), y2 = std::min(a.y + a.h, b.y + b.h);
    float intersection = std::max(0.f, x2-x1) * std::max(0.f, y2-y1);
    return intersection / std::max(1.f, a.w*a.h + b.w*b.h - intersection);
}

/* RGB frame -> NCHW. YuNet was trained for OpenCV BGR pixels; the classifier
 * models use normalized RGB ImageNet pixels. */
static void resize_to_nchw(const frame_copy &frame, box roi, int out_w, int out_h,
                           bool bgr, bool normalize, std::vector<float> &output) {
    output.resize((size_t)3 * out_w * out_h);
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float scale[3] = {1.f/58.395f, 1.f/57.12f, 1.f/57.375f};
    for (int y = 0; y < out_h; ++y) {
        int sy = std::min(roi.y + roi.h - 1, roi.y + (y * roi.h) / out_h);
        for (int x = 0; x < out_w; ++x) {
            int sx = std::min(roi.x + roi.w - 1, roi.x + (x * roi.w) / out_w);
            const unsigned char *pixel = frame.rgb.data() + (size_t)sy * frame.stride + sx * 3;
            for (int c = 0; c < 3; ++c) {
                int source_channel = bgr ? 2-c : c;
                float value = pixel[source_channel];
                if (normalize) value = (value - mean[c]) * scale[c];
                output[(size_t)c*out_w*out_h + y*out_w + x] = value;
            }
        }
    }
}

class ort_model {
public:
    ort_model(Ort::Env &env, const char *path) {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_.reset(new Ort::Session(env, path, options_));
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1) throw std::runtime_error("model must have one input");
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
    }
    std::vector<Ort::Value> run(std::vector<float> &input) {
        size_t expected = 1;
        for (int64_t dimension : shape_) expected *= (size_t)dimension;
        if (input.size() != expected) throw std::runtime_error("input tensor size mismatch");
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                              shape_.data(), shape_.size());
        std::vector<const char *> names;
        for (const std::string &name : output_names_) names.push_back(name.c_str());
        const char *input_name = input_name_.c_str();
        return session_->Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                             names.data(), names.size());
    }
private:
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> shape_;
};

static std::vector<face> detect_faces(ort_model &model, const frame_copy &frame) {
    std::vector<float> input;
    resize_to_nchw(frame, {0, 0, (int)frame.width, (int)frame.height}, 640, 640, true, false, input);
    std::vector<Ort::Value> output = model.run(input);
    if (output.size() != 12) throw std::runtime_error("unexpected YuNet output count");
    std::vector<face> candidates;
    const int strides[] = {8, 16, 32};
    for (int level = 0; level < 3; ++level) {
        const float *cls = output[level].GetTensorData<float>();
        const float *obj = output[level+3].GetTensorData<float>();
        const float *bbox = output[level+6].GetTensorData<float>();
        const float *kps = output[level+9].GetTensorData<float>();
        const int grid_w = 640 / strides[level], total = grid_w * grid_w;
        for (int index = 0; index < total; ++index) {
            float score = sqrtf(std::max(0.f, cls[index] * obj[index]));
            if (score < .60f) continue;
            int gx = index % grid_w, gy = index / grid_w; const float *b = bbox + index*4;
            float cx = (gx+b[0])*strides[level], cy = (gy+b[1])*strides[level];
            float width = expf(b[2])*strides[level], height = expf(b[3])*strides[level];
            face found = {(cx-width/2)*frame.width/640.f, (cy-height/2)*frame.height/640.f,
                          width*frame.width/640.f, height*frame.height/640.f, score, {}};
            const float *p = kps + index*10;
            for (int k = 0; k < 5; ++k) {
                found.landmark[k] = {(gx+p[k*2])*strides[level]*frame.width/640.f,
                                     (gy+p[k*2+1])*strides[level]*frame.height/640.f};
            }
            candidates.push_back(found);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const face &a, const face &b) { return a.score > b.score; });
    std::vector<face> kept;
    for (const face &candidate : candidates) {
        bool suppress = false; for (const face &chosen : kept) if (overlap(candidate, chosen) > .30f) suppress = true;
        if (!suppress) kept.push_back(candidate);
        if (kept.size() == 8) break;
    }
    return kept;
}

static bool classify(ort_model &model, const frame_copy &frame, box roi, float logits[2]) {
    roi = center_224(roi, frame); if (roi.w < 8 || roi.h < 8) return false;
    std::vector<float> input; resize_to_nchw(frame, roi, 224, 224, false, true, input);
    std::vector<Ort::Value> output = model.run(input);
    if (output.size() != 1 || output[0].GetTensorTypeAndShapeInfo().GetElementCount() != 2) return false;
    const float *values = output[0].GetTensorData<float>(); logits[0] = values[0]; logits[1] = values[1]; return true;
}

static float probability(float a, float b) { float m=std::max(a,b); return expf(b-m)/(expf(a-m)+expf(b-m)); }

struct head_tracker {
    std::vector<float> calibration[3]; float neutral[3] = {}; bool ready = false; std::deque<bool> history;
    bool update(const face &v, float &yaw, float &pitch, float &roll) {
        const point &right=v.landmark[0], &left=v.landmark[1], &nose=v.landmark[2], &rm=v.landmark[3], &lm=v.landmark[4];
        float ex=(right.x+left.x)/2, ey=(right.y+left.y)/2, mx=(rm.x+lm.x)/2, my=(rm.y+lm.y)/2;
        float eye=hypotf(right.x-left.x,right.y-left.y), height=hypotf(ex-mx,ey-my); if (eye<4 || height<4) return false;
        float current[3] = {(nose.x-ex)/eye,(nose.y-ey)/height,atan2f(left.y-right.y,left.x-right.x)*57.2957795f};
        if (!ready) { for(int i=0;i<3;++i) calibration[i].push_back(current[i]); if(calibration[0].size()>=10) { for(int i=0;i<3;++i) { std::sort(calibration[i].begin(),calibration[i].end()); neutral[i]=calibration[i][calibration[i].size()/2]; } ready=true; } yaw=pitch=roll=0; return false; }
        yaw=current[0]-neutral[0]; pitch=current[1]-neutral[1]; roll=fmodf(current[2]-neutral[2]+540.f,360.f)-180.f;
        bool deviated=fabsf(yaw)>=.18f || fabsf(pitch)>=.15f || fabsf(roll)>=12.f; history.push_back(deviated); if(history.size()>6) history.pop_front(); return history.size()==6 && std::count(history.begin(),history.end(),true)>=4;
    }
};

int main(int argc, char **argv) {
    const char *configured = getenv("VIGIL_MODEL_ROOT");
    const char *root = argc > 1 ? argv[1] : (configured && configured[0] ? configured : "./models_ort");
    if (vigil_configure_sporadic(3, 40, 15, 250, 1000) == -1) return EXIT_FAILURE;
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler); signal(SIGABRT, fatal_handler); signal(SIGSEGV, fatal_handler); signal(SIGILL, fatal_handler);
    try {
        char face_path[512], eye_path[512], yawn_path[512];
        snprintf(face_path,sizeof(face_path),"%s/face_yunet.ort",root); snprintf(eye_path,sizeof(eye_path),"%s/eye_state.ort",root); snprintf(yawn_path,sizeof(yawn_path),"%s/yawn_frame.ort",root);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"vigil-ort"); ort_model face_model(env,face_path), eye_model(env,eye_path), yawn_model(env,yawn_path);
        int fd=shm_open(VIGIL_FRAME_SHM,O_RDONLY,0); if(fd==-1) { perror("shm_open RGB frames"); return EXIT_FAILURE; }
        void *mapping=mmap(NULL,sizeof(vigil_frame_buffer_t),PROT_READ,MAP_SHARED,fd,0); close(fd); if(mapping==MAP_FAILED) { perror("mmap RGB frames"); return EXIT_FAILURE; }
        const vigil_frame_buffer_t *shared=static_cast<const vigil_frame_buffer_t *>(mapping); int decision=-1; uint32_t sequence=0,last_frame=0; uint64_t last_sample=0; std::deque<std::pair<float,float>> yawn_window; int yawn_streak=0; head_tracker head;
        fprintf(stdout,"PIPELINE stage=inference shared=%s CPU3 policy=SPORADIC priority=40/15 budget_ms=250 period_ms=1000 runtime=onnxruntime-%s models=%s\n",VIGIL_FRAME_SHM,OrtGetApiBase()->GetVersionString(),root); fflush(stdout);
        while(running) {
            frame_copy frame; inference_phase=1;
            if(!snapshot(shared,frame) || frame.id==last_frame || (last_sample && vigil_now_ns()-last_sample<1000000000ULL)) { vigil_heartbeat("/tmp/vigil_inference.hb"); delay(10); continue; }
            last_frame=frame.id; last_sample=vigil_now_ns(); uint64_t started=last_sample; inference_phase=2; fprintf(stderr,"INFERENCE_TRACE frame=%u phase=face_ort\n",frame.id); fflush(stderr);
            std::vector<face> faces=detect_faces(face_model,frame); vigil_heartbeat("/tmp/vigil_inference.hb"); float eye_closed=0,yawn=0,yaw=0,pitch=0,roll=0; uint32_t flags=VIGIL_MODEL_READY;
            if(faces.size()==1) { const face &f=faces[0]; flags|=VIGIL_FACE_VALID; float eye_logits[2][2]; bool eyes_ok=true;
                for(int i=0;i<2;++i) { inference_phase=3; point e=f.landmark[i]; box roi=bounded(e.x-.18f*f.w,e.y-.12f*f.h,.36f*f.w,.24f*f.h,frame); eyes_ok &= classify(eye_model,frame,roi,eye_logits[i]); vigil_heartbeat("/tmp/vigil_inference.hb"); }
                if(eyes_ok) eye_closed=(probability(eye_logits[0][1],eye_logits[0][0])+probability(eye_logits[1][1],eye_logits[1][0]))/2;
                point rm=f.landmark[3],lm=f.landmark[4]; float mx=(rm.x+lm.x)/2,my=(rm.y+lm.y)/2; float mouth_width=std::max(hypotf(rm.x-lm.x,rm.y-lm.y)*1.9f,.55f*f.w); float yawn_logits[2]; inference_phase=4;
                if(classify(yawn_model,frame,bounded(mx-mouth_width/2,my-.17f*f.h,mouth_width,.44f*f.h,frame),yawn_logits)) { yawn_window.emplace_back(yawn_logits[0],yawn_logits[1]); if(yawn_window.size()>16)yawn_window.pop_front(); if(yawn_window.size()>=8) { std::vector<float> evidence; for(const auto &item:yawn_window)evidence.push_back(item.second-item.first); std::sort(evidence.begin(),evidence.end(),std::greater<float>()); unsigned count=(evidence.size()+3)/4; float total=0; for(unsigned i=0;i<count;++i)total+=evidence[i]; yawn=1.f/(1.f+expf(-total/count)); yawn_streak=yawn>=.45f?yawn_streak+1:0; if(yawn_streak<2)yawn=0; } }
                if(head.update(f,yaw,pitch,roll)) flags|=VIGIL_HEAD_CUE;
            } else { yawn_window.clear(); yawn_streak=0; head=head_tracker(); }
            ++sequence; if(sequence==1) flags|=VIGIL_SENDER_RESET; vigil_ipc_message_t message={1,0,sequence,vigil_now_ns(),eye_closed,yawn,(uint32_t)((vigil_now_ns()-started)/1000ULL),flags}; inference_phase=5;
            while(running && decision==-1) { decision=name_open(VIGIL_DECISION_SERVICE,0); if(decision==-1) delay(100); }
            vigil_ipc_reply_t reply={}; if(decision!=-1 && MsgSend(decision,&message,sizeof(message),&reply,sizeof(reply))==-1) { name_close(decision); decision=-1; } else { fprintf(stdout,"INFERENCE frame=%u seq=%u faces=%u eye=%.3f yawn=%.3f head=%u yaw=%.3f pitch=%.3f roll=%.1f inference_ms=%.1f state=%u\n",frame.id,sequence,(unsigned)faces.size(),eye_closed,yawn,!!(flags&VIGIL_HEAD_CUE),yaw,pitch,roll,message.inference_us/1000.f,reply.state); fflush(stdout); }
            vigil_heartbeat("/tmp/vigil_inference.hb"); inference_phase=0;
        }
        if (decision != -1) name_close(decision);
        munmap(mapping, sizeof(vigil_frame_buffer_t));
        unlink("/tmp/vigil_inference.hb");
        return EXIT_SUCCESS;
    } catch(const std::exception &error) { fprintf(stderr,"INFERENCE_ERROR phase=%d detail=%s\n",(int)inference_phase,error.what()); return EXIT_FAILURE; }
}
