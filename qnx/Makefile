QCC ?= q++
QNX_VARIANT ?= -Vgcc_ntoaarch64le
CXXFLAGS = -std=c++17 -D_QNX_SOURCE -O2 -Wall -Wextra -Iinclude
LDFLAGS = -lsocket
NCNN_ROOT ?= third_party/ncnn/install
NCNN_CXXFLAGS = -I$(NCNN_ROOT)/include/ncnn
NCNN_LDFLAGS = -L$(NCNN_ROOT)/lib -lncnn -lm
ORT_ROOT ?= third_party/onnxruntime
ORT_BUILD ?= $(ORT_ROOT)/build-qnx-minimal
ORT_CXXFLAGS = -I$(ORT_ROOT)/include/onnxruntime/core/session
ORT_STATIC_LIBS = $(shell find $(ORT_BUILD) -type f -name '*.a' | sort)
BIN = build/vigil_ingest build/vigil_decision build/vigil_logger build/vigil_supervisor build/vigil_stress build/vigil_monitor build/vigil_video_rx build/vigil_frame_monitor build/vigil_gps build/vigil_alert build/vigil_inference build/vigil_ort_inference

all: $(BIN)

build:
	mkdir -p build

build/protocol.o: src/protocol.cpp include/vigil_protocol.h include/protocol_helpers.h | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) -c $< -o $@

build/vigil_ingest: src/vigil_ingest.cpp build/protocol.o | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/vigil_decision: src/vigil_decision.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_logger: src/vigil_logger.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_supervisor: src/vigil_supervisor.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_stress: src/vigil_stress.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_monitor: src/vigil_monitor.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_video_rx: src/vigil_video_rx.cpp build/protocol.o | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -limg

build/vigil_frame_monitor: src/vigil_frame_monitor.cpp build/protocol.o | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_gps: src/vigil_gps.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@

build/vigil_alert: src/vigil_alert.cpp | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/vigil_inference: src/vigil_inference.cpp $(NCNN_ROOT)/lib/libncnn.a | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $(NCNN_CXXFLAGS) $< -o $@ $(NCNN_LDFLAGS)

build/vigil_ort_inference: src/vigil_ort_inference.cpp $(ORT_BUILD)/libonnxruntime_session.a | build
	$(QCC) $(QNX_VARIANT) $(CXXFLAGS) $(ORT_CXXFLAGS) $< -o $@ -Wl,--start-group $(ORT_STATIC_LIBS) -Wl,--end-group -lm -lsocket

clean:
	rm -rf build

.PHONY: all clean
