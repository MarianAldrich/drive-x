// Run a fixed test tensor on QNX and compare every output to the host reference.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> read_floats(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open " + path);
  const auto bytes = file.tellg();
  if (bytes <= 0 || bytes % sizeof(float)) throw std::runtime_error("Invalid float file " + path);
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(values.data()), bytes))
    throw std::runtime_error("Cannot read " + path);
  return values;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: ort_smoke /path/models_ort/eye_state (without extension)\n";
    return 2;
  }
  try {
    const std::string prefix(argv[1]);
    std::cout << "ORT version=" << OrtGetApiBase()->GetVersionString() << " phase=load" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qnx-smoke");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, (prefix + ".ort").c_str(), options);
    if (session.GetInputCount() != 1) throw std::runtime_error("Expected one input");
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    const char* inputs[] = {input_name.get()};
    auto input_type = session.GetInputTypeInfo(0);
    auto info = input_type.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto data = read_floats(prefix + ".input.f32");
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != data.size()) throw std::runtime_error("Input tensor mismatch");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
    std::vector<Ort::AllocatedStringPtr> names;
    std::vector<const char*> outputs;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      names.push_back(session.GetOutputNameAllocated(i, allocator));
      outputs.push_back(names.back().get());
    }
    std::cout << "ORT phase=run input=" << inputs[0] << " elements=" << data.size() << std::endl;
    const auto start = std::chrono::steady_clock::now();
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs.data(), outputs.size());
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    bool pass = true;
    for (size_t i = 0; i < result.size(); ++i) {
      auto expected = read_floats(prefix + ".output" + std::to_string(i) + ".f32");
      auto out_info = result[i].GetTensorTypeAndShapeInfo();
      if (out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          out_info.GetElementCount() != expected.size()) throw std::runtime_error("Output tensor mismatch");
      const float* actual = result[i].GetTensorData<float>();
      float max_error = 0;
      size_t failures = 0;
      for (size_t j = 0; j < expected.size(); ++j) {
        float error = std::abs(actual[j] - expected[j]);
        if (!std::isfinite(actual[j]) || !std::isfinite(expected[j]) ||
            error > 0.002f + 0.002f * std::abs(expected[j])) ++failures;
        max_error = std::max(max_error, error);
      }
      pass &= failures == 0;
      std::cout << "OUTPUT name=" << outputs[i] << " elements=" << expected.size()
                << " max_abs_error=" << max_error << " mismatches=" << failures << '\n';
      if (expected.size() <= 10) {
        std::cout << "VALUES";
        for (size_t j = 0; j < expected.size(); ++j) std::cout << ' ' << actual[j];
        std::cout << '\n';
      }
    }
    std::cout << "ORT result=" << (pass ? "PASS" : "FAIL") << " inference_us=" << us << std::endl;
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "ORT ERROR: " << error.what() << std::endl;
    return 1;
  }
}
