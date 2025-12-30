#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef PARAKEET_MOCK
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::cout << "[trt_runtime_smoke] MOCK build: skipping TensorRT execution.\n";
  return 0;
}
#else

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << "\n";
    }
  }
};

static Logger gLogger;

struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const noexcept {
    if (p) p->destroy();
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open file: " + path);
  f.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  f.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}

static const char* dtype_str(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return "FP32";
    case nvinfer1::DataType::kHALF:
      return "FP16";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
#endif
    default:
      return "UNKNOWN";
  }
}

static size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kUINT8:
      return 1;
#endif
    default:
      throw std::runtime_error("Unsupported dtype for sizing");
  }
}

static std::string dims_to_string(const nvinfer1::Dims& d) {
  std::string s = "[";
  for (int i = 0; i < d.nbDims; ++i) {
    s += std::to_string(d.d[i]);
    if (i + 1 < d.nbDims) s += ",";
  }
  s += "]";
  return s;
}

static size_t volume(const nvinfer1::Dims& d) {
  if (d.nbDims <= 0) return 0;
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) throw std::runtime_error("Dynamic dim still present in " + dims_to_string(d));
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}

struct Engine {
  std::string name;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> ctx;
};

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
};

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

static void print_engine_bindings(const Engine& e) {
  const int nb = e.engine->getNbBindings();
  std::cout << "=== " << e.name << " bindings ===\n";
  for (int i = 0; i < nb; ++i) {
    const char* bn = e.engine->getBindingName(i);
    const bool is_input = e.engine->bindingIsInput(i);
    auto dt = e.engine->getBindingDataType(i);
    std::cout << "  [" << i << "] " << (is_input ? "IN " : "OUT") << " " << bn << " dtype=" << dtype_str(dt)
              << " dims=" << dims_to_string(e.ctx->getBindingDimensions(i)) << "\n";
  }
}

static void set_shape_or_throw(Engine& e, const char* binding_name, const std::vector<int32_t>& dims) {
  const int idx = e.engine->getBindingIndex(binding_name);
  if (idx < 0) throw std::runtime_error(std::string("Missing binding: ") + binding_name);
  if (!e.engine->bindingIsInput(idx)) throw std::runtime_error(std::string("Binding not input: ") + binding_name);
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int>(dims.size());
  for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[static_cast<size_t>(i)];
  if (!e.ctx->setBindingDimensions(idx, d)) {
    throw std::runtime_error(std::string("setBindingDimensions failed for ") + binding_name);
  }
}

static void allocate_buffers_for_current_shapes(
    Engine& e, std::vector<DeviceBuffer>& bufs, std::vector<void*>& bindings, cudaStream_t stream) {
  const int nb = e.engine->getNbBindings();
  bufs.resize(static_cast<size_t>(nb));
  bindings.resize(static_cast<size_t>(nb), nullptr);

  for (int i = 0; i < nb; ++i) {
    const auto dims = e.ctx->getBindingDimensions(i);
    const auto dt = e.engine->getBindingDataType(i);
    const size_t bytes = volume(dims) * dtype_size(dt);

    auto& b = bufs[static_cast<size_t>(i)];
    if (b.ptr) {
      // Reuse only if large enough.
      if (b.bytes >= bytes) {
        bindings[static_cast<size_t>(i)] = b.ptr;
        continue;
      }
      cuda_check(cudaFree(b.ptr), "cudaFree");
      b.ptr = nullptr;
      b.bytes = 0;
    }

    cuda_check(cudaMalloc(&b.ptr, bytes), "cudaMalloc");
    b.bytes = bytes;
    bindings[static_cast<size_t>(i)] = b.ptr;

    // Deterministic init for smoke: zeros for all buffers.
    cuda_check(cudaMemsetAsync(b.ptr, 0, bytes, stream), "cudaMemsetAsync");
  }
}

static void run_once(Engine& e, cudaStream_t stream, std::vector<void*>& bindings) {
  auto t0 = std::chrono::high_resolution_clock::now();
  if (!e.ctx->enqueueV2(bindings.data(), stream, nullptr)) {
    throw std::runtime_error("enqueueV2 failed for engine: " + e.name);
  }
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
  std::cout << e.name << ": ok (" << ms << " ms)\n";
}

static void check_output_non_nan(Engine& e, const char* out_name, int max_values_to_check, cudaStream_t stream,
                                 const std::vector<void*>& bindings) {
  const int idx = e.engine->getBindingIndex(out_name);
  if (idx < 0) throw std::runtime_error(std::string("Missing output binding: ") + out_name);
  if (e.engine->bindingIsInput(idx)) throw std::runtime_error(std::string("Binding is input, expected output: ") + out_name);

  const auto dims = e.ctx->getBindingDimensions(idx);
  const auto dt = e.engine->getBindingDataType(idx);
  const size_t n = volume(dims);
  const size_t k = std::min<size_t>(n, static_cast<size_t>(max_values_to_check));

  if (dt != nvinfer1::DataType::kFLOAT && dt != nvinfer1::DataType::kHALF) {
    std::cout << "  " << out_name << ": dtype=" << dtype_str(dt) << " (skipping NaN check)\n";
    return;
  }

  std::vector<float> host(k, 0.0f);
  if (dt == nvinfer1::DataType::kFLOAT) {
    cuda_check(cudaMemcpyAsync(host.data(), bindings[static_cast<size_t>(idx)], k * sizeof(float), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  } else {
    // FP16 -> convert via CPU half decode (very small check). This is a smoke check, not perf-critical.
    std::vector<uint16_t> h16(k, 0);
    cuda_check(cudaMemcpyAsync(h16.data(), bindings[static_cast<size_t>(idx)], k * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    for (size_t i = 0; i < k; ++i) {
      const uint16_t u = h16[i];
      // FP16 to FP32 (minimal, handles normals/zeros; NaN/Inf still detected as NaN/Inf).
      const uint32_t sign = (u & 0x8000u) << 16;
      const uint32_t exp = (u & 0x7C00u) >> 10;
      const uint32_t mant = (u & 0x03FFu);
      uint32_t f;
      if (exp == 0) {
        f = sign;  // treat subnorms as 0 for smoke
      } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);  // inf/nan
      } else {
        const uint32_t e32 = (exp - 15 + 127) & 0xFFu;
        f = sign | (e32 << 23) | (mant << 13);
      }
      float out;
      std::memcpy(&out, &f, sizeof(float));
      host[i] = out;
    }
  }

  int bad = 0;
  for (size_t i = 0; i < k; ++i) {
    const float v = host[i];
    if (!(v == v)) bad++;  // NaN check
  }
  if (bad) {
    throw std::runtime_error(std::string("NaN detected in output: ") + out_name);
  }
  std::cout << "  " << out_name << ": checked " << k << " values (non-NaN)\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = "models/parakeet-tdt-0.6b-v3";
  int device = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model_dir" && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (a == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: trt_runtime_smoke [--model_dir DIR] [--device N]\n";
      return 0;
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      return 2;
    }
  }

  cuda_check(cudaSetDevice(device), "cudaSetDevice");
  cudaStream_t stream{};
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
  if (!runtime) {
    std::cerr << "Failed to create TensorRT runtime\n";
    return 1;
  }

  auto load = [&](const std::string& name) -> Engine {
    const std::string path = model_dir + "/" + name + ".engine";
    auto data = read_file(path);
    nvinfer1::ICudaEngine* raw_engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!raw_engine) throw std::runtime_error("Failed to deserialize engine: " + path);
    nvinfer1::IExecutionContext* raw_ctx = raw_engine->createExecutionContext();
    if (!raw_ctx) throw std::runtime_error("Failed to create execution context: " + name);
    Engine e;
    e.name = name;
    e.engine = TrtUniquePtr<nvinfer1::ICudaEngine>(raw_engine);
    e.ctx = TrtUniquePtr<nvinfer1::IExecutionContext>(raw_ctx);
    return e;
  };

  try {
    Engine enc = load("encoder");
    Engine pred = load("predictor");
    Engine joint = load("joint");

    // Set opt shapes (Phase 2 contract defaults) before allocating buffers.
    set_shape_or_throw(enc, "audio_signal", {1, 128, 64});
    set_shape_or_throw(enc, "length", {1});

    set_shape_or_throw(pred, "y", {1, 1});
    set_shape_or_throw(pred, "h", {2, 1, 640});
    set_shape_or_throw(pred, "c", {2, 1, 640});

    set_shape_or_throw(joint, "encoder_output", {1, 1024, 64});
    set_shape_or_throw(joint, "predictor_output", {1, 640, 1});

    // Allocate once, reuse.
    std::vector<DeviceBuffer> enc_bufs, pred_bufs, joint_bufs;
    std::vector<void*> enc_bindings, pred_bindings, joint_bindings;

    allocate_buffers_for_current_shapes(enc, enc_bufs, enc_bindings, stream);
    allocate_buffers_for_current_shapes(pred, pred_bufs, pred_bindings, stream);
    allocate_buffers_for_current_shapes(joint, joint_bufs, joint_bindings, stream);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    // Fill required INT64 inputs deterministically.
    {
      int64_t host_len = 64;
      const int idx = enc.engine->getBindingIndex("length");
      cuda_check(cudaMemcpyAsync(enc_bindings[static_cast<size_t>(idx)], &host_len, sizeof(host_len), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(length)");

      int64_t host_y = 0;  // blank/0
      const int y_idx = pred.engine->getBindingIndex("y");
      cuda_check(cudaMemcpyAsync(pred_bindings[static_cast<size_t>(y_idx)], &host_y, sizeof(host_y), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(y)");
    }

    std::cout << "Loaded engines from: " << model_dir << "\n";
    print_engine_bindings(enc);
    print_engine_bindings(pred);
    print_engine_bindings(joint);

    std::cout << "=== smoke run (opt shapes) ===\n";
    run_once(enc, stream, enc_bindings);
    run_once(pred, stream, pred_bindings);
    run_once(joint, stream, joint_bindings);

    // Minimal sanity checks: output shapes resolved + non-NaN (small sample).
    check_output_non_nan(enc, "encoder_output", 32, stream, enc_bindings);
    check_output_non_nan(pred, "g", 32, stream, pred_bindings);
    check_output_non_nan(joint, "joint_output", 32, stream, joint_bindings);

    // Cleanup
    for (auto& b : enc_bufs) cudaFree(b.ptr);
    for (auto& b : pred_bufs) cudaFree(b.ptr);
    for (auto& b : joint_bufs) cudaFree(b.ptr);
    cudaStreamDestroy(stream);

    std::cout << "OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    cudaStreamDestroy(stream);
    return 1;
  }
}

#endif  // PARAKEET_MOCK


