#include "parakeet_trt.h"
#include "tokenizer.h"
#include "decoder.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Parakeet TDT 0.6B v3 runtime constants (see tools/export_onnx/out/model_meta.json).
constexpr int32_t kBlankId = 8192;
constexpr int32_t kJointVocabSize = 8198;     // token_logits (8193) + duration_logits (5)
constexpr int32_t kTokenVocabSize = 8193;     // includes blank_id
constexpr int32_t kNumDurations = 5;          // duration_values = [0,1,2,3,4]
constexpr int32_t kEncDim = 1024;
constexpr int32_t kPredDim = 640;
constexpr int32_t kPredLayers = 2;
constexpr int32_t kNMels = 128;
constexpr int32_t kTrtMinT = 16;              // current engine profiles are built with min T=16
constexpr int32_t kTrtChunkT = 16;            // decode using T=16 joint slices (first timestep)

static int32_t kDurationValues[kNumDurations] = {0, 1, 2, 3, 4};

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

static Logger gLogger;

struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const noexcept {
#if NV_TENSORRT_MAJOR >= 10
    delete p;
#else
    if (p) p->destroy();
#endif
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
};

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

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

static size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      throw std::runtime_error("Unsupported dtype for sizing");
  }
}

static size_t volume(const nvinfer1::Dims& d) {
  if (d.nbDims <= 0) return 0;
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) throw std::runtime_error("Dynamic dim still present in volume()");
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}

// Very small FP32->FP16 conversion (sufficient for inference inputs).
static uint16_t f32_to_fp16(float x) {
  // IEEE754 float -> half, round-to-nearest-even (approx; adequate for inputs).
  uint32_t u;
  std::memcpy(&u, &x, sizeof(uint32_t));
  const uint32_t sign = (u >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127;
  uint32_t mant = u & 0x7FFFFF;
  if (exp <= -15) {
    // Subnormal/zero
    return static_cast<uint16_t>(sign << 15);
  }
  if (exp >= 16) {
    // Inf
    return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }
  const uint16_t hexp = static_cast<uint16_t>(exp + 15);
  // Mantissa: take top 10 bits (no perfect rounding needed for this demo).
  const uint16_t hmant = static_cast<uint16_t>(mant >> 13);
  return static_cast<uint16_t>((sign << 15) | (hexp << 10) | hmant);
}

static float fp16_to_f32(uint16_t u) {
  const uint32_t sign = (u & 0x8000u) << 16;
  const uint32_t exp = (u & 0x7C00u) >> 10;
  const uint32_t mant = (u & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    f = sign;  // treat subnormals as 0
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);  // inf/nan
  } else {
    const uint32_t e32 = (exp - 15 + 127) & 0xFFu;
    f = sign | (e32 << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &f, sizeof(float));
  return out;
}

struct TrtEngine {
  std::string name;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> ctx;
  std::vector<DeviceBuffer> bufs;
  std::map<std::string, void*> tensors;
};

static void set_input_shape_or_throw(TrtEngine& e, const char* name, const std::vector<int32_t>& dims) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int>(dims.size());
  for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[static_cast<size_t>(i)];
#if NV_TENSORRT_MAJOR >= 10
  if (!e.ctx->setInputShape(name, d)) throw std::runtime_error(std::string("setInputShape failed for ") + name);
#else
  const int idx = e.engine->getBindingIndex(name);
  if (idx < 0) throw std::runtime_error(std::string("Missing binding: ") + name);
  if (!e.ctx->setBindingDimensions(idx, d)) throw std::runtime_error(std::string("setBindingDimensions failed for ") + name);
#endif
}

static void allocate_buffers_for_current_shapes(TrtEngine& e, cudaStream_t stream) {
#if NV_TENSORRT_MAJOR >= 10
  const int nb = e.engine->getNbIOTensors();
  e.bufs.resize(static_cast<size_t>(nb));

  for (int i = 0; i < nb; ++i) {
    const char* tn = e.engine->getIOTensorName(i);
    const auto dims = e.ctx->getTensorShape(tn);
    const auto dt = e.engine->getTensorDataType(tn);
    const size_t bytes = volume(dims) * dtype_size(dt);

    auto& b = e.bufs[static_cast<size_t>(i)];
    if (b.ptr) {
      if (b.bytes >= bytes) {
        e.tensors[tn] = b.ptr;
        if (!e.ctx->setTensorAddress(tn, b.ptr)) throw std::runtime_error(std::string("setTensorAddress failed for ") + tn);
        continue;
      }
      cuda_check(cudaFree(b.ptr), "cudaFree");
      b.ptr = nullptr;
      b.bytes = 0;
    }
    cuda_check(cudaMalloc(&b.ptr, bytes), "cudaMalloc");
    b.bytes = bytes;
    e.tensors[tn] = b.ptr;
    if (!e.ctx->setTensorAddress(tn, b.ptr)) throw std::runtime_error(std::string("setTensorAddress failed for ") + tn);
    cuda_check(cudaMemsetAsync(b.ptr, 0, bytes, stream), "cudaMemsetAsync");
  }
#else
  (void)stream;
  throw std::runtime_error("TensorRT < 10 not supported by this demo runtime");
#endif
}

static void enqueue_or_throw(TrtEngine& e, cudaStream_t stream) {
#if NV_TENSORRT_MAJOR >= 10
  if (!e.ctx->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed for engine: " + e.name);
#else
  (void)stream;
  throw std::runtime_error("TensorRT < 10 not supported by this demo runtime");
#endif
}

static std::string find_vocab_path(const std::string& model_dir) {
  namespace fs = std::filesystem;
  fs::path md(model_dir);
  fs::path direct = md / "vocab.txt";
  if (fs::exists(direct)) return direct.string();

  // Common layout in this repo: engines under `models/...`, vocab under `tools/export_onnx/out/`.
  // Try: <repo_root>/tools/export_onnx/out/vocab.txt
  fs::path repo_root = md.parent_path().parent_path();
  fs::path fallback = repo_root / "tools" / "export_onnx" / "out" / "vocab.txt";
  if (fs::exists(fallback)) return fallback.string();

  throw std::runtime_error("Could not locate vocab.txt (tried " + direct.string() + " and " + fallback.string() + ")");
}

}  // namespace

struct ParakeetSession {
  std::string model_dir;
  int32_t device_id = 0;
  bool use_fp16 = true;

  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtEngine enc;
  TrtEngine pred;
  TrtEngine joint;
  cudaStream_t stream{};

  std::shared_ptr<Tokenizer> tokenizer;
  std::unique_ptr<Decoder> decoder;

  // Predictor state buffers (device pointers).
  void* d_h = nullptr;
  void* d_c = nullptr;
  void* d_h_out = nullptr;
  void* d_c_out = nullptr;
  void* d_y = nullptr;
  void* d_g = nullptr;

  // Joint buffers.
  void* d_joint_out = nullptr;
  void* d_joint_enc_in = nullptr;
  void* d_joint_pred_in = nullptr;

  // Encoder buffers.
  void* d_audio = nullptr;
  void* d_length = nullptr;
  void* d_enc_out = nullptr;
  void* d_enc_len = nullptr;

  // Event plumbing.
  std::queue<ParakeetEvent> event_queue;
  std::mutex event_mutex;
  std::string current_text;
  std::string current_error;

  ParakeetSession(const ParakeetConfig* config)
      : model_dir(config->model_dir), device_id(config->device_id), use_fp16(config->use_fp16) {}
};

ParakeetSession* parakeet_create_session(const ParakeetConfig* config) {
  if (!config || !config->model_dir) return nullptr;
  try {
    auto session = std::make_unique<ParakeetSession>(config);

    cuda_check(cudaSetDevice(session->device_id), "cudaSetDevice");
    cuda_check(cudaStreamCreate(&session->stream), "cudaStreamCreate");

    session->runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!session->runtime) throw std::runtime_error("Failed to create TensorRT runtime");

    auto load = [&](const std::string& name) -> TrtEngine {
      TrtEngine e;
      e.name = name;
      const std::string path = session->model_dir + "/" + name + ".engine";
      auto data = read_file(path);
      nvinfer1::ICudaEngine* raw_engine = session->runtime->deserializeCudaEngine(data.data(), data.size());
      if (!raw_engine) throw std::runtime_error("Failed to deserialize engine: " + path);
      nvinfer1::IExecutionContext* raw_ctx = raw_engine->createExecutionContext();
      if (!raw_ctx) throw std::runtime_error("Failed to create execution context: " + name);
      e.engine = TrtUniquePtr<nvinfer1::ICudaEngine>(raw_engine);
      e.ctx = TrtUniquePtr<nvinfer1::IExecutionContext>(raw_ctx);
      return e;
    };

    session->enc = load("encoder");
    session->pred = load("predictor");
    session->joint = load("joint");

    // Tokenizer + decoder.
    session->tokenizer = std::make_shared<Tokenizer>(find_vocab_path(session->model_dir));
    // Token vocab includes blank_id; duration head handled separately.
    session->decoder = std::make_unique<Decoder>(session->tokenizer, kBlankId, kTokenVocabSize);

    // Configure predictor fixed shapes (U=1).
    set_input_shape_or_throw(session->pred, "y", {1, 1});
    set_input_shape_or_throw(session->pred, "h", {kPredLayers, 1, kPredDim});
    set_input_shape_or_throw(session->pred, "c", {kPredLayers, 1, kPredDim});
    allocate_buffers_for_current_shapes(session->pred, session->stream);
    session->d_y = session->pred.tensors["y"];
    session->d_h = session->pred.tensors["h"];
    session->d_c = session->pred.tensors["c"];
    session->d_g = session->pred.tensors["g"];
    session->d_h_out = session->pred.tensors["h_out"];
    session->d_c_out = session->pred.tensors["c_out"];

    // Initialize predictor state to zeros.
    cuda_check(cudaMemsetAsync(session->d_h, 0, kPredLayers * 1 * kPredDim * 2, session->stream), "cudaMemsetAsync(h)");
    cuda_check(cudaMemsetAsync(session->d_c, 0, kPredLayers * 1 * kPredDim * 2, session->stream), "cudaMemsetAsync(c)");

    // Configure joint shapes for slice mode: encoder_output [1,1024,16], predictor_output [1,640,1].
    set_input_shape_or_throw(session->joint, "encoder_output", {1, kEncDim, kTrtChunkT});
    set_input_shape_or_throw(session->joint, "predictor_output", {1, kPredDim, 1});
    allocate_buffers_for_current_shapes(session->joint, session->stream);
    session->d_joint_enc_in = session->joint.tensors["encoder_output"];
    session->d_joint_pred_in = session->joint.tensors["predictor_output"];
    session->d_joint_out = session->joint.tensors["joint_output"];

    return session.release();
  } catch (const std::exception& e) {
    std::cerr << "Failed to create session: " << e.what() << std::endl;
    return nullptr;
  }
}

void parakeet_destroy_session(ParakeetSession* session) {
  if (!session) return;
  if (session->stream) {
    cudaStreamDestroy(session->stream);
    session->stream = nullptr;
  }
  delete session;
}

void parakeet_reset_utterance(ParakeetSession* session) {
  if (!session) return;
  session->decoder->reset();
  // Reset predictor state to zeros.
  cuda_check(cudaMemsetAsync(session->d_h, 0, kPredLayers * 1 * kPredDim * 2, session->stream), "cudaMemsetAsync(h)");
  cuda_check(cudaMemsetAsync(session->d_c, 0, kPredLayers * 1 * kPredDim * 2, session->stream), "cudaMemsetAsync(c)");
  std::lock_guard<std::mutex> lock(session->event_mutex);
  while (!session->event_queue.empty()) session->event_queue.pop();
}

int parakeet_push_features(ParakeetSession* session, const float* features_bct_f32, size_t num_frames) {
  if (!session || !features_bct_f32) return -1;
  if (num_frames == 0) return 0;
  try {
    // Encoder engines are currently profiled for 16..256 frames.
    if (num_frames > 256) {
      throw std::runtime_error("num_frames exceeds encoder profile max (256) for this demo runtime");
    }

    const int32_t T_valid = static_cast<int32_t>(num_frames);
    const int32_t T_shape = std::max<int32_t>(kTrtMinT, T_valid);

    // Configure encoder shapes for this chunk.
    set_input_shape_or_throw(session->enc, "audio_signal", {1, kNMels, T_shape});
    set_input_shape_or_throw(session->enc, "length", {1});
    allocate_buffers_for_current_shapes(session->enc, session->stream);
    session->d_audio = session->enc.tensors["audio_signal"];
    session->d_length = session->enc.tensors["length"];
    session->d_enc_out = session->enc.tensors["encoder_output"];
    session->d_enc_len = session->enc.tensors["encoded_lengths"];

    // Host -> device: length
    const int64_t host_len = static_cast<int64_t>(T_valid);
    cuda_check(cudaMemcpyAsync(session->d_length, &host_len, sizeof(host_len), cudaMemcpyHostToDevice, session->stream),
               "cudaMemcpyAsync(length)");

    // Host -> device: audio features (FP16 expected).
    std::vector<uint16_t> host_audio_fp16(static_cast<size_t>(kNMels) * static_cast<size_t>(T_shape), 0);
    for (int32_t m = 0; m < kNMels; ++m) {
      for (int32_t t = 0; t < T_valid; ++t) {
        const float v = features_bct_f32[static_cast<size_t>(m) * static_cast<size_t>(T_valid) + static_cast<size_t>(t)];
        host_audio_fp16[static_cast<size_t>(m) * static_cast<size_t>(T_shape) + static_cast<size_t>(t)] = f32_to_fp16(v);
      }
    }
    cuda_check(cudaMemcpyAsync(session->d_audio, host_audio_fp16.data(),
                               host_audio_fp16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, session->stream),
               "cudaMemcpyAsync(audio_signal)");

    // Run encoder.
    enqueue_or_throw(session->enc, session->stream);
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(encoder)");

    // Read encoded length.
    int64_t enc_len_host = 0;
    cuda_check(cudaMemcpyAsync(&enc_len_host, session->d_enc_len, sizeof(enc_len_host), cudaMemcpyDeviceToHost, session->stream),
               "cudaMemcpyAsync(encoded_lengths)");
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(encoded_lengths)");
    const int32_t T_enc = static_cast<int32_t>(enc_len_host);
    if (T_enc <= 0 || T_enc > T_shape) throw std::runtime_error("Invalid encoded_lengths from encoder");

    // Copy encoder output to host once for decoding (FP16, layout [1,1024,T_enc] BCT).
    std::vector<uint16_t> host_enc_out_fp16(static_cast<size_t>(kEncDim) * static_cast<size_t>(T_enc));
    cuda_check(cudaMemcpyAsync(host_enc_out_fp16.data(), session->d_enc_out,
                               host_enc_out_fp16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, session->stream),
               "cudaMemcpyAsync(encoder_output)");
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(encoder_output)");

    // Greedy TDT decode loop using joint slices (T=16) due to engine min profile.
    int32_t y_id = kBlankId;
    std::vector<int> emitted;
    emitted.reserve(static_cast<size_t>(T_enc));

    std::vector<uint16_t> host_joint_enc_slice_fp16(static_cast<size_t>(kEncDim) * static_cast<size_t>(kTrtChunkT));
    std::vector<uint16_t> host_joint_logits_fp16(static_cast<size_t>(kJointVocabSize));
    std::vector<float> host_joint_logits_f32(static_cast<size_t>(kJointVocabSize));

    int time_idx = 0;
    while (time_idx < T_enc) {
      // Predictor: y (INT64)
      const int64_t host_y = static_cast<int64_t>(y_id);
      cuda_check(cudaMemcpyAsync(session->d_y, &host_y, sizeof(host_y), cudaMemcpyHostToDevice, session->stream),
                 "cudaMemcpyAsync(y)");
      enqueue_or_throw(session->pred, session->stream);
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(predictor)");

      // Copy h_out/c_out back into h/c for next step.
      const size_t hc_bytes = static_cast<size_t>(kPredLayers) * static_cast<size_t>(1) * static_cast<size_t>(kPredDim) * sizeof(uint16_t);
      cuda_check(cudaMemcpyAsync(session->d_h, session->d_h_out, hc_bytes, cudaMemcpyDeviceToDevice, session->stream),
                 "cudaMemcpyAsync(h_out->h)");
      cuda_check(cudaMemcpyAsync(session->d_c, session->d_c_out, hc_bytes, cudaMemcpyDeviceToDevice, session->stream),
                 "cudaMemcpyAsync(c_out->c)");

      // Prepare joint encoder slice: replicate encoder frame across T=16.
      for (int32_t c = 0; c < kEncDim; ++c) {
        const uint16_t v = host_enc_out_fp16[static_cast<size_t>(c) * static_cast<size_t>(T_enc) + static_cast<size_t>(time_idx)];
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(kTrtChunkT);
        for (int32_t t = 0; t < kTrtChunkT; ++t) {
          host_joint_enc_slice_fp16[base + static_cast<size_t>(t)] = v;
        }
      }
      cuda_check(cudaMemcpyAsync(session->d_joint_enc_in, host_joint_enc_slice_fp16.data(),
                                 host_joint_enc_slice_fp16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, session->stream),
                 "cudaMemcpyAsync(joint.encoder_output)");

      // Wire predictor_output from predictor engine directly into joint by copying g -> joint predictor input.
      const size_t g_bytes = static_cast<size_t>(kPredDim) * sizeof(uint16_t);
      cuda_check(cudaMemcpyAsync(session->d_joint_pred_in, session->d_g, g_bytes, cudaMemcpyDeviceToDevice, session->stream),
                 "cudaMemcpyAsync(g->joint.predictor_output)");

      // Run joint.
      enqueue_or_throw(session->joint, session->stream);
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(joint)");

      // Copy logits for first timestep (t=0,u=0) to host.
      cuda_check(cudaMemcpyAsync(host_joint_logits_fp16.data(), session->d_joint_out,
                                 host_joint_logits_fp16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, session->stream),
                 "cudaMemcpyAsync(joint_output)");
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(joint_output)");

      for (int32_t i = 0; i < kJointVocabSize; ++i) {
        host_joint_logits_f32[static_cast<size_t>(i)] = fp16_to_f32(host_joint_logits_fp16[static_cast<size_t>(i)]);
      }

      // Token argmax over [0..kTokenVocabSize)
      int best_tok = 0;
      float best_tok_v = host_joint_logits_f32[0];
      for (int32_t i = 1; i < kTokenVocabSize; ++i) {
        const float v = host_joint_logits_f32[static_cast<size_t>(i)];
        if (v > best_tok_v) {
          best_tok_v = v;
          best_tok = i;
        }
      }

      // Duration argmax over tail [kTokenVocabSize..kJointVocabSize)
      int best_dur_idx = 0;
      float best_dur_v = host_joint_logits_f32[static_cast<size_t>(kTokenVocabSize)];
      for (int32_t i = 1; i < kNumDurations; ++i) {
        const float v = host_joint_logits_f32[static_cast<size_t>(kTokenVocabSize + i)];
        if (v > best_dur_v) {
          best_dur_v = v;
          best_dur_idx = i;
        }
      }
      const int duration = kDurationValues[best_dur_idx];

      if (best_tok != kBlankId) {
        emitted.push_back(best_tok);
        y_id = best_tok;
      }

      if (duration <= 0) {
        time_idx += 1;
      } else {
        time_idx += duration;
      }
    }

    // Emit final transcript event.
    {
      std::lock_guard<std::mutex> lock(session->event_mutex);
      ParakeetEvent ev{};
      ev.type = PARAKEET_EVENT_FINAL_TEXT;
      ev.segment_id = 0;
      session->current_text = session->tokenizer->decode(emitted);
      ev.text = session->current_text.c_str();
      session->event_queue.push(ev);
    }

    return 0;
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(session->event_mutex);
    ParakeetEvent ev{};
    ev.type = PARAKEET_EVENT_ERROR;
    session->current_error = e.what();
    ev.error_message = session->current_error.c_str();
    session->event_queue.push(ev);
    return -2;
  }
}

bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event) {
  if (!session || !event) return false;
  std::lock_guard<std::mutex> lock(session->event_mutex);
  if (session->event_queue.empty()) return false;
  *event = session->event_queue.front();
  session->event_queue.pop();
  return true;
}
