#include "parakeet_trt.h"
#include "tokenizer.h"
#include "decoder.h"

#include <cmath>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Parakeet TDT 0.6B v3 runtime constants (see tools/export_onnx/out/model_meta.json).
constexpr int32_t kBlankId = 8192;
constexpr int32_t kJointVocabSize = 8198;     // token_logits (8193) + duration_logits (5)
constexpr int32_t kTokenVocabSize = 8193;     // includes blank_id
constexpr int32_t kNumDurations = 5;          // duration_values = [0,1,2,3,4]
constexpr int32_t kEncLayers = 24;
constexpr int32_t kEncDim = 1024;
constexpr int32_t kPredDim = 640;
constexpr int32_t kPredLayers = 2;
constexpr int32_t kNMels = 128;
constexpr int32_t kTrtMinT = 16;              // current engine profiles are built with min T=16
constexpr int32_t kTrtChunkT = 16;            // decode using T=16 joint slices (first timestep)
constexpr int32_t kCacheSize = 256;
constexpr int32_t kCacheTime = 4;

static int32_t kDurationValues[kNumDurations] = {0, 1, 2, 3, 4};

float get_blank_penalty() {
  const char* v = std::getenv("PARAKEET_BLANK_PENALTY");
  if (!v || !*v) {
    return 0.0f;
  }
  try {
    return std::stof(v);
  } catch (...) {
    return 0.0f;
  }
}

bool get_debug_topk() {
  const char* v = std::getenv("PARAKEET_DEBUG_TOPK");
  if (!v || !*v) {
    return false;
  }
  return std::string(v) == "1";
}

bool get_env_bool(const char* name, bool fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  const std::string s(v);
  if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
  if (s == "0" || s == "false" || s == "no" || s == "off") return false;
  return fallback;
}

uint64_t get_env_u64(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  try {
    return static_cast<uint64_t>(std::stoull(v));
  } catch (...) {
    return fallback;
  }
}

uint64_t get_slow_enqueue_ms() {
  const uint64_t from_env = get_env_u64("PARAKEET_SLOW_ENQUEUE_MS", 0);
  if (from_env > 0) return from_env;
  return get_env_u64("PARAKEET_SLOW_CHUNK_MS", 250);
}

void log_slow_enqueue_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const uint64_t slow_ms = get_slow_enqueue_ms();
    std::cerr << "[parakeet_trt] slow_enqueue_ms=" << slow_ms
              << " (env PARAKEET_SLOW_ENQUEUE_MS or PARAKEET_SLOW_CHUNK_MS)\n";
  });
}

enum class SlotReuseMode {
  kLog,
  kWait,
  kFail,
};

struct SlotReuseConfig {
  bool enabled = false;
  SlotReuseMode mode = SlotReuseMode::kLog;
  uint64_t log_limit = 4;
  uint64_t slot_cap_override = 0;
};

SlotReuseMode parse_slot_reuse_mode(const char* v, SlotReuseMode fallback) {
  if (!v || !*v) return fallback;
  const std::string s(v);
  if (s == "log") return SlotReuseMode::kLog;
  if (s == "wait") return SlotReuseMode::kWait;
  if (s == "fail") return SlotReuseMode::kFail;
  return fallback;
}

const char* slot_reuse_mode_name(SlotReuseMode mode) {
  switch (mode) {
    case SlotReuseMode::kLog:
      return "log";
    case SlotReuseMode::kWait:
      return "wait";
    case SlotReuseMode::kFail:
      return "fail";
    default:
      return "log";
  }
}

SlotReuseConfig get_slot_reuse_config() {
  static SlotReuseConfig cfg = []() {
    SlotReuseConfig out{};
    const char* mode_env = std::getenv("PARAKEET_SLOT_REUSE_MODE");
    if (mode_env && *mode_env) {
      out.enabled = true;
      out.mode = parse_slot_reuse_mode(mode_env, SlotReuseMode::kLog);
    }
    out.enabled = get_env_bool("PARAKEET_SLOT_REUSE_CHECK", out.enabled);
    out.log_limit = get_env_u64("PARAKEET_SLOT_REUSE_LOG_THRESHOLD", 4);
    out.slot_cap_override = get_env_u64("PARAKEET_SLOT_REUSE_CAP", 0);
    return out;
  }();
  return cfg;
}

void log_slot_reuse_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto cfg = get_slot_reuse_config();
    if (!cfg.enabled) return;
    std::cerr << "[parakeet_trt] slot_reuse_check=1 mode=" << slot_reuse_mode_name(cfg.mode)
              << " log_limit=" << cfg.log_limit
              << " slot_cap_override=" << cfg.slot_cap_override << "\n";
  });
}

enum class DebugSyncTarget {
  kAll,
  kEncoder,
  kPredictor,
  kJoint,
};

struct DebugSyncConfig {
  bool enabled = false;
  DebugSyncTarget target = DebugSyncTarget::kAll;
  uint64_t limit = 0;
};

DebugSyncTarget parse_debug_sync_target(const char* v, DebugSyncTarget fallback) {
  if (!v || !*v) return fallback;
  const std::string s(v);
  if (s == "encoder") return DebugSyncTarget::kEncoder;
  if (s == "predictor") return DebugSyncTarget::kPredictor;
  if (s == "joint") return DebugSyncTarget::kJoint;
  if (s == "all") return DebugSyncTarget::kAll;
  return fallback;
}

const char* debug_sync_target_name(DebugSyncTarget t) {
  switch (t) {
    case DebugSyncTarget::kEncoder:
      return "encoder";
    case DebugSyncTarget::kPredictor:
      return "predictor";
    case DebugSyncTarget::kJoint:
      return "joint";
    case DebugSyncTarget::kAll:
    default:
      return "all";
  }
}

DebugSyncConfig get_debug_sync_config() {
  static DebugSyncConfig cfg = []() {
    DebugSyncConfig out{};
    out.enabled = get_env_bool("PARAKEET_DEBUG_SYNC", false);
    out.target = parse_debug_sync_target(std::getenv("PARAKEET_DEBUG_SYNC_ENGINE"), DebugSyncTarget::kAll);
    out.limit = get_env_u64("PARAKEET_DEBUG_SYNC_LIMIT", 0);
    return out;
  }();
  return cfg;
}

void log_debug_sync_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto cfg = get_debug_sync_config();
    if (!cfg.enabled) return;
    std::cerr << "[parakeet_trt] debug_sync=1 target=" << debug_sync_target_name(cfg.target)
              << " limit=" << cfg.limit << "\n";
  });
}

struct DebugContext {
  std::string id;
  uint64_t utt_seq = 0;
  uint64_t audio_chunk_idx = 0;
  uint64_t feature_idx = 0;
};

enum class DebugDeviceSyncPoint {
  kPush,
  kDestroy,
  kBoth,
};

struct DebugDeviceSyncConfig {
  bool enabled = false;
  DebugDeviceSyncPoint point = DebugDeviceSyncPoint::kBoth;
};

DebugDeviceSyncPoint parse_debug_device_sync_point(const char* v, DebugDeviceSyncPoint fallback) {
  if (!v || !*v) return fallback;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (s == "push" || s == "push_features") return DebugDeviceSyncPoint::kPush;
  if (s == "destroy" || s == "teardown") return DebugDeviceSyncPoint::kDestroy;
  if (s == "both" || s == "all") return DebugDeviceSyncPoint::kBoth;
  return fallback;
}

const char* debug_device_sync_point_name(DebugDeviceSyncPoint p) {
  switch (p) {
    case DebugDeviceSyncPoint::kPush:
      return "push";
    case DebugDeviceSyncPoint::kDestroy:
      return "destroy";
    case DebugDeviceSyncPoint::kBoth:
    default:
      return "both";
  }
}

DebugDeviceSyncConfig get_debug_device_sync_config() {
  static DebugDeviceSyncConfig cfg = []() {
    DebugDeviceSyncConfig out{};
    out.enabled = get_env_bool("PARAKEET_DEBUG_DEVICE_SYNC", false);
    out.point = parse_debug_device_sync_point(std::getenv("PARAKEET_DEBUG_DEVICE_SYNC_POINT"),
                                              DebugDeviceSyncPoint::kBoth);
    return out;
  }();
  return cfg;
}

void log_debug_device_sync_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto cfg = get_debug_device_sync_config();
    if (!cfg.enabled) return;
    std::cerr << "[parakeet_trt] debug_device_sync=1 point="
              << debug_device_sync_point_name(cfg.point) << "\n";
  });
}

bool debug_device_sync_point_matches(DebugDeviceSyncPoint cfg_point, DebugDeviceSyncPoint current) {
  if (cfg_point == DebugDeviceSyncPoint::kBoth) return true;
  return cfg_point == current;
}

void debug_device_sync(DebugDeviceSyncPoint point, const char* label, const DebugContext* ctx) {
  const auto cfg = get_debug_device_sync_config();
  if (!cfg.enabled) return;
  if (!debug_device_sync_point_matches(cfg.point, point)) return;
  log_debug_device_sync_config_once();
  const cudaError_t pre_err = cudaPeekAtLastError();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t sync_err = cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t sync_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  const cudaError_t post_err = cudaGetLastError();
  std::cerr << "[parakeet_trt] device_sync point=" << debug_device_sync_point_name(point)
            << " label=" << (label ? label : "")
            << " id=" << (ctx ? ctx->id : "")
            << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
            << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
            << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
            << " sync_ms=" << sync_ms
            << " pre_err=" << cudaGetErrorString(pre_err)
            << " sync_err=" << cudaGetErrorString(sync_err)
            << " post_err=" << cudaGetErrorString(post_err)
            << "\n";
}

// ---------------------------------------------------------------------------
// Stage Markers: Ultra-low-latency unbuffered logging for hang diagnosis.
// Enable with PARAKEET_DEBUG_STAGE_MARKERS=1
// ---------------------------------------------------------------------------
struct StageMarkerConfig {
  bool enabled = false;
};

StageMarkerConfig get_stage_marker_config() {
  static StageMarkerConfig cfg = []() {
    StageMarkerConfig out{};
    out.enabled = get_env_bool("PARAKEET_DEBUG_STAGE_MARKERS", false);
    return out;
  }();
  return cfg;
}

void ensure_stderr_unbuffered() {
  static std::once_flag once;
  std::call_once(once, []() {
    setvbuf(stderr, NULL, _IONBF, 0);
  });
}

void log_stage_marker_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto cfg = get_stage_marker_config();
    if (!cfg.enabled) return;
    ensure_stderr_unbuffered();
    std::cerr << "[parakeet_trt] stage_markers=1 (unbuffered stderr)\n";
  });
}

static std::chrono::steady_clock::time_point g_stage_marker_start;
static std::once_flag g_stage_marker_start_once;

void debug_stage_marker(const char* stage, const DebugContext* ctx, cudaStream_t stream = nullptr, int loop_idx = -1) {
  const auto cfg = get_stage_marker_config();
  if (!cfg.enabled) return;
  log_stage_marker_config_once();
  std::call_once(g_stage_marker_start_once, []() {
    g_stage_marker_start = std::chrono::steady_clock::now();
  });
  const auto now = std::chrono::steady_clock::now();
  const uint64_t ms_since_start = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now - g_stage_marker_start).count());
  std::cerr << "[parakeet_trt] stage=" << (stage ? stage : "")
            << " id=" << (ctx ? ctx->id : "")
            << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
            << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
            << " feature_idx=" << (ctx ? ctx->feature_idx : 0);
  if (loop_idx >= 0) {
    std::cerr << " loop_idx=" << loop_idx;
  }
  if (stream) {
    std::cerr << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec;
  }
  std::cerr << " ms=" << ms_since_start << "\n";
}

struct DebugMemcpyConfig {
  bool enabled = false;
  bool sync = false;
  uint64_t slow_ms = 0;
};

const char* memcpy_kind_name(cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return "H2D";
    case cudaMemcpyDeviceToHost:
      return "D2H";
    case cudaMemcpyDeviceToDevice:
      return "D2D";
    case cudaMemcpyHostToHost:
      return "H2H";
    default:
      return "UNKNOWN";
  }
}

DebugMemcpyConfig get_debug_memcpy_config() {
  static DebugMemcpyConfig cfg = []() {
    DebugMemcpyConfig out{};
    out.sync = get_env_bool("PARAKEET_DEBUG_SYNC_MEMCPY", false);
    out.slow_ms = get_env_u64("PARAKEET_SLOW_MEMCPY_MS", 50);
    out.enabled = out.sync || out.slow_ms > 0;
    return out;
  }();
  return cfg;
}

void log_debug_memcpy_config_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    const auto cfg = get_debug_memcpy_config();
    if (!cfg.enabled) return;
    std::cerr << "[parakeet_trt] debug_memcpy=1 sync=" << (cfg.sync ? 1 : 0)
              << " slow_ms=" << cfg.slow_ms << "\n";
  });
}

static void debug_memcpy_async(void* dst,
                               const void* src,
                               size_t bytes,
                               cudaMemcpyKind kind,
                               cudaStream_t stream,
                               const char* label,
                               const DebugContext* ctx) {
  const auto cfg = get_debug_memcpy_config();
  log_debug_memcpy_config_once();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t copy_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  const cudaError_t peek_err = cudaPeekAtLastError();
  uint64_t sync_ms = 0;
  cudaError_t sync_err = cudaSuccess;
  cudaError_t post_err = cudaSuccess;
  if (cfg.sync) {
    const auto s0 = std::chrono::steady_clock::now();
    sync_err = cudaStreamSynchronize(stream);
    const auto s1 = std::chrono::steady_clock::now();
    sync_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count());
    post_err = cudaPeekAtLastError();
  }

  const bool slow = cfg.slow_ms > 0 && (copy_ms >= cfg.slow_ms || sync_ms >= cfg.slow_ms);
  if (cfg.sync || slow || err != cudaSuccess || peek_err != cudaSuccess ||
      sync_err != cudaSuccess || post_err != cudaSuccess) {
    std::cerr << "[parakeet_trt] memcpy_async label=" << (label ? label : "")
              << " id=" << (ctx ? ctx->id : "")
              << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
              << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
              << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
              << " bytes=" << bytes
              << " kind=" << memcpy_kind_name(kind)
              << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
              << " src=0x" << std::hex << reinterpret_cast<uintptr_t>(src) << std::dec
              << " dst=0x" << std::hex << reinterpret_cast<uintptr_t>(dst) << std::dec
              << " copy_ms=" << copy_ms
              << " err=" << cudaGetErrorString(err)
              << " peek_err=" << cudaGetErrorString(peek_err)
              << " sync_ms=" << sync_ms
              << " sync_err=" << cudaGetErrorString(sync_err)
              << " post_err=" << cudaGetErrorString(post_err)
              << "\n";
  }

  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string(label ? label : "cudaMemcpyAsync") + ": " +
                             cudaGetErrorString(err));
  }
}

static void debug_memset_async(void* ptr,
                               int value,
                               size_t bytes,
                               cudaStream_t stream,
                               const char* label,
                               const DebugContext* ctx) {
  const auto cfg = get_debug_memcpy_config();
  log_debug_memcpy_config_once();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t err = cudaMemsetAsync(ptr, value, bytes, stream);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t set_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  const cudaError_t peek_err = cudaPeekAtLastError();
  uint64_t sync_ms = 0;
  cudaError_t sync_err = cudaSuccess;
  cudaError_t post_err = cudaSuccess;
  if (cfg.sync) {
    const auto s0 = std::chrono::steady_clock::now();
    sync_err = cudaStreamSynchronize(stream);
    const auto s1 = std::chrono::steady_clock::now();
    sync_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count());
    post_err = cudaPeekAtLastError();
  }

  const bool slow = cfg.slow_ms > 0 && (set_ms >= cfg.slow_ms || sync_ms >= cfg.slow_ms);
  if (cfg.sync || slow || err != cudaSuccess || peek_err != cudaSuccess ||
      sync_err != cudaSuccess || post_err != cudaSuccess) {
    std::cerr << "[parakeet_trt] memset_async label=" << (label ? label : "")
              << " id=" << (ctx ? ctx->id : "")
              << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
              << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
              << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
              << " bytes=" << bytes
              << " value=" << value
              << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
              << " ptr=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec
              << " set_ms=" << set_ms
              << " err=" << cudaGetErrorString(err)
              << " peek_err=" << cudaGetErrorString(peek_err)
              << " sync_ms=" << sync_ms
              << " sync_err=" << cudaGetErrorString(sync_err)
              << " post_err=" << cudaGetErrorString(post_err)
              << "\n";
  }

  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string(label ? label : "cudaMemsetAsync") + ": " +
                             cudaGetErrorString(err));
  }
}

static void debug_cuda_free(void* ptr, const char* label) {
  if (!ptr) return;
  const auto cfg = get_debug_memcpy_config();
  log_debug_memcpy_config_once();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t err = cudaFree(ptr);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t free_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  cudaError_t sync_err = cudaSuccess;
  uint64_t sync_ms = 0;
  cudaError_t post_err = cudaSuccess;
  if (cfg.sync) {
    const auto s0 = std::chrono::steady_clock::now();
    sync_err = cudaDeviceSynchronize();
    const auto s1 = std::chrono::steady_clock::now();
    sync_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count());
    post_err = cudaPeekAtLastError();
  }
  if (cfg.sync || err != cudaSuccess || sync_err != cudaSuccess) {
    std::cerr << "[parakeet_trt] cudaFree label=" << (label ? label : "")
              << " ptr=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec
              << " free_ms=" << free_ms
              << " err=" << cudaGetErrorString(err)
              << " sync_ms=" << sync_ms
              << " sync_err=" << cudaGetErrorString(sync_err)
              << " post_err=" << cudaGetErrorString(post_err)
              << "\n";
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(label ? label : "cudaFree") + ": " + cudaGetErrorString(err));
  }
}

static void debug_cuda_event_destroy(cudaEvent_t ev, const char* label) {
  if (!ev) return;
  const auto cfg = get_debug_memcpy_config();
  log_debug_memcpy_config_once();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t err = cudaEventDestroy(ev);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t destroy_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  if (cfg.sync || err != cudaSuccess) {
    std::cerr << "[parakeet_trt] cudaEventDestroy label=" << (label ? label : "")
              << " destroy_ms=" << destroy_ms
              << " err=" << cudaGetErrorString(err)
              << "\n";
  }
}

static void debug_cuda_stream_destroy(cudaStream_t stream, const char* label) {
  if (!stream) return;
  const auto cfg = get_debug_memcpy_config();
  log_debug_memcpy_config_once();
  const auto t0 = std::chrono::steady_clock::now();
  const cudaError_t err = cudaStreamDestroy(stream);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t destroy_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  if (cfg.sync || err != cudaSuccess) {
    std::cerr << "[parakeet_trt] cudaStreamDestroy label=" << (label ? label : "")
              << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
              << " destroy_ms=" << destroy_ms
              << " err=" << cudaGetErrorString(err)
              << "\n";
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

static Logger gLogger;

// #region agent log
static std::atomic<int> g_dbg_n{0};
static std::atomic<int> g_empty_text_dump_n{0};
static bool is_control_token_str(const std::string& tok) {
  if (tok.empty()) return false;
  if (tok == "<blank>" || tok == "<pad>" || tok == "<unk>") return true;
  return tok.front() == '<' && tok.back() == '>';
}
static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(c));
          out += oss.str();
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}
static void dbglog_ndjson(const char* hypothesisId,
                          const char* location,
                          const char* message,
                          const std::string& data_json) {
  // NOTE: debug mode log path (do not include secrets).
  const char* kLogPath = "/home/emmy/git/parakeet/.cursor/debug.log";
  // Gate to avoid log spam.
  const int n = g_dbg_n.fetch_add(1, std::memory_order_relaxed);
  if (n > 120) return;

  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

  std::ofstream f(kLogPath, std::ios::out | std::ios::app);
  if (!f) return;
  f << "{\"sessionId\":\"debug-session\",\"runId\":\"cpp\",\"hypothesisId\":\""
    << hypothesisId << "\",\"location\":\"" << location << "\",\"message\":\""
    << message << "\",\"data\":" << data_json << ",\"timestamp\":" << now_ms
    << "}\n";
}
// #endregion

static int find_token_id_in_vocab_txt(const std::string& vocab_path, const std::string& token) {
  std::ifstream f(vocab_path);
  if (!f) return -1;
  std::string line;
  int idx = 0;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == token) return idx;
    idx++;
  }
  return -1;
}

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

static uint16_t f32_to_fp16(float x);
static float fp16_to_f32(uint16_t u);

static float max_abs_device(void* ptr,
                            size_t count,
                            nvinfer1::DataType dt,
                            cudaStream_t stream,
                            const char* label,
                            const DebugContext* ctx) {
  if (!ptr || count == 0) return 0.0f;
  if (dt == nvinfer1::DataType::kHALF) {
    std::vector<uint16_t> host(count);
    debug_memcpy_async(host.data(), ptr, host.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream, label, ctx);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(max_abs_device:f16)");
    float max_abs = 0.0f;
    for (size_t i = 0; i < host.size(); ++i) {
      const float v = std::fabs(fp16_to_f32(host[i]));
      if (v > max_abs) max_abs = v;
    }
    return max_abs;
  }
  if (dt == nvinfer1::DataType::kFLOAT) {
    std::vector<float> host(count);
    debug_memcpy_async(host.data(), ptr, host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream, label, ctx);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(max_abs_device:f32)");
    float max_abs = 0.0f;
    for (size_t i = 0; i < host.size(); ++i) {
      const float v = std::fabs(host[i]);
      if (v > max_abs) max_abs = v;
    }
    return max_abs;
  }
  throw std::runtime_error("max_abs_device unsupported dtype");
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
  std::vector<cudaEvent_t> slot_events;
  std::vector<uint8_t> slot_event_ready;
  uint64_t slot_seq = 0;
  uint64_t slot_log_count = 0;
  uint64_t debug_sync_count = 0;
  uint64_t debug_sync_utt_seq = 0;
  bool debug_sync_utt_seq_set = false;
};

static int io_tensor_index(const TrtEngine& e, const char* name) {
  const int nb = e.engine->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char* tn = e.engine->getIOTensorName(i);
    if (tn && std::strcmp(tn, name) == 0) {
      return i;
    }
  }
  return -1;
}

static const char* resolve_tensor_name(const TrtEngine& e, const char* name) {
  const int nb = e.engine->getNbIOTensors();
  const size_t name_len = std::strlen(name);
  const char* exact_match = nullptr;
  const char* suffix_match = nullptr;
  for (int i = 0; i < nb; ++i) {
    const char* tn = e.engine->getIOTensorName(i);
    if (!tn) continue;
    if (std::strcmp(tn, name) == 0) {
      exact_match = tn;
      break;
    }
  }
  if (exact_match) {
    static std::mutex mu;
    static std::unordered_set<std::string> seen;
    const std::string key = e.name + ":" + name;
    std::lock_guard<std::mutex> lock(mu);
    if (seen.insert(key).second) {
      const int idx = io_tensor_index(e, exact_match);
      std::cerr << "[parakeet_trt] resolved tensor name engine=" << e.name
                << " requested=" << name << " resolved=" << exact_match
                << " idx=" << idx << "\n";
    }
    return exact_match;
  }
  for (int i = 0; i < nb; ++i) {
    const char* tn = e.engine->getIOTensorName(i);
    if (!tn) continue;
    if (std::strncmp(tn, name, name_len) == 0 && tn[name_len] == '.') {
      const char* suffix = tn + name_len + 1;
      if (!suffix || !*suffix) continue;
      bool digits_only = true;
      for (const char* p = suffix; *p; ++p) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) {
          digits_only = false;
          break;
        }
      }
      if (!digits_only) continue;
      if (suffix_match && std::strcmp(suffix_match, tn) != 0) {
        throw std::runtime_error(std::string("Ambiguous tensor binding for ") + name +
                                 " (multiple suffix matches)");
      }
      suffix_match = tn;
    }
  }
  if (suffix_match) {
    static std::mutex mu;
    static std::unordered_set<std::string> seen;
    const std::string key = e.name + ":" + name;
    std::lock_guard<std::mutex> lock(mu);
    if (seen.insert(key).second) {
      const int idx = io_tensor_index(e, suffix_match);
      std::cerr << "[parakeet_trt] resolved tensor name engine=" << e.name
                << " requested=" << name << " resolved=" << suffix_match
                << " idx=" << idx << "\n";
    }
    return suffix_match;
  }
  return nullptr;
}

static bool engine_has_tensor(const TrtEngine& e, const char* name) {
  return resolve_tensor_name(e, name) != nullptr;
}

static void* tensor_ptr_or_throw(const TrtEngine& e, const char* name) {
  const char* resolved = resolve_tensor_name(e, name);
  if (!resolved) {
    throw std::runtime_error(std::string("Missing tensor binding: ") + name);
  }
  const auto it = e.tensors.find(resolved);
  if (it == e.tensors.end() || !it->second) {
    throw std::runtime_error(std::string("Missing tensor pointer: ") + resolved);
  }
  return it->second;
}

const char* dtype_name(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return "f32";
    case nvinfer1::DataType::kHALF:
      return "f16";
    case nvinfer1::DataType::kINT8:
      return "i8";
    case nvinfer1::DataType::kINT32:
      return "i32";
    case nvinfer1::DataType::kINT64:
      return "i64";
    case nvinfer1::DataType::kBOOL:
      return "bool";
    default:
      return "unknown";
  }
}

std::string dims_to_string(const nvinfer1::Dims& d) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < d.nbDims; ++i) {
    if (i > 0) oss << "x";
    oss << d.d[i];
  }
  oss << "]";
  return oss.str();
}

void dump_engine_bindings(const TrtEngine& e, cudaStream_t stream) {
  const int nb = e.engine->getNbIOTensors();
  std::ostringstream oss;
  oss << "[parakeet_trt] bindings engine=" << e.name << " stream=0x"
      << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec;
  for (int i = 0; i < nb; ++i) {
    const char* tn = e.engine->getIOTensorName(i);
    const auto dims = e.ctx->getTensorShape(tn);
    const auto dt = e.engine->getTensorDataType(tn);
    const size_t bytes = volume(dims) * dtype_size(dt);
    void* ptr = nullptr;
    const auto it = e.tensors.find(tn);
    if (it != e.tensors.end()) {
      ptr = it->second;
    }
    size_t buf_idx = 0;
    size_t buf_bytes = 0;
    bool buf_match = false;
    for (size_t j = 0; j < e.bufs.size(); ++j) {
      if (e.bufs[j].ptr == ptr) {
        buf_idx = j;
        buf_bytes = e.bufs[j].bytes;
        buf_match = true;
        break;
      }
    }
    const uintptr_t base_ptr = buf_match ? reinterpret_cast<uintptr_t>(e.bufs[buf_idx].ptr) : 0;
    const uintptr_t cur_ptr = reinterpret_cast<uintptr_t>(ptr);
    const intptr_t offset = buf_match ? static_cast<intptr_t>(cur_ptr - base_ptr) : 0;
    oss << " " << tn << "=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr)
        << std::dec << " dims=" << dims_to_string(dims)
        << " dtype=" << dtype_name(dt) << " bytes=" << bytes
        << " buf_idx=" << (buf_match ? static_cast<int>(buf_idx) : -1)
        << " buf_cap=" << e.bufs.size()
        << " base=0x" << std::hex << base_ptr
        << " offset=0x" << offset
        << std::dec << " buf_bytes=" << buf_bytes;
  }
  std::cerr << oss.str() << "\n";
}

static void set_input_shape_or_throw(TrtEngine& e, const char* name, const std::vector<int32_t>& dims) {
  const char* resolved = resolve_tensor_name(e, name);
  if (!resolved) {
    throw std::runtime_error(std::string("Missing binding: ") + name);
  }
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int>(dims.size());
  for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[static_cast<size_t>(i)];
#if NV_TENSORRT_MAJOR >= 10
  if (!e.ctx->setInputShape(resolved, d)) {
    throw std::runtime_error(std::string("setInputShape failed for ") + resolved);
  }
#else
  const int idx = e.engine->getBindingIndex(resolved);
  if (idx < 0) throw std::runtime_error(std::string("Missing binding: ") + resolved);
  if (!e.ctx->setBindingDimensions(idx, d)) throw std::runtime_error(std::string("setBindingDimensions failed for ") + name);
#endif
}

static void set_tensor_address_or_throw(TrtEngine& e, const char* name, void* ptr) {
  const char* resolved = resolve_tensor_name(e, name);
  if (!resolved) {
    throw std::runtime_error(std::string("Missing binding: ") + name);
  }
#if NV_TENSORRT_MAJOR >= 10
  if (!e.ctx->setTensorAddress(resolved, ptr)) {
    throw std::runtime_error(std::string("setTensorAddress failed for ") + resolved);
  }
#else
  (void)ptr;
  throw std::runtime_error("TensorRT < 10 not supported by this demo runtime");
#endif
  e.tensors[resolved] = ptr;
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
      debug_cuda_free(b.ptr, "alloc_buffers:cudaFree");
      b.ptr = nullptr;
      b.bytes = 0;
    }
    cuda_check(cudaMalloc(&b.ptr, bytes), "cudaMalloc");
    b.bytes = bytes;
    e.tensors[tn] = b.ptr;
    if (!e.ctx->setTensorAddress(tn, b.ptr)) throw std::runtime_error(std::string("setTensorAddress failed for ") + tn);
    debug_memset_async(b.ptr, 0, bytes, stream, "alloc_buffers:memset", nullptr);
  }
#else
  (void)stream;
  throw std::runtime_error("TensorRT < 10 not supported by this demo runtime");
#endif
}

static size_t slot_reuse_cap_for_engine(const TrtEngine& e) {
  const auto cfg = get_slot_reuse_config();
  if (cfg.slot_cap_override > 0) {
    return static_cast<size_t>(cfg.slot_cap_override);
  }
  return e.bufs.size();
}

static void ensure_slot_events(TrtEngine& e, size_t cap) {
  if (cap == 0) return;
  if (e.slot_events.size() == cap) return;
  for (size_t i = 0; i < e.slot_events.size(); ++i) {
    if (e.slot_events[i]) {
      debug_cuda_event_destroy(e.slot_events[i], e.name.c_str());
    }
  }
  e.slot_events.clear();
  e.slot_event_ready.clear();
  e.slot_events.resize(cap, nullptr);
  e.slot_event_ready.resize(cap, 0);
  for (size_t i = 0; i < cap; ++i) {
    cudaEvent_t ev{};
    const cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      std::cerr << "[parakeet_trt] slot_reuse_event_create_failed engine=" << e.name
                << " slot=" << i << " err=" << cudaGetErrorString(err) << "\n";
      continue;
    }
    e.slot_events[i] = ev;
  }
}

static void destroy_slot_events(TrtEngine& e) {
  for (auto ev : e.slot_events) {
    if (ev) cudaEventDestroy(ev);
  }
  e.slot_events.clear();
  e.slot_event_ready.clear();
  e.slot_seq = 0;
  e.slot_log_count = 0;
}

static void slot_reuse_check_before_enqueue(TrtEngine& e,
                                            cudaStream_t stream,
                                            const DebugContext* ctx,
                                            size_t* slot_out) {
  const auto cfg = get_slot_reuse_config();
  if (!cfg.enabled) return;
  const size_t cap = slot_reuse_cap_for_engine(e);
  if (cap == 0) return;
  ensure_slot_events(e, cap);
  if (e.slot_events.empty()) return;
  const size_t slot = static_cast<size_t>(e.slot_seq % cap);
  if (slot_out) *slot_out = slot;
  if (slot >= e.slot_events.size() || !e.slot_event_ready[slot] || !e.slot_events[slot]) {
    return;
  }
  const cudaError_t q = cudaEventQuery(e.slot_events[slot]);
  if (q == cudaErrorNotReady) {
    if (cfg.log_limit > 0 && e.slot_log_count < cfg.log_limit) {
      e.slot_log_count++;
      std::cerr << "[parakeet_trt] slot_reuse_inflight engine=" << e.name
                << " id=" << (ctx ? ctx->id : "")
                << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
                << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
                << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
                << " slot=" << slot << " cap=" << cap
                << " mode=" << slot_reuse_mode_name(cfg.mode)
                << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
                << "\n";
    }
    if (cfg.mode == SlotReuseMode::kWait) {
      cudaEventSynchronize(e.slot_events[slot]);
    } else if (cfg.mode == SlotReuseMode::kFail) {
      throw std::runtime_error("slot reuse detected (in-flight event)");
    }
  } else if (q != cudaSuccess) {
    if (cfg.log_limit > 0 && e.slot_log_count < cfg.log_limit) {
      e.slot_log_count++;
      std::cerr << "[parakeet_trt] slot_reuse_query_err engine=" << e.name
                << " id=" << (ctx ? ctx->id : "")
                << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
                << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
                << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
                << " slot=" << slot << " cap=" << cap
                << " err=" << cudaGetErrorString(q)
                << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
                << "\n";
    }
  }
}

static void slot_reuse_record_after_enqueue(TrtEngine& e,
                                            cudaStream_t stream,
                                            size_t slot) {
  const auto cfg = get_slot_reuse_config();
  if (!cfg.enabled) return;
  if (e.slot_events.empty() || slot >= e.slot_events.size() || !e.slot_events[slot]) {
    return;
  }
  const cudaError_t rec = cudaEventRecord(e.slot_events[slot], stream);
  if (rec != cudaSuccess && cfg.log_limit > 0 && e.slot_log_count < cfg.log_limit) {
    e.slot_log_count++;
    std::cerr << "[parakeet_trt] slot_reuse_record_err engine=" << e.name
              << " slot=" << slot << " err=" << cudaGetErrorString(rec) << "\n";
  } else {
    e.slot_event_ready[slot] = 1;
  }
  e.slot_seq = e.slot_seq + 1;
}

static bool debug_sync_target_matches(const TrtEngine& e, DebugSyncTarget target) {
  switch (target) {
    case DebugSyncTarget::kEncoder:
      return e.name == "encoder";
    case DebugSyncTarget::kPredictor:
      return e.name == "predictor";
    case DebugSyncTarget::kJoint:
      return e.name == "joint";
    case DebugSyncTarget::kAll:
    default:
      return true;
  }
}

static bool should_debug_sync(TrtEngine& e, const DebugSyncConfig& cfg, const DebugContext* ctx) {
  if (!cfg.enabled) return false;
  if (!debug_sync_target_matches(e, cfg.target)) return false;
  if (cfg.limit == 0) return true;
  if (ctx) {
    if (!e.debug_sync_utt_seq_set || e.debug_sync_utt_seq != ctx->utt_seq) {
      e.debug_sync_utt_seq = ctx->utt_seq;
      e.debug_sync_count = 0;
      e.debug_sync_utt_seq_set = true;
    }
  }
  if (e.debug_sync_count >= cfg.limit) return false;
  e.debug_sync_count += 1;
  return true;
}

static void enqueue_or_throw(TrtEngine& e, cudaStream_t stream, const DebugContext* ctx) {
#if NV_TENSORRT_MAJOR >= 10
  log_slot_reuse_config_once();
  log_debug_sync_config_once();
  size_t slot_idx = 0;
  slot_reuse_check_before_enqueue(e, stream, ctx, &slot_idx);
  const auto t0 = std::chrono::steady_clock::now();
  size_t free_before = 0;
  size_t total_before = 0;
  const cudaError_t mem_before_err = cudaMemGetInfo(&free_before, &total_before);
  const bool ok = e.ctx->enqueueV3(stream);
  const auto t1 = std::chrono::steady_clock::now();
  const uint64_t elapsed_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  size_t free_after = 0;
  size_t total_after = 0;
  const cudaError_t mem_after_err = cudaMemGetInfo(&free_after, &total_after);
  const cudaError_t peek_err = cudaPeekAtLastError();
  const uint64_t slow_ms = get_slow_enqueue_ms();
  const bool slow = slow_ms > 0 && elapsed_ms >= slow_ms;
  const auto dbg_cfg = get_debug_sync_config();
  if (should_debug_sync(e, dbg_cfg, ctx)) {
    const auto s0 = std::chrono::steady_clock::now();
    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    const auto s1 = std::chrono::steady_clock::now();
    const uint64_t sync_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count());
    const cudaError_t post_err = cudaPeekAtLastError();
    std::cerr << std::fixed << std::setprecision(1);
    std::cerr << "[parakeet_trt] debug_sync engine=" << e.name
              << " id=" << (ctx ? ctx->id : "")
              << " utt_seq=" << (ctx ? ctx->utt_seq : 0)
              << " audio_chunk_idx=" << (ctx ? ctx->audio_chunk_idx : 0)
              << " feature_idx=" << (ctx ? ctx->feature_idx : 0)
              << " enqueue_ok=" << (ok ? 1 : 0)
              << " enqueue_ms=" << elapsed_ms
              << " sync_ms=" << sync_ms
              << " sync_err=" << cudaGetErrorString(sync_err)
              << " post_err=" << cudaGetErrorString(post_err)
              << " enqueue_err=" << cudaGetErrorString(peek_err)
              << "\n";
    std::cerr.unsetf(std::ios::floatfield);
  }
  if (slow || !ok || peek_err != cudaSuccess) {
    const double mb = 1024.0 * 1024.0;
    const bool mem_ok = mem_before_err == cudaSuccess && mem_after_err == cudaSuccess;
    const double free_before_mb = mem_ok ? static_cast<double>(free_before) / mb : -1.0;
    const double free_after_mb = mem_ok ? static_cast<double>(free_after) / mb : -1.0;
    const double delta_mb = mem_ok ? (free_before_mb - free_after_mb) : 0.0;
    std::cerr << std::fixed << std::setprecision(1);
    std::cerr << "[parakeet_trt] enqueueV3 engine=" << e.name
              << " enqueue_ms=" << elapsed_ms
              << " slow_ms=" << slow_ms
              << " ok=" << (ok ? 1 : 0)
              << " mem_free_before_mb=" << free_before_mb
              << " mem_free_after_mb=" << free_after_mb
              << " mem_delta_mb=" << (mem_ok ? delta_mb : 0.0)
              << " mem_before_err=" << cudaGetErrorString(mem_before_err)
              << " mem_after_err=" << cudaGetErrorString(mem_after_err)
              << " cuda_err=" << cudaGetErrorString(peek_err)
              << "\n";
    std::cerr.unsetf(std::ios::floatfield);
    if (get_env_bool("PARAKEET_DUMP_BINDINGS_ON_SLOW", false)) {
      dump_engine_bindings(e, stream);
    }
  }
  slot_reuse_record_after_enqueue(e, stream, slot_idx);
  if (!ok) {
    throw std::runtime_error("enqueueV3 failed for engine: " + e.name);
  }
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

struct ParakeetEventInternal {
  ParakeetEventType type;
  int32_t segment_id;
  std::string text;
  std::string error_message;
};

}  // namespace

struct ParakeetSession {
  std::string model_dir;
  int32_t device_id = 0;
  bool use_fp16 = true;
  bool enc_streaming = false;
  bool enc_cache_zeroed = false;
  bool cache_enabled = false;
  bool cache_len_in_set = false;
  bool cache_len_pre_logged = false;
  bool cache_len_logged = false;
  bool cache_out_logged = false;
  bool cache_enable_logged = false;
  int cache_out_state = -1;  // -1 unknown, 0 zero, 1 nonzero
  int64_t cache_len_in = 0;
  int64_t cache_len_capacity = 0;
  size_t cache_ch_bytes = 0;
  size_t cache_tm_bytes = 0;

  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtEngine enc;
  TrtEngine pred;
  TrtEngine joint;
  cudaStream_t stream{};

  std::shared_ptr<Tokenizer> tokenizer;
  std::unique_ptr<Decoder> decoder;
  int32_t tok_start = -1;  // e.g. <|startoftranscript|>
  int32_t tok_lang = -1;   // e.g. <|en|>
  int32_t tok_nopnc = -1;  // e.g. <|nopnc|>
  int32_t tok_noitn = -1;  // e.g. <|noitn|>

  // Streaming decode state: last token id fed into predictor (carried across push_features calls).
  int32_t y_id = kBlankId;

  // Streaming partial hypothesis state (accumulated across push_features calls).
  std::vector<int> accumulated_tokens;
  std::chrono::steady_clock::time_point last_partial_emit;
  int last_partial_token_count = 0;

  DebugContext debug;

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
  void* d_cache_last_channel = nullptr;
  void* d_cache_last_time = nullptr;
  void* d_cache_last_channel_len = nullptr;
  void* d_cache_last_channel_out = nullptr;
  void* d_cache_last_time_out = nullptr;
  void* d_cache_last_channel_len_out = nullptr;
  void* cache_ch_in_ptr = nullptr;
  void* cache_ch_out_ptr = nullptr;
  void* cache_tm_in_ptr = nullptr;
  void* cache_tm_out_ptr = nullptr;
  bool cache_ptrs_init = false;

  // Event plumbing.
  std::queue<ParakeetEventInternal> event_queue;
  std::mutex event_mutex;
  std::string last_poll_text;
  std::string last_poll_err;

  std::vector<uint16_t> host_joint_logits_fp16;
  std::vector<float> host_joint_logits_f32;

  ParakeetSession(const ParakeetConfig* config)
      : model_dir(config->model_dir), device_id(config->device_id), use_fp16(config->use_fp16),
        last_partial_emit(std::chrono::steady_clock::now() - std::chrono::milliseconds(1000)) {
    host_joint_logits_fp16.resize(kJointVocabSize);
    host_joint_logits_f32.resize(kJointVocabSize);
  }
};

ParakeetSession* parakeet_create_session(const ParakeetConfig* config) {
  if (!config || !config->model_dir) return nullptr;
  try {
    auto session = std::make_unique<ParakeetSession>(config);
    log_slow_enqueue_config_once();

    cuda_check(cudaSetDevice(session->device_id), "cudaSetDevice");
    cuda_check(cudaStreamCreate(&session->stream), "cudaStreamCreate");

    session->runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!session->runtime) throw std::runtime_error("Failed to create TensorRT runtime");

    const char* override_encoder = std::getenv("PARAKEET_STREAMING_ENCODER_PATH");
    if (!override_encoder || !*override_encoder) {
      override_encoder = std::getenv("PARAKEET_ENCODER_PATH");
    }
    if (override_encoder && *override_encoder) {
      std::cerr << "[parakeet_trt] encoder_override_path=" << override_encoder << "\n";
    }

    auto load = [&](const std::string& name, const char* explicit_path) -> TrtEngine {
      TrtEngine e;
      e.name = name;
      const std::string path =
          (explicit_path && *explicit_path) ? std::string(explicit_path) : (session->model_dir + "/" + name + ".engine");
      auto data = read_file(path);
      nvinfer1::ICudaEngine* raw_engine = session->runtime->deserializeCudaEngine(data.data(), data.size());
      if (!raw_engine) throw std::runtime_error("Failed to deserialize engine: " + path);
      nvinfer1::IExecutionContext* raw_ctx = raw_engine->createExecutionContext();
      if (!raw_ctx) throw std::runtime_error("Failed to create execution context: " + name);
      e.engine = TrtUniquePtr<nvinfer1::ICudaEngine>(raw_engine);
      e.ctx = TrtUniquePtr<nvinfer1::IExecutionContext>(raw_ctx);
      return e;
    };

    session->enc = load("encoder", override_encoder);
    session->pred = load("predictor", nullptr);
    session->joint = load("joint", nullptr);

    const bool has_cache_ch = engine_has_tensor(session->enc, "cache_last_channel");
    const bool has_cache_tm = engine_has_tensor(session->enc, "cache_last_time");
    const bool has_cache_len = engine_has_tensor(session->enc, "cache_last_channel_len");
    const bool has_cache_len_out = engine_has_tensor(session->enc, "cache_last_channel_len_out");
    const bool has_cache_ch_out = engine_has_tensor(session->enc, "cache_last_channel_out");
    const bool has_cache_tm_out = engine_has_tensor(session->enc, "cache_last_time_out");
    session->enc_streaming = has_cache_ch && has_cache_tm && has_cache_len && has_cache_len_out && has_cache_ch_out && has_cache_tm_out;
    if (session->enc_streaming) {
      std::cerr << "[parakeet_trt] encoder_streaming=1\n";
    }


    // Tokenizer + decoder.
    const std::string vocab_path = find_vocab_path(session->model_dir);
    session->tokenizer = std::make_shared<Tokenizer>(vocab_path);
    // Token vocab includes blank_id; duration head handled separately.
    session->decoder = std::make_unique<Decoder>(session->tokenizer, kBlankId, kTokenVocabSize);

    // Attempt to locate prompt tokens in the vocab (NeMo-style). These are needed to prime the predictor.
    session->tok_start = find_token_id_in_vocab_txt(vocab_path, "<|startoftranscript|>");
    session->tok_lang = find_token_id_in_vocab_txt(vocab_path, "<|en|>");
    session->tok_nopnc = find_token_id_in_vocab_txt(vocab_path, "<|nopnc|>");
    session->tok_noitn = find_token_id_in_vocab_txt(vocab_path, "<|noitn|>");
    // #region agent log
    dbglog_ndjson(
        "H8",
        "cpp/src/parakeet_trt.cpp:parakeet_create_session:vocab",
        "Vocab prompt token ids",
        std::string("{\"vocab_path_len\":") + std::to_string(vocab_path.size()) +
            ",\"tok_start\":" + std::to_string(session->tok_start) +
            ",\"tok_lang\":" + std::to_string(session->tok_lang) +
            ",\"tok_nopnc\":" + std::to_string(session->tok_nopnc) +
            ",\"tok_noitn\":" + std::to_string(session->tok_noitn) + "}");
    // #endregion

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
    const size_t h_bytes = volume(session->pred.ctx->getTensorShape("h")) * dtype_size(session->pred.engine->getTensorDataType("h"));
    const size_t c_bytes = volume(session->pred.ctx->getTensorShape("c")) * dtype_size(session->pred.engine->getTensorDataType("c"));
  debug_memset_async(session->d_h, 0, h_bytes, session->stream, "prime_state:h", &session->debug);
  debug_memset_async(session->d_c, 0, c_bytes, session->stream, "prime_state:c", &session->debug);

    // Configure joint shapes for slice mode: encoder_output [1,1024,16], predictor_output [1,640,1].
    set_input_shape_or_throw(session->joint, "encoder_output", {1, kEncDim, kTrtChunkT});
    set_input_shape_or_throw(session->joint, "predictor_output", {1, kPredDim, 1});
    allocate_buffers_for_current_shapes(session->joint, session->stream);
    session->d_joint_enc_in = session->joint.tensors["encoder_output"];
    session->d_joint_pred_in = session->joint.tensors["predictor_output"];
    session->d_joint_out = session->joint.tensors["joint_output"];

    // Initialize utterance state once (reset + prompt priming).
    parakeet_reset_utterance(session.get());
    // #region agent log
    dbglog_ndjson(
        "H14",
        "cpp/src/parakeet_trt.cpp:parakeet_create_session:init",
        "Session primed (streaming)",
        std::string("{\"y_id\":") + std::to_string(session->y_id) + "}");
    // #endregion

    return session.release();
  } catch (const std::exception& e) {
    std::cerr << "Failed to create session: " << e.what() << std::endl;
    return nullptr;
  }
}

void parakeet_destroy_session(ParakeetSession* session) {
  if (!session) return;
  debug_device_sync(DebugDeviceSyncPoint::kDestroy, "destroy_session:pre", &session->debug);
  destroy_slot_events(session->enc);
  destroy_slot_events(session->pred);
  destroy_slot_events(session->joint);
  if (session->stream) {
    debug_cuda_stream_destroy(session->stream, "parakeet_destroy_session");
    session->stream = nullptr;
  }
  delete session;
}

void parakeet_reset_utterance(ParakeetSession* session) {
  if (!session) return;
  session->decoder->reset();
  // Reset predictor state to zeros.
  debug_memset_async(session->d_h, 0, kPredLayers * 1 * kPredDim * 2, session->stream, "reset:h", &session->debug);
  debug_memset_async(session->d_c, 0, kPredLayers * 1 * kPredDim * 2, session->stream, "reset:c", &session->debug);

  // Reset accumulated tokens for streaming partial emission.
  session->accumulated_tokens.clear();
  session->last_partial_token_count = 0;
  session->last_partial_emit = std::chrono::steady_clock::now() - std::chrono::milliseconds(1000);
  session->enc_cache_zeroed = false;
  session->cache_enabled = false;
  session->cache_len_in_set = false;
  session->cache_len_in = 0;
  session->cache_out_state = -1;
  session->cache_ptrs_init = false;
  session->cache_ch_in_ptr = nullptr;
  session->cache_ch_out_ptr = nullptr;
  session->cache_tm_in_ptr = nullptr;
  session->cache_tm_out_ptr = nullptr;

  // Prime predictor with NeMo-style prompt tokens once per utterance.
  // This seeds h/c and initializes y_id so decoding can continue across push_features calls.
  auto prime_token = [&](int32_t tok) {
    if (tok < 0) return;
    const int64_t host_y = static_cast<int64_t>(tok);
    debug_memcpy_async(session->d_y, &host_y, sizeof(host_y), cudaMemcpyHostToDevice, session->stream,
                       "prime:y", &session->debug);
    enqueue_or_throw(session->pred, session->stream, &session->debug);
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(predictor_prime)");
    const size_t h_bytes = volume(session->pred.ctx->getTensorShape("h")) * dtype_size(session->pred.engine->getTensorDataType("h"));
    const size_t c_bytes = volume(session->pred.ctx->getTensorShape("c")) * dtype_size(session->pred.engine->getTensorDataType("c"));
    const size_t g_bytes = volume(session->pred.ctx->getTensorShape("g")) * dtype_size(session->pred.engine->getTensorDataType("g"));
    debug_memcpy_async(session->d_h, session->d_h_out, h_bytes, cudaMemcpyDeviceToDevice, session->stream,
                       "prime:h_out->h", &session->debug);
    debug_memcpy_async(session->d_c, session->d_c_out, c_bytes, cudaMemcpyDeviceToDevice, session->stream,
                       "prime:c_out->c", &session->debug);
    // Cache predictor output for the joint (so we don't need to rerun predictor on blank steps).
    debug_memcpy_async(session->d_joint_pred_in, session->d_g, g_bytes, cudaMemcpyDeviceToDevice, session->stream,
                       "prime:g->joint_pred", &session->debug);
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(predictor_prime_copy)");
  };

  // Start token first, then language token.
  // NOTE: We intentionally do NOT apply optional constraint tokens here (e.g. <|nopnc|>, <|noitn|>)
  // because in observed runs they bias decoding toward '.' spam ("pe...........") rather than real text.
  if (session->tok_start >= 0) {
    prime_token(session->tok_start);
    session->y_id = session->tok_start;
  } else {
    session->y_id = kBlankId;
  }
  if (session->tok_lang >= 0) {
    prime_token(session->tok_lang);
    session->y_id = session->tok_lang;
  }
  // Optional constraint tokens intentionally skipped; they bias decoding toward "." spam ("pe...").
  // NOTE: We do NOT force y_id to blank here anymore.
  // The decode loop uses cached predictor_output `g` and only runs predictor on non-blank emissions.

  std::lock_guard<std::mutex> lock(session->event_mutex);
  while (!session->event_queue.empty()) session->event_queue.pop();
}

void parakeet_set_debug_context(ParakeetSession* session,
                                const char* id,
                                uint64_t utt_seq,
                                uint64_t audio_chunk_idx,
                                uint64_t feature_idx) {
  if (!session) return;
  if (id) {
    session->debug.id = id;
  } else {
    session->debug.id.clear();
  }
  session->debug.utt_seq = utt_seq;
  session->debug.audio_chunk_idx = audio_chunk_idx;
  session->debug.feature_idx = feature_idx;
}

int parakeet_push_features(ParakeetSession* session, const float* features_bct_f32, size_t num_frames) {
  if (!session || !features_bct_f32) return -1;
  if (num_frames == 0) return 0;
  try {
    debug_stage_marker("push_features:enter", &session->debug, session->stream);
    // #region agent log
    dbglog_ndjson(
        "H8",
        "cpp/src/parakeet_trt.cpp:parakeet_push_features:entry",
        "push_features called",
        std::string("{\"num_frames\":") + std::to_string(num_frames) +
            ",\"y_id_in\":" + std::to_string(session->y_id) + "}");
    // #endregion

    const int32_t T_valid = static_cast<int32_t>(num_frames);
    int32_t T_shape = T_valid;
    if (session->enc_streaming) {
      if (T_valid < 584 || T_valid > 592) {
        throw std::runtime_error("num_frames must be in [584,592] for streaming encoder");
      }
    } else {
      // Encoder engines are currently profiled for 16..256 frames.
      if (num_frames > 256) {
        throw std::runtime_error("num_frames exceeds encoder profile max (256) for this demo runtime");
      }
      T_shape = std::max<int32_t>(kTrtMinT, T_valid);
    }

    // Configure encoder shapes for this chunk.
    set_input_shape_or_throw(session->enc, "audio_signal", {1, kNMels, T_shape});
    set_input_shape_or_throw(session->enc, "length", {1});
    if (session->enc_streaming) {
      set_input_shape_or_throw(session->enc, "cache_last_channel", {kEncLayers, 1, kCacheSize, kEncDim});
      set_input_shape_or_throw(session->enc, "cache_last_time", {kEncLayers, 1, kEncDim, kCacheTime});
      set_input_shape_or_throw(session->enc, "cache_last_channel_len", {1});
    }
    allocate_buffers_for_current_shapes(session->enc, session->stream);
    session->d_audio = session->enc.tensors["audio_signal"];
    session->d_length = session->enc.tensors["length"];
    session->d_enc_out = session->enc.tensors["encoder_output"];
    session->d_enc_len = session->enc.tensors["encoded_lengths"];
    if (session->enc_streaming) {
      if (!session->cache_ptrs_init) {
        session->cache_ch_in_ptr = tensor_ptr_or_throw(session->enc, "cache_last_channel");
        session->cache_tm_in_ptr = tensor_ptr_or_throw(session->enc, "cache_last_time");
        session->cache_ch_out_ptr = tensor_ptr_or_throw(session->enc, "cache_last_channel_out");
        session->cache_tm_out_ptr = tensor_ptr_or_throw(session->enc, "cache_last_time_out");
        session->cache_ptrs_init = true;
        session->enc_cache_zeroed = false;
      }
      set_tensor_address_or_throw(session->enc, "cache_last_channel", session->cache_ch_in_ptr);
      set_tensor_address_or_throw(session->enc, "cache_last_time", session->cache_tm_in_ptr);
      set_tensor_address_or_throw(session->enc, "cache_last_channel_out", session->cache_ch_out_ptr);
      set_tensor_address_or_throw(session->enc, "cache_last_time_out", session->cache_tm_out_ptr);
      session->d_cache_last_channel = session->cache_ch_in_ptr;
      session->d_cache_last_time = session->cache_tm_in_ptr;
      session->d_cache_last_channel_out = session->cache_ch_out_ptr;
      session->d_cache_last_time_out = session->cache_tm_out_ptr;
      session->d_cache_last_channel_len = tensor_ptr_or_throw(session->enc, "cache_last_channel_len");
      session->d_cache_last_channel_len_out = tensor_ptr_or_throw(session->enc, "cache_last_channel_len_out");
      if (!session->enc_cache_zeroed) {
        const char* cache_ch_name = resolve_tensor_name(session->enc, "cache_last_channel");
        const char* cache_tm_name = resolve_tensor_name(session->enc, "cache_last_time");
        if (!cache_ch_name || !cache_tm_name) {
          throw std::runtime_error("Missing cache tensor binding");
        }
        const auto cache_ch_shape = session->enc.ctx->getTensorShape(cache_ch_name);
        const auto cache_tm_shape = session->enc.ctx->getTensorShape(cache_tm_name);
        const size_t cache_ch_bytes =
            volume(cache_ch_shape) * dtype_size(session->enc.engine->getTensorDataType(cache_ch_name));
        const size_t cache_tm_bytes =
            volume(cache_tm_shape) * dtype_size(session->enc.engine->getTensorDataType(cache_tm_name));
        session->cache_ch_bytes = cache_ch_bytes;
        session->cache_tm_bytes = cache_tm_bytes;
        if (cache_ch_shape.nbDims >= 3) {
          session->cache_len_capacity = cache_ch_shape.d[2];
        } else if (cache_ch_shape.nbDims >= 1) {
          session->cache_len_capacity = cache_ch_shape.d[cache_ch_shape.nbDims - 1];
        } else {
          session->cache_len_capacity = 0;
        }
        debug_memset_async(session->d_cache_last_channel, 0, cache_ch_bytes, session->stream,
                           "enc:cache_last_channel_zero", &session->debug);
        debug_memset_async(session->d_cache_last_time, 0, cache_tm_bytes, session->stream,
                           "enc:cache_last_time_zero", &session->debug);
        session->enc_cache_zeroed = true;
      }
      const char* cache_len_in_name = resolve_tensor_name(session->enc, "cache_last_channel_len");
      if (!cache_len_in_name) {
        throw std::runtime_error("Missing tensor binding: cache_last_channel_len");
      }
      const auto cache_len_in_dt = session->enc.engine->getTensorDataType(cache_len_in_name);
      if (cache_len_in_dt != nvinfer1::DataType::kINT64 && cache_len_in_dt != nvinfer1::DataType::kINT32) {
        throw std::runtime_error("cache_last_channel_len dtype must be int32 or int64");
      }
      if (!session->cache_len_in_set) {
        session->cache_len_in = 0;
      }
      if (cache_len_in_dt == nvinfer1::DataType::kINT64) {
        const int64_t cache_len_in_host = session->cache_len_in;
        debug_memcpy_async(session->d_cache_last_channel_len, &cache_len_in_host, sizeof(cache_len_in_host),
                           cudaMemcpyHostToDevice, session->stream, "enc:cache_len_in", &session->debug);
      } else {
        const int32_t cache_len_in_host = static_cast<int32_t>(session->cache_len_in);
        debug_memcpy_async(session->d_cache_last_channel_len, &cache_len_in_host, sizeof(cache_len_in_host),
                           cudaMemcpyHostToDevice, session->stream, "enc:cache_len_in", &session->debug);
      }
      const char* cache_len_out_name = resolve_tensor_name(session->enc, "cache_last_channel_len_out");
      if (!cache_len_out_name) {
        throw std::runtime_error("Missing tensor binding: cache_last_channel_len_out");
      }
      const auto cache_len_out_shape = session->enc.ctx->getTensorShape(cache_len_out_name);
      const auto cache_len_out_dt = session->enc.engine->getTensorDataType(cache_len_out_name);
      const size_t cache_len_out_bytes = volume(cache_len_out_shape) * dtype_size(cache_len_out_dt);
      if (cache_len_out_bytes == 0) {
        throw std::runtime_error("cache_last_channel_len_out has zero-sized shape (shape=" +
                                 dims_to_string(cache_len_out_shape) + ")");
      }
      debug_memset_async(session->d_cache_last_channel_len_out, 0, cache_len_out_bytes, session->stream,
                         "enc:cache_len_out_zero", &session->debug);
      if (!session->cache_len_pre_logged) {
        int64_t cache_len_out_pre = -1;
        if (cache_len_out_dt == nvinfer1::DataType::kINT64) {
          debug_memcpy_async(&cache_len_out_pre, session->d_cache_last_channel_len_out, sizeof(cache_len_out_pre),
                             cudaMemcpyDeviceToHost, session->stream, "enc:cache_len_out_pre", &session->debug);
        } else if (cache_len_out_dt == nvinfer1::DataType::kINT32) {
          int32_t cache_len_out_pre32 = -1;
          debug_memcpy_async(&cache_len_out_pre32, session->d_cache_last_channel_len_out, sizeof(cache_len_out_pre32),
                             cudaMemcpyDeviceToHost, session->stream, "enc:cache_len_out_pre", &session->debug);
          cache_len_out_pre = static_cast<int64_t>(cache_len_out_pre32);
        }
        cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(cache_len_out_pre)");
        const int cache_len_out_idx = io_tensor_index(session->enc, cache_len_out_name);
        std::cerr << "[parakeet_trt] cache_last_channel_len_out pre binding=" << cache_len_out_name
                  << " idx=" << cache_len_out_idx
                  << " dtype=" << dtype_name(cache_len_out_dt)
                  << " dims=" << dims_to_string(cache_len_out_shape)
                  << " bytes=" << cache_len_out_bytes
                  << " value=" << cache_len_out_pre << "\n";
        session->cache_len_pre_logged = true;
      }
    }

    // Host -> device: length
    const int64_t host_len = static_cast<int64_t>(T_valid);
    debug_memcpy_async(session->d_length, &host_len, sizeof(host_len), cudaMemcpyHostToDevice, session->stream,
                       "enc:length", &session->debug);

    // Host -> device: audio features.
    nvinfer1::DataType audio_dt = session->enc.engine->getTensorDataType("audio_signal");
    if (audio_dt == nvinfer1::DataType::kHALF) {
      std::vector<uint16_t> host_audio_fp16(static_cast<size_t>(kNMels) * static_cast<size_t>(T_shape), 0);
      for (int32_t m = 0; m < kNMels; ++m) {
        for (int32_t t = 0; t < T_valid; ++t) {
          host_audio_fp16[static_cast<size_t>(m) * static_cast<size_t>(T_shape) + static_cast<size_t>(t)] = f32_to_fp16(features_bct_f32[static_cast<size_t>(m) * static_cast<size_t>(T_valid) + static_cast<size_t>(t)]);
        }
      }
      debug_memcpy_async(session->d_audio, host_audio_fp16.data(), host_audio_fp16.size() * 2,
                         cudaMemcpyHostToDevice, session->stream, "enc:audio_fp16", &session->debug);
    } else {
      std::vector<float> host_audio_f32(static_cast<size_t>(kNMels) * static_cast<size_t>(T_shape), 0.0f);
      for (int32_t m = 0; m < kNMels; ++m) {
        for (int32_t t = 0; t < T_valid; ++t) {
          host_audio_f32[static_cast<size_t>(m) * static_cast<size_t>(T_shape) + static_cast<size_t>(t)] = features_bct_f32[static_cast<size_t>(m) * static_cast<size_t>(T_valid) + static_cast<size_t>(t)];
        }
      }
      debug_memcpy_async(session->d_audio, host_audio_f32.data(), host_audio_f32.size() * 4,
                         cudaMemcpyHostToDevice, session->stream, "enc:audio_f32", &session->debug);
    }

    // Run encoder.
    debug_stage_marker("enc:pre_enqueue", &session->debug, session->stream);
    enqueue_or_throw(session->enc, session->stream, &session->debug);
    debug_stage_marker("enc:post_enqueue", &session->debug, session->stream);
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(encoder)");
    debug_stage_marker("enc:post_sync", &session->debug, session->stream);

    // Read encoded length.
    int64_t enc_len_host = 0;
    debug_memcpy_async(&enc_len_host, session->d_enc_len, sizeof(enc_len_host), cudaMemcpyDeviceToHost, session->stream,
                       "enc:encoded_lengths", &session->debug);
    cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(encoded_lengths)");
    const int32_t T_enc = static_cast<int32_t>(enc_len_host);
    if (session->enc_streaming) {
      if (T_enc != 1) {
        throw std::runtime_error("Streaming contract violated: encoded_lengths != 1");
      }
    } else if (T_enc <= 0 || T_enc > T_shape) {
      throw std::runtime_error("Invalid encoded_lengths from encoder");
    }

    if (session->enc_streaming) {
      const auto enc_out_shape = session->enc.ctx->getTensorShape("encoder_output");
      if (enc_out_shape.nbDims <= 0 || enc_out_shape.d[enc_out_shape.nbDims - 1] != 1) {
        throw std::runtime_error("Streaming contract violated: encoder_output time_dim != 1 (shape=" +
                                 dims_to_string(enc_out_shape) + ")");
      }
    }

    // #region agent log
    dbglog_ndjson(
        "H8",
        "cpp/src/parakeet_trt.cpp:parakeet_push_features:enc",
        "Encoded lengths",
        std::string("{\"T_valid\":") + std::to_string(T_valid) + ",\"T_shape\":" +
            std::to_string(T_shape) + ",\"T_enc\":" + std::to_string(T_enc) + "}");
    // #endregion

    if (session->enc_streaming) {
      const char* cache_len_out_name = resolve_tensor_name(session->enc, "cache_last_channel_len_out");
      if (!cache_len_out_name) {
        throw std::runtime_error("Missing tensor binding: cache_last_channel_len_out");
      }
      const auto cache_len_dt = session->enc.engine->getTensorDataType(cache_len_out_name);
      if (cache_len_dt != nvinfer1::DataType::kINT64 && cache_len_dt != nvinfer1::DataType::kINT32) {
        throw std::runtime_error("Streaming contract violated: cache_last_channel_len_out dtype must be int32 or int64");
      }
      const auto cache_len_out_shape = session->enc.ctx->getTensorShape(cache_len_out_name);
      const size_t cache_len_out_bytes = volume(cache_len_out_shape) * dtype_size(cache_len_dt);
      if (cache_len_out_bytes == 0) {
        throw std::runtime_error("cache_last_channel_len_out has zero-sized shape (shape=" +
                                 dims_to_string(cache_len_out_shape) + ")");
      }
      int64_t cache_len_out_val = -1;
      if (cache_len_dt == nvinfer1::DataType::kINT64) {
        int64_t cache_len_out_host = -1;
        debug_memcpy_async(&cache_len_out_host, session->d_cache_last_channel_len_out, sizeof(cache_len_out_host),
                           cudaMemcpyDeviceToHost, session->stream, "enc:cache_len_out", &session->debug);
        cache_len_out_val = cache_len_out_host;
      } else {
        int32_t cache_len_out_host = -1;
        debug_memcpy_async(&cache_len_out_host, session->d_cache_last_channel_len_out, sizeof(cache_len_out_host),
                           cudaMemcpyDeviceToHost, session->stream, "enc:cache_len_out", &session->debug);
        cache_len_out_val = static_cast<int64_t>(cache_len_out_host);
      }
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(cache_len_out)");
      if (!session->cache_len_logged) {
        const int cache_len_out_idx = io_tensor_index(session->enc, cache_len_out_name);
        std::cerr << "[parakeet_trt] cache_last_channel_len_out binding=" << cache_len_out_name
                  << " idx=" << cache_len_out_idx
                  << " dtype=" << dtype_name(cache_len_dt)
                  << " dims=" << dims_to_string(cache_len_out_shape)
                  << " bytes=" << cache_len_out_bytes
                  << " value=" << cache_len_out_val << "\n";
        session->cache_len_logged = true;
      }
      if (session->cache_out_state < 0) {
        const char* cache_ch_out_name = resolve_tensor_name(session->enc, "cache_last_channel_out");
        const char* cache_tm_out_name = resolve_tensor_name(session->enc, "cache_last_time_out");
        if (!cache_ch_out_name || !cache_tm_out_name) {
          throw std::runtime_error("Missing cache output tensor binding");
        }
        const auto cache_ch_out_shape = session->enc.ctx->getTensorShape(cache_ch_out_name);
        const auto cache_tm_out_shape = session->enc.ctx->getTensorShape(cache_tm_out_name);
        const auto cache_ch_out_dt = session->enc.engine->getTensorDataType(cache_ch_out_name);
        const auto cache_tm_out_dt = session->enc.engine->getTensorDataType(cache_tm_out_name);
        const size_t cache_ch_out_count = volume(cache_ch_out_shape);
        const size_t cache_tm_out_count = volume(cache_tm_out_shape);
        const float cache_ch_out_max =
            max_abs_device(session->d_cache_last_channel_out, cache_ch_out_count, cache_ch_out_dt, session->stream,
                           "enc:cache_ch_out", &session->debug);
        const float cache_tm_out_max =
            max_abs_device(session->d_cache_last_time_out, cache_tm_out_count, cache_tm_out_dt, session->stream,
                           "enc:cache_tm_out", &session->debug);
        const float max_eps = 1.0e-6f;
        const int new_state = (cache_ch_out_max < max_eps && cache_tm_out_max < max_eps) ? 0 : 1;
        session->cache_out_state = new_state;
        if (!session->cache_out_logged) {
          std::cerr << "[parakeet_trt] cache_out max_abs cache_last_channel_out=" << cache_ch_out_max
                    << " cache_last_time_out=" << cache_tm_out_max
                    << " state=" << new_state << "\n";
          session->cache_out_logged = true;
        }
      }

      if (cache_len_out_val == 0) {
        if (session->cache_out_state > 0) {
          throw std::runtime_error("Streaming contract violated: cache_len_out=0 but cache_out nonzero");
        }
        session->cache_enabled = false;
      } else if (cache_len_out_val == -1) {
        if (session->cache_out_state <= 0) {
          throw std::runtime_error("Streaming contract violated: cache_len_out=-1 but cache_out is zero");
        }
        if (!session->cache_enabled) {
          session->cache_enabled = true;
          if (!session->cache_len_in_set && session->cache_len_capacity > 0) {
            session->cache_len_in = session->cache_len_capacity;
            session->cache_len_in_set = true;
          }
          if (!session->cache_enable_logged) {
            std::cerr << "[parakeet_trt] cache_len_out=-1 sentinel; enabling cache propagation\n";
            session->cache_enable_logged = true;
          }
        }
      } else if (cache_len_out_val > 0) {
        if (session->cache_out_state <= 0) {
          throw std::runtime_error("Streaming contract violated: cache_len_out>0 but cache_out is zero");
        }
        session->cache_enabled = true;
      } else {
        throw std::runtime_error("Streaming contract violated: cache_last_channel_len_out < -1 (value=" +
                                 std::to_string(cache_len_out_val) + ")");
      }

      if (cache_len_out_val >= 0) {
        int64_t next_len = cache_len_out_val;
        if (session->cache_len_capacity > 0) {
          if (next_len > session->cache_len_capacity) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
              std::cerr << "[parakeet_trt] WARN: cache_len_out exceeds capacity; clamping "
                        << next_len << " -> " << session->cache_len_capacity << "\n";
            }
            next_len = session->cache_len_capacity;
          }
        }
        session->cache_len_in = next_len;
        session->cache_len_in_set = true;
      } else if (cache_len_out_val == -1 && session->cache_out_state > 0) {
        if (!session->cache_len_in_set && session->cache_len_capacity > 0) {
          session->cache_len_in = session->cache_len_capacity;
          session->cache_len_in_set = true;
        }
      }

      if (session->cache_enabled) {
        if (!session->cache_ptrs_init) {
          throw std::runtime_error("cache propagation enabled but cache pointers not initialized");
        }
        std::swap(session->cache_ch_in_ptr, session->cache_ch_out_ptr);
        std::swap(session->cache_tm_in_ptr, session->cache_tm_out_ptr);
      }
    }

    // Copy encoder output to host once for decoding.
    nvinfer1::DataType enc_out_dt = session->enc.engine->getTensorDataType("encoder_output");
    std::vector<float> host_enc_out_f32(static_cast<size_t>(kEncDim) * static_cast<size_t>(T_enc));
    if (enc_out_dt == nvinfer1::DataType::kHALF) {
      std::vector<uint16_t> tmp(host_enc_out_f32.size());
      debug_memcpy_async(tmp.data(), session->d_enc_out, tmp.size() * 2, cudaMemcpyDeviceToHost, session->stream,
                         "enc:out_fp16", &session->debug);
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize");
      for (size_t i = 0; i < tmp.size(); ++i) host_enc_out_f32[i] = fp16_to_f32(tmp[i]);
    } else {
      debug_memcpy_async(host_enc_out_f32.data(), session->d_enc_out, host_enc_out_f32.size() * 4,
                         cudaMemcpyDeviceToHost, session->stream, "enc:out_f32", &session->debug);
      cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize");
    }

    // STREAMING MODE (chunk-isolated):
    // The validated contract requires cache_len=0, so we feed zero caches and never
    // carry cache outputs across chunks. Do not enable inter-chunk caching without
    // re-running full parity/stability validation.

    // Greedy TDT/RNNT decode loop using joint slices (T=16) due to engine min profile.
    //
    // IMPORTANT semantics:
    // - Predictor runs ONLY when a non-blank token is emitted (u increments).
    // - When joint predicts blank, we advance encoder time but keep the same predictor output `g`
    //   (stored in `session->d_joint_pred_in`).
    //
    // `session->y_id` is tracked as "last emitted token id" for visibility/logging.
    int32_t y_id = session->y_id;
    if (y_id < 0) y_id = kBlankId;
    // Use session-level accumulated_tokens for partial emission across chunks.
    // Local emitted is still used for the final event of this chunk.
    std::vector<int>& emitted = session->accumulated_tokens;
    const size_t emitted_start = emitted.size();  // Track tokens emitted before this chunk
    bool did_emit_partial_event = false;
    int last_best_tok = -1;
    bool last_best_blank = false;

    nvinfer1::DataType joint_enc_dt = session->joint.engine->getTensorDataType("encoder_output");
    std::vector<float> host_joint_enc_slice_f32(static_cast<size_t>(kEncDim) * static_cast<size_t>(kTrtChunkT));
    nvinfer1::DataType joint_out_dt = session->joint.engine->getTensorDataType("joint_output");
    std::vector<float> host_joint_logits_f32(static_cast<size_t>(kJointVocabSize));

    const auto partial_interval = std::chrono::milliseconds(100);

    int time_idx = 0;
    int dbg_steps = 0;
    const int max_symbols_per_timestep = 8;  // safety cap to avoid infinite loops
    while (time_idx < T_enc) {
      // Prepare joint encoder slice once per encoder timestep (replicate frame across T=16 to satisfy profile).
      for (int32_t c = 0; c < kEncDim; ++c) {
        const float v = host_enc_out_f32[static_cast<size_t>(c) * static_cast<size_t>(T_enc) + static_cast<size_t>(time_idx)];
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(kTrtChunkT);
        for (int32_t t = 0; t < kTrtChunkT; ++t) {
          host_joint_enc_slice_f32[base + static_cast<size_t>(t)] = v;
        }
      }
      if (joint_enc_dt == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> tmp(host_joint_enc_slice_f32.size());
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = f32_to_fp16(host_joint_enc_slice_f32[i]);
        debug_memcpy_async(session->d_joint_enc_in, tmp.data(), tmp.size() * 2, cudaMemcpyHostToDevice,
                           session->stream, "joint:enc_in_fp16", &session->debug);
      } else {
        debug_memcpy_async(session->d_joint_enc_in, host_joint_enc_slice_f32.data(),
                           host_joint_enc_slice_f32.size() * 4, cudaMemcpyHostToDevice, session->stream,
                           "joint:enc_in_f32", &session->debug);
      }

      bool emitted_blank = false;  // Track if we hit a blank in the inner loop
      for (int u = 0; u < max_symbols_per_timestep && time_idx < T_enc; ++u) {
        // Run joint using the cached predictor output `g` (session->d_joint_pred_in).
        debug_stage_marker("joint:pre_enqueue", &session->debug, session->stream, time_idx);
        enqueue_or_throw(session->joint, session->stream, &session->debug);
        debug_stage_marker("joint:post_enqueue", &session->debug, session->stream, time_idx);
        cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(joint)");
        debug_stage_marker("joint:post_sync", &session->debug, session->stream, time_idx);

        // Copy logits for (t=0,u=0) to host.
        if (joint_out_dt == nvinfer1::DataType::kHALF) {
          debug_memcpy_async(session->host_joint_logits_fp16.data(), session->d_joint_out,
                             session->host_joint_logits_fp16.size() * 2, cudaMemcpyDeviceToHost, session->stream,
                             "joint:out_fp16", &session->debug);
          cuda_check(cudaStreamSynchronize(session->stream), "sync_joint_out");
          for (size_t i = 0; i < session->host_joint_logits_f32.size(); ++i) {
            float f = fp16_to_f32(session->host_joint_logits_fp16[i]);
            session->host_joint_logits_f32[i] = std::isnan(f) ? -100.0f : f;
          }
        } else {
          debug_memcpy_async(session->host_joint_logits_f32.data(), session->d_joint_out,
                             session->host_joint_logits_f32.size() * 4, cudaMemcpyDeviceToHost, session->stream,
                             "joint:out_f32", &session->debug);
          cuda_check(cudaStreamSynchronize(session->stream), "sync_joint_out");
          for (size_t i = 0; i < session->host_joint_logits_f32.size(); ++i) {
            if (std::isnan(session->host_joint_logits_f32[i])) session->host_joint_logits_f32[i] = -100.0f;
          }
        }

        // Token argmax over [0..kTokenVocabSize)
        // Apply blank penalty (positive penalizes blank, negative boosts blank).
        const float blank_penalty = get_blank_penalty();
        if (blank_penalty != 0.0f && kBlankId >= 0 && kBlankId < kTokenVocabSize) {
          session->host_joint_logits_f32[static_cast<size_t>(kBlankId)] -= blank_penalty;
        }
        int best_tok = 0;
        float best_tok_v = session->host_joint_logits_f32[0];
        int second_tok = -1;
        float second_tok_v = -1.0e9f;
        for (int32_t i = 1; i < kTokenVocabSize; ++i) {
          const float v = session->host_joint_logits_f32[static_cast<size_t>(i)];
          if (v > best_tok_v) {
            second_tok = best_tok;
            second_tok_v = best_tok_v;
            best_tok_v = v;
            best_tok = i;
          } else if (v > second_tok_v) {
            second_tok = i;
            second_tok_v = v;
          }
        }
        // No forced substitution: use blank penalty via env var instead.

        bool suppress_punct = false;
        // Suppress leading punctuation-only emission; it can poison y_id and stall decoding on some utterances.
        if (emitted.empty() && session->tokenizer && session->tokenizer->is_punct_only(best_tok)) {
          suppress_punct = true;
          best_tok = kBlankId;
          best_tok_v = session->host_joint_logits_f32[static_cast<size_t>(kBlankId)];
        }
        last_best_tok = best_tok;
        last_best_blank = (best_tok == kBlankId);

        if (get_debug_topk() && u == 0 &&
            (time_idx == 0 || time_idx == (T_enc / 2) || time_idx + 1 == T_enc)) {
          const int k = 5;
          int top_ids[k] = {0};
          float top_vals[k];
          for (int i = 0; i < k; ++i) top_vals[i] = -1.0e9f;
          for (int32_t i = 0; i < kTokenVocabSize; ++i) {
            const float v = session->host_joint_logits_f32[static_cast<size_t>(i)];
            int insert = -1;
            for (int j = 0; j < k; ++j) {
              if (v > top_vals[j]) {
                insert = j;
                break;
              }
            }
            if (insert >= 0) {
              for (int j = k - 1; j > insert; --j) {
                top_vals[j] = top_vals[j - 1];
                top_ids[j] = top_ids[j - 1];
              }
              top_vals[insert] = v;
              top_ids[insert] = i;
            }
          }
          std::string topk = "[";
          for (int j = 0; j < k; ++j) {
            if (j) topk += ",";
            topk += "{\"id\":" + std::to_string(top_ids[j]) +
                    ",\"v\":" + std::to_string(top_vals[j]) + "}";
          }
          topk += "]";
          dbglog_ndjson(
              "H15",
              "cpp/src/parakeet_trt.cpp:parakeet_push_features:topk",
              "Top-k logits (pre-decode)",
              std::string("{\"time_idx\":") + std::to_string(time_idx) +
                  ",\"t_enc\":" + std::to_string(T_enc) +
                  ",\"blank_logit\":" + std::to_string(session->host_joint_logits_f32[static_cast<size_t>(kBlankId)]) +
                  ",\"topk\":" + topk + "}");
        }

        // Duration argmax over tail [kTokenVocabSize..kJointVocabSize)
        int best_dur_idx = 0;
        float best_dur_v = session->host_joint_logits_f32[static_cast<size_t>(kTokenVocabSize)];
        for (int32_t i = 1; i < kNumDurations; ++i) {
          const float v = session->host_joint_logits_f32[static_cast<size_t>(kTokenVocabSize + i)];
          if (v > best_dur_v) {
            best_dur_v = v;
            best_dur_idx = i;
          }
        }
        const int duration = kDurationValues[best_dur_idx];
        // Evidence-driven experiment: duration head behavior appears to over-skip encoder time and suppress
        // meaningful emissions (we saw frequent duration=4 and almost no tokens). For now, always advance
        // by 1 on blank so we evaluate each encoder timestep. Keep logging duration for analysis.
        const int advance = 1;

        // #region agent log
        if (dbg_steps < 18) {
          dbg_steps++;
          dbglog_ndjson(
              "H15",
              "cpp/src/parakeet_trt.cpp:parakeet_push_features:decode_step_v2",
              "Decode step decision (v2)",
              std::string("{\"time_idx\":") + std::to_string(time_idx) +
                  ",\"u\":" + std::to_string(u) +
                  ",\"y_id\":" + std::to_string(y_id) +
                  ",\"best_tok\":" + std::to_string(best_tok) +
                  ",\"best_tok_v\":" + std::to_string(best_tok_v) +
                  ",\"blank_logit\":" + std::to_string(session->host_joint_logits_f32[static_cast<size_t>(kBlankId)]) +
                  ",\"second_tok\":" + std::to_string(second_tok) +
                  ",\"second_tok_v\":" + std::to_string(second_tok_v) +
                  ",\"is_blank\":" + std::string(best_tok == kBlankId ? "true" : "false") +
                  ",\"suppressed_punct\":" + std::string(suppress_punct ? "true" : "false") +
                  ",\"best_dur_idx\":" + std::to_string(best_dur_idx) +
                  ",\"duration\":" + std::to_string(duration) +
                  ",\"advance\":" + std::to_string(advance) + "}");
        }
        // #endregion

        if (best_tok != kBlankId) {
          emitted.push_back(best_tok);
          // Predictor update (ONLY on non-blank token): y=best_tok, update h/c and refresh cached `g`.
          const size_t h_bytes = volume(session->pred.ctx->getTensorShape("h")) * dtype_size(session->pred.engine->getTensorDataType("h"));
          const size_t c_bytes = volume(session->pred.ctx->getTensorShape("c")) * dtype_size(session->pred.engine->getTensorDataType("c"));
          const size_t g_bytes = volume(session->pred.ctx->getTensorShape("g")) * dtype_size(session->pred.engine->getTensorDataType("g"));

          const int64_t host_y = static_cast<int64_t>(best_tok);
          debug_memcpy_async(session->d_y, &host_y, sizeof(host_y), cudaMemcpyHostToDevice, session->stream,
                             "predictor:y", &session->debug);
          debug_stage_marker("pred:pre_enqueue", &session->debug, session->stream, time_idx);
          enqueue_or_throw(session->pred, session->stream, &session->debug);
          debug_stage_marker("pred:post_enqueue", &session->debug, session->stream, time_idx);
          cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(predictor)");
          debug_stage_marker("pred:post_sync", &session->debug, session->stream, time_idx);

          debug_memcpy_async(session->d_h, session->d_h_out, h_bytes, cudaMemcpyDeviceToDevice, session->stream,
                             "predictor:h_out->h", &session->debug);
          debug_memcpy_async(session->d_c, session->d_c_out, c_bytes, cudaMemcpyDeviceToDevice, session->stream,
                             "predictor:c_out->c", &session->debug);
          debug_memcpy_async(session->d_joint_pred_in, session->d_g, g_bytes, cudaMemcpyDeviceToDevice, session->stream,
                             "predictor:g->joint_pred", &session->debug);
          cuda_check(cudaStreamSynchronize(session->stream), "cudaStreamSynchronize(predictor_commit)");
          debug_stage_marker("pred:post_commit", &session->debug, session->stream, time_idx);

          y_id = best_tok;
          continue;  // stay on the same encoder timestep, try emitting more symbols
        }

        // Blank: advance encoder time using duration head.
        emitted_blank = true;
        time_idx += advance;

        break;  // move to next encoder time window
      }

      // Safety: if inner loop exhausted max_symbols_per_timestep without emitting blank,
      // force time_idx advancement to prevent infinite spin.
      // This can happen when the model produces pathological output (all non-blank tokens).
      if (!emitted_blank && time_idx < T_enc) {
        // Always log forced advance warnings - important production diagnostic
        std::cerr << "[parakeet_trt] WARN: forced time_idx advance at " << time_idx
                  << " (no blank after " << max_symbols_per_timestep << " symbols)"
                  << " id=" << session->debug.id
                  << " utt_seq=" << session->debug.utt_seq
                  << " audio_chunk_idx=" << session->debug.audio_chunk_idx
                  << " feature_idx=" << session->debug.feature_idx
                  << "\n";
        time_idx += 1;
      }

      // Emit partial hypothesis at most every ~100ms (wall-clock).
      // This uses session-level state to track accumulated tokens across push_features calls.
      if (std::chrono::steady_clock::now() - session->last_partial_emit >= partial_interval) {
        if (static_cast<int>(emitted.size()) != session->last_partial_token_count) {
          session->last_partial_token_count = static_cast<int>(emitted.size());
          ParakeetEventInternal pev{};
          pev.type = PARAKEET_EVENT_PARTIAL_TEXT;
          pev.segment_id = 0;
          const std::string partial_text = session->tokenizer ? session->tokenizer->decode(emitted) : std::string();
          pev.text = partial_text;
          {
            std::lock_guard<std::mutex> lock(session->event_mutex);
            session->event_queue.push(std::move(pev));
          }
          const size_t preview_len = 48;
          const std::string preview = partial_text.substr(0, preview_len);
          const uint64_t t_ms = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count());
          dbglog_ndjson(
              "H21",
              "cpp/src/parakeet_trt.cpp:parakeet_push_features:partial_emit",
              "Partial event queued",
              std::string("{\"utt_seq\":") + std::to_string(session->debug.utt_seq) +
                  ",\"audio_chunk_idx\":" + std::to_string(session->debug.audio_chunk_idx) +
                  ",\"feature_idx\":" + std::to_string(session->debug.feature_idx) +
                  ",\"t_ms\":" + std::to_string(t_ms) +
                  ",\"text_preview\":\"" + json_escape(preview) + "\"" +
                  ",\"text_len\":" + std::to_string(partial_text.size()) +
                  ",\"stable_prefix_len\":-1}");
          did_emit_partial_event = true;
        }
        session->last_partial_emit = std::chrono::steady_clock::now();
      }
    }

    const size_t tokens_emitted_this_chunk = emitted.size() - emitted_start;
    int last_token_id = -1;
    if (!emitted.empty()) last_token_id = emitted.back();
    int current_text_len = 0;
    if (session->tokenizer) {
      const std::string decoded = session->tokenizer->decode(emitted);
      current_text_len = static_cast<int>(decoded.size());
    }
    if (tokens_emitted_this_chunk > 0 && current_text_len == 0) {
      const int dump_n = g_empty_text_dump_n.fetch_add(1, std::memory_order_relaxed);
      if (dump_n < 12) {
        const size_t chunk_count = tokens_emitted_this_chunk;
        const size_t max_dump = 12;
        const size_t dump_count = std::min(chunk_count, max_dump);
        const size_t dump_start = emitted.size() - dump_count;
        std::string tokens_json = "[";
        for (size_t i = dump_start; i < emitted.size(); ++i) {
          const int id = emitted[i];
          const bool is_small_id = id >= 0 && id < kNumDurations;
          const bool is_blank = (id == kBlankId);
          std::string tok = session->tokenizer ? session->tokenizer->token_at(id) : std::string();
          const bool is_control = is_control_token_str(tok);
          if (i > dump_start) tokens_json += ",";
          tokens_json += std::string("{\"id\":") + std::to_string(id) +
                         ",\"blank\":" + (is_blank ? "true" : "false") +
                         ",\"small_id\":" + (is_small_id ? "true" : "false") +
                         ",\"control\":" + (is_control ? "true" : "false") +
                         ",\"tok\":\"" + json_escape(tok) + "\"}";
        }
        tokens_json += "]";
        std::cerr << "[parakeet_trt] empty_text_tokens utt_seq=" << session->debug.utt_seq
                  << " audio_chunk_idx=" << session->debug.audio_chunk_idx
                  << " feature_idx=" << session->debug.feature_idx
                  << " tokens_emitted_this_chunk=" << tokens_emitted_this_chunk
                  << " last_best_tok_id=" << last_best_tok
                  << " dump=" << tokens_json
                  << "\n";
      }
    }
    dbglog_ndjson(
        "H22",
        "cpp/src/parakeet_trt.cpp:parakeet_push_features:chunk_summary",
        "Decode chunk summary",
        std::string("{\"utt_seq\":") + std::to_string(session->debug.utt_seq) +
            ",\"audio_chunk_idx\":" + std::to_string(session->debug.audio_chunk_idx) +
            ",\"feature_idx\":" + std::to_string(session->debug.feature_idx) +
            ",\"tokens_emitted_this_chunk\":" + std::to_string(tokens_emitted_this_chunk) +
            ",\"last_token_id\":" + std::to_string(last_token_id) +
            ",\"best_tok_is_blank\":" + std::string(last_best_blank ? "true" : "false") +
            ",\"last_best_tok_id\":" + std::to_string(last_best_tok) +
            ",\"text_len\":" + std::to_string(current_text_len) +
            ",\"did_emit_partial_event\":" + std::string(did_emit_partial_event ? "true" : "false") +
            "}");

    // Emit final transcript event for this chunk.
    // Only emit the tokens added during this chunk (delta from emitted_start).
    {
      std::lock_guard<std::mutex> lock(session->event_mutex);
      ParakeetEventInternal ev{};
      ev.type = PARAKEET_EVENT_FINAL_TEXT;
      ev.segment_id = 0;
      // Only decode the tokens emitted in this chunk (not the full accumulated buffer).
      std::vector<int> chunk_tokens(emitted.begin() + static_cast<std::ptrdiff_t>(emitted_start), emitted.end());
      ev.text = session->tokenizer->decode(chunk_tokens);
      session->event_queue.push(std::move(ev));
    }

    // #region agent log
    dbglog_ndjson(
        "H8",
        "cpp/src/parakeet_trt.cpp:parakeet_push_features:final",
        "Final event queued",
        std::string("{\"emitted_ct\":") + std::to_string(emitted.size()) +
            ",\"chunk_tokens\":" + std::to_string(emitted.size() - emitted_start) +
            ",\"decoded_len\":" + std::to_string(session->event_queue.back().text.size()) + "}");
    // #endregion

    // Persist streaming predictor token id for the next chunk.
    //
    // Evidence-driven fix:
    // Carrying the last emitted token id across chunk boundaries is unstable and can cause "one update then silence".
    // Preserve the last emitted token across chunks for continuity.
    const int32_t y_id_out = y_id;
    const bool cleared_punct_state = false;
    session->y_id = y_id_out;

    // #region agent log
    dbglog_ndjson(
        "H18",
        "cpp/src/parakeet_trt.cpp:parakeet_push_features:y_id_out",
        "Persisted y_id for next chunk",
        std::string("{\"y_id_out\":") + std::to_string(y_id_out) +
            ",\"cleared_punct_state\":" + std::string(cleared_punct_state ? "true" : "false") + "}");
    // #endregion

    debug_stage_marker("push_features:before_return", &session->debug, session->stream);
    debug_device_sync(DebugDeviceSyncPoint::kPush, "push_features:end", &session->debug);

    return 0;
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(session->event_mutex);
    ParakeetEventInternal ev{};
    ev.type = PARAKEET_EVENT_ERROR;
    ev.error_message = e.what();
    session->event_queue.push(std::move(ev));
    return -2;
  }
}

bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event) {
  if (!session || !event) return false;
  std::lock_guard<std::mutex> lock(session->event_mutex);
  if (session->event_queue.empty()) return false;
  
  ParakeetEventInternal& internal = session->event_queue.front();
  session->last_poll_text = internal.text;
  session->last_poll_err = internal.error_message;

  event->type = internal.type;
  event->segment_id = internal.segment_id;
  event->text = session->last_poll_text.c_str();
  event->error_message = session->last_poll_err.c_str();

  session->event_queue.pop();
  return true;
}
