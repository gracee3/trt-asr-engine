#include "trt_asr.h"

#include "parakeet_trt.h"

#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

struct TrtAsrSession {
  ParakeetSession* inner = nullptr;
  std::queue<TrtAsrEvent> q;
  std::mutex mu;

  // Backing storage for returned pointers (valid until next poll/reset/destroy).
  std::string s_text;
  std::string s_err;

  int32_t segment_id = 0;
};

static float fp16_to_f32(uint16_t u) {
  // Minimal FP16->FP32 conversion for prototyping.
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

TrtAsrSession* trt_asr_create_session(const TrtAsrConfig* config) {
  if (!config || !config->model_dir) return nullptr;
  auto* s = new TrtAsrSession();

  ParakeetConfig pc{};
  pc.model_dir = config->model_dir;
  pc.device_id = config->device_id;
  pc.use_fp16 = config->use_fp16;
  s->inner = parakeet_create_session(&pc);
  if (!s->inner) {
    delete s;
    return nullptr;
  }
  return s;
}

void trt_asr_destroy_session(TrtAsrSession* session) {
  if (!session) return;
  if (session->inner) parakeet_destroy_session(session->inner);
  delete session;
}

void trt_asr_reset_session(TrtAsrSession* session) {
  if (!session) return;
  parakeet_reset_utterance(session->inner);
  std::lock_guard<std::mutex> lock(session->mu);
  while (!session->q.empty()) session->q.pop();
  session->s_text.clear();
  session->s_err.clear();
  session->segment_id = 0;
}

int trt_asr_push_features_f32(TrtAsrSession* session, const float* features_f32, int32_t T, int32_t length) {
  (void)length;  // currently unused by legacy API
  if (!session || !features_f32 || T <= 0) return -1;
  session->segment_id++;
  // Legacy API only accepts (features, num_frames). We treat T as frames for now.
  return parakeet_push_features(session->inner, features_f32, static_cast<size_t>(T));
}

int trt_asr_push_features_f16(TrtAsrSession* session, const uint16_t* features_f16, int32_t T, int32_t length) {
  if (!session || !features_f16 || T <= 0) return -1;
  // Prototype: CPU convert to f32 then use legacy push.
  // Layout is currently opaque; caller and runtime must agree. This conversion preserves element order.
  const size_t n = static_cast<size_t>(128) * static_cast<size_t>(T);
  std::vector<float> tmp(n);
  for (size_t i = 0; i < n; ++i) tmp[i] = fp16_to_f32(features_f16[i]);
  return trt_asr_push_features_f32(session, tmp.data(), T, length);
}

bool trt_asr_poll_event(TrtAsrSession* session, TrtAsrEvent* out_event) {
  if (!session || !out_event) return false;

  // Translate any pending legacy events into the new event stream.
  ParakeetEvent ev{};
  while (parakeet_poll_event(session->inner, &ev)) {
    TrtAsrEvent out{};
    out.segment_id = ev.segment_id;
    out.token_id = -1;
    out.text = nullptr;
    out.error_message = nullptr;

    switch (ev.type) {
      case PARAKEET_EVENT_PARTIAL_TEXT:
        out.type = TRT_ASR_EVENT_PARTIAL_TEXT;
        session->s_text = ev.text ? ev.text : "";
        out.text = session->s_text.c_str();
        break;
      case PARAKEET_EVENT_FINAL_TEXT:
        out.type = TRT_ASR_EVENT_FINAL_TEXT;
        session->s_text = ev.text ? ev.text : "";
        out.text = session->s_text.c_str();
        break;
      case PARAKEET_EVENT_ERROR:
      default:
        out.type = TRT_ASR_EVENT_ERROR;
        session->s_err = ev.error_message ? ev.error_message : "unknown error";
        out.error_message = session->s_err.c_str();
        break;
    }

    std::lock_guard<std::mutex> lock(session->mu);
    session->q.push(out);
  }

  std::lock_guard<std::mutex> lock(session->mu);
  if (session->q.empty()) return false;
  *out_event = session->q.front();
  session->q.pop();
  return true;
}


