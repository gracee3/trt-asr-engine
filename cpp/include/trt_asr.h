#ifndef TRT_ASR_H
#define TRT_ASR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Versioned, minimal C ABI intended for Rust FFI stability.
// This does NOT replace parakeet_trt.h yet; it's a forward-looking API.

typedef struct TrtAsrSession TrtAsrSession;

typedef struct {
  const char* model_dir;
  int32_t device_id;
  bool use_fp16;
} TrtAsrConfig;

typedef enum {
  TRT_ASR_EVENT_TOKEN = 0,
  TRT_ASR_EVENT_PARTIAL_TEXT = 1,
  TRT_ASR_EVENT_FINAL_TEXT = 2,
  TRT_ASR_EVENT_ERROR = 3,
} TrtAsrEventType;

typedef struct {
  TrtAsrEventType type;
  int32_t segment_id;

  // For TRT_ASR_EVENT_TOKEN
  int32_t token_id;

  // For *_TEXT / ERROR (UTF-8, owned by session; valid until next poll/reset/destroy).
  const char* text;
  const char* error_message;
} TrtAsrEvent;

TrtAsrSession* trt_asr_create_session(const TrtAsrConfig* config);
void trt_asr_destroy_session(TrtAsrSession* session);

void trt_asr_reset_session(TrtAsrSession* session);

// Push log-mel features for batch=1.
// Features layout: [128, T] contiguous (column-major-in-time: features[m* T + t] or vice versa is a contract detail).
// For now, we treat it as opaque and only provide an entry point; runtime decode will define exact layout.
int trt_asr_push_features_f16(TrtAsrSession* session, const uint16_t* features_f16, int32_t T, int32_t length);
int trt_asr_push_features_f32(TrtAsrSession* session, const float* features_f32, int32_t T, int32_t length);

bool trt_asr_poll_event(TrtAsrSession* session, TrtAsrEvent* out_event);

#ifdef __cplusplus
}
#endif

#endif  // TRT_ASR_H


