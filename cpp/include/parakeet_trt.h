#ifndef PARAKEET_TRT_H
#define PARAKEET_TRT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PARAKEET_EVENT_PARTIAL_TEXT = 0,
    PARAKEET_EVENT_FINAL_TEXT = 1,
    PARAKEET_EVENT_ERROR = 2
} ParakeetEventType;

typedef struct {
    ParakeetEventType type;
    int32_t segment_id;
    const char* text;
    const char* error_message;
} ParakeetEvent;

typedef struct ParakeetSession ParakeetSession;

typedef struct {
    const char* model_dir;
    int32_t device_id;
    bool use_fp16;
} ParakeetConfig;

ParakeetSession* parakeet_create_session(const ParakeetConfig* config);
void parakeet_destroy_session(ParakeetSession* session);

void parakeet_reset_utterance(ParakeetSession* session);

int parakeet_push_features(ParakeetSession* session, const float* features, size_t num_frames);

void parakeet_set_debug_context(ParakeetSession* session,
                                const char* id,
                                uint64_t utt_seq,
                                uint64_t audio_chunk_idx,
                                uint64_t feature_idx);

bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event);

#ifdef __cplusplus
}
#endif

#endif // PARAKEET_TRT_H
