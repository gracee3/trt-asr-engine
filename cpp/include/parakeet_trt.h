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

// features: flat array of [num_frames, feature_dim]
// returns 0 on success, negative on error
int parakeet_push_features(ParakeetSession* session, const float* features, size_t num_frames);

// Poll for next event. returns true if an event was written to *event.
// event->text pointers are valid until the next call to parakeet_poll_event or session destruction.
bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event);

#ifdef __cplusplus
}
#endif

#endif // PARAKEET_TRT_H
