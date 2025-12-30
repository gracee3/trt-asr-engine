#include "parakeet_trt.h"
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <memory>

// Forward declarations of internal classes
class EngineManager;
class Tokenizer;
class Decoder;

struct ParakeetSession {
    std::string model_dir;
    int32_t device_id;
    bool use_fp16;

    std::unique_ptr<EngineManager> engine_manager;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<Decoder> decoder;

    std::queue<ParakeetEvent> event_queue;
    std::mutex event_mutex;

    // Local storage for strings in events to ensure pointer validity
    std::string current_text;
    std::string current_error;

    ParakeetSession(const ParakeetConfig* config) 
        : model_dir(config->model_dir), device_id(config->device_id), use_fp16(config->use_fp16) {}
};

ParakeetSession* parakeet_create_session(const ParakeetConfig* config) {
    try {
        auto session = std::make_unique<ParakeetSession>(config);
        // Initialize engines and components here
        // (Implementation details to follow)
        return session.release();
    } catch (const std::exception& e) {
        std::cerr << "Failed to create session: " << e.what() << std::endl;
        return nullptr;
    }
}

void parakeet_destroy_session(ParakeetSession* session) {
    delete session;
}

void parakeet_reset_utterance(ParakeetSession* session) {
    if (!session) return;
    // session->decoder->reset();
    std::lock_guard<std::mutex> lock(session->event_mutex);
    while(!session->event_queue.empty()) session->event_queue.pop();
}

int parakeet_push_features(ParakeetSession* session, const float* features, size_t num_frames) {
    if (!session || !features) return -1;
    try {
        // 1. Run inference
        // 2. Run decoder
        // 3. Push events to queue
        return 0;
    } catch (const std::exception& e) {
        // push error event
        return -2;
    }
}

bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event) {
    if (!session || !event) return false;
    std::lock_guard<std::mutex> lock(session->event_mutex);
    if (session->event_queue.empty()) return false;

    *event = session->event_queue.front();
    session->event_queue.pop();

    // Note: The caller must use the string pointers before they are invalidated.
    // In this draft, we just copy the struct which contains pointers.
    // We should ensure the pointed-to memory stays alive.
    return true;
}
