#include "parakeet_trt.h"
#include <string>
#include <vector>
#include <iostream>

struct ParakeetSession {
    std::string model_dir;
    std::vector<std::string> messages;
    int32_t segment_id = 0;
};

ParakeetSession* parakeet_create_session(const ParakeetConfig* config) {
    auto session = new ParakeetSession();
    session->model_dir = config->model_dir;
    return session;
}

void parakeet_destroy_session(ParakeetSession* session) {
    delete session;
}

void parakeet_reset_utterance(ParakeetSession* session) {
    session->segment_id = 0;
    session->messages.clear();
}

int parakeet_push_features(ParakeetSession* session, const float* features, size_t num_frames) {
    session->segment_id++;
    session->messages.push_back("Mock transcription for " + std::to_string(num_frames) + " frames");
    return 0;
}

bool parakeet_poll_event(ParakeetSession* session, ParakeetEvent* event) {
    if (session->messages.empty()) return false;
    
    event->type = PARAKEET_EVENT_FINAL_TEXT;
    event->segment_id = session->segment_id;
    // Note: Leak for simplicity in mock, or use static
    static std::string last_msg;
    last_msg = session->messages.back();
    event->text = last_msg.c_str();
    session->messages.pop_back();
    
    return true;
}
