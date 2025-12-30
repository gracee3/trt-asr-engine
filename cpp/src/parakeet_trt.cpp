#include "parakeet_trt.h"
#include "engine_manager.h"
#include "tokenizer.h"
#include "decoder.h"
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <memory>

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
        
        session->engine_manager = std::make_unique<EngineManager>(session->model_dir, session->use_fp16);
        session->engine_manager->load_engine("encoder");
        session->engine_manager->load_engine("predictor");
        session->engine_manager->load_engine("joint");

        session->tokenizer = std::make_shared<Tokenizer>(session->model_dir + "/tokenizer.model");
        
        // These values should ideally come from model_meta.json
        int blank_id = session->tokenizer->get_blank_id();
        int vocab_size = 8192; // Default for 0.6B v3

        session->decoder = std::make_unique<Decoder>(session->tokenizer, blank_id, vocab_size);

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
        // 1. Run Encoder (assuming encoder takes [1, feature_dim, num_frames])
        // We'd need to manage GPU memory here. 
        // For this prototype, we assume EngineManager handles some allocations
        // or we use pre-allocated buffers.
        
        /* 
        std::map<std::string, void*> inputs = {{"audio_signal", d_features}};
        std::map<std::string, void*> outputs = {{"encoder_output", d_enc_output}};
        session->engine_manager->run_inference("encoder", inputs, outputs, {{"audio_signal", {1, 80, num_frames}}});
        */

        // 2. Greedy TDT Loop
        // For simplicity, let's assume d_enc_output contains [num_frames, enc_dim]
        int time_idx = 0;
        while (time_idx < (int)num_frames) {
            // A. Run Predictor on last emitted token
            // B. Run Joint on enc_frame[time_idx] and predictor_output
            
            /*
            session->engine_manager->run_inference("joint", joint_inputs, joint_outputs, ...);
            */

            // C. Decode step
            std::string out_text;
            // Mocking logits for now as we don't have GPU memory access here
            float mock_token_logits[1024] = {0}; // vocab_size
            float mock_duration_logits[5] = {0}; // num_durations
            
            int duration = session->decoder->process_step(mock_token_logits, mock_duration_logits, 5, out_text);
            
            if (!out_text.empty()) {
                std::lock_guard<std::mutex> lock(session->event_mutex);
                ParakeetEvent ev;
                ev.type = PARAKEET_EVENT_PARTIAL_TEXT;
                ev.segment_id = session->decoder->get_time_offset(); // Use time or count
                session->current_text = out_text; 
                ev.text = session->current_text.c_str();
                session->event_queue.push(ev);
            }

            // D. Advance time
            // Transition Duration Transformation: duration could be 0, 1, 2, 3...
            if (duration == 0) {
                // Potential repeat on same frame, but usually we increment at least 1 
                // if we don't emit a token to avoid stall.
                time_idx++; 
            } else {
                time_idx += duration;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(session->event_mutex);
        ParakeetEvent ev;
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

    // Note: The caller must use the string pointers before they are invalidated.
    // In this draft, we just copy the struct which contains pointers.
    // We should ensure the pointed-to memory stays alive.
    return true;
}
