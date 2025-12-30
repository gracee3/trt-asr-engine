#ifndef DECODER_H
#define DECODER_H

#include <vector>
#include <string>
#include <memory>
#include "tokenizer.h"

class Decoder {
public:
    Decoder(std::shared_ptr<Tokenizer> tokenizer, int blank_id, int vocab_size) 
        : tokenizer_(tokenizer), blank_id_(blank_id), vocab_size_(vocab_size) {
        reset();
    }

    void reset() {
        hypotheses_ids_.clear();
        time_offset_ = 0;
    }

    // Process a single joint output (token_logits and duration_logits)
    // Returns number of frames to skip
    int process_step(const float* token_logits, const float* duration_logits, 
                     int num_durations, std::string& out_text) {
        // 1. Argmax token
        int best_token = -1;
        float max_token_logit = -1e9;
        for (int i = 0; i < vocab_size_; ++i) {
            if (token_logits[i] > max_token_logit) {
                max_token_logit = token_logits[i];
                best_token = i;
            }
        }

        // 2. Argmax duration
        int best_duration = 0;
        float max_duration_logit = -1e9;
        for (int i = 0; i < num_durations; ++i) {
            if (duration_logits[i] > max_duration_logit) {
                max_duration_logit = duration_logits[i];
                best_duration = i;
            }
        }

        if (best_token != blank_id_) {
            hypotheses_ids_.push_back(best_token);
            out_text = tokenizer_->decode({best_token});
        }

        // TDT models typically predict 0, 1, 2, 3... frames to skip.
        // We always skip at least 1 in standard transducer if we reach a limit, 
        // but TDT models are trained to skip more.
        return best_duration;
    }

    int get_time_offset() const { return time_offset_; }

private:
    std::shared_ptr<Tokenizer> tokenizer_;
    int blank_id_;
    int vocab_size_;
    std::vector<int> hypotheses_ids_;
    int time_offset_;
};

#endif
