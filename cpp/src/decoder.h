#ifndef DECODER_H
#define DECODER_H

#include <vector>
#include <string>
#include "tokenizer.h"

class Decoder {
public:
    Decoder(std::shared_ptr<Tokenizer> tokenizer, int blank_id) 
        : tokenizer_(tokenizer), blank_id_(blank_id) {
        reset();
    }

    void reset() {
        last_token_ = 0; // Usually 0 or BOS
        hypotheses_ids_.clear();
    }

    // Returns new text segments since last call
    std::string process_logits(const float* logits, size_t vocab_size) {
        // Greedy argmax
        int best_id = -1;
        float max_logit = -1e9;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_id = i;
            }
        }

        if (best_id != blank_id_) {
            hypotheses_ids_.push_back(best_id);
            last_token_ = best_id;
            return tokenizer_->decode({best_id});
        }
        return "";
    }

private:
    std::shared_ptr<Tokenizer> tokenizer_;
    int blank_id_;
    int last_token_;
    std::vector<int> hypotheses_ids_;
};

#endif
