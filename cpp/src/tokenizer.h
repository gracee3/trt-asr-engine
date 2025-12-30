#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <sentencepiece_processor.h>
#include <string>
#include <vector>

class Tokenizer {
public:
    Tokenizer(const std::string& model_path) {
        auto status = processor_.Load(model_path);
        if (!status.ok()) {
            throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
        }
    }

    std::string decode(const std::vector<int>& ids) {
        std::string text;
        processor_.Decode(ids, &text);
        return text;
    }

    int get_blank_id() const {
        return processor_.PieceToId("<blank>"); // Or from model_meta.json
    }

private:
    sentencepiece::SentencePieceProcessor processor_;
};

#endif
