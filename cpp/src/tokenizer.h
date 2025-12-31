#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

class Tokenizer {
public:
    // Minimal vocab-based decoder (no SentencePiece dependency).
    // `vocab_path` should point to a `vocab.txt` where each line is a token string.
    explicit Tokenizer(const std::string& vocab_path);

    // Decode SentencePiece-style pieces:
    // - tokens starting with '▁' start a new word (space before the remainder)
    // - special tokens like <pad> are ignored
    std::string decode(const std::vector<int>& ids) const;

    // True if token is punctuation-only (no alnum), ignoring special tokens and SP whitespace marker.
    bool is_punct_only(int id) const;

    // Return the raw vocab token string (or empty if out of range).
    const std::string& token_at(int id) const;

    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::vector<std::string> vocab_;
};

#endif
