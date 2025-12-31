#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) {
    throw std::runtime_error("Failed to open vocab file: " + vocab_path);
  }
  std::string line;
  while (std::getline(f, line)) {
    // Strip CR for Windows line endings.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    vocab_.push_back(line);
  }
  if (vocab_.empty()) {
    throw std::runtime_error("Vocab file is empty: " + vocab_path);
  }
}

static bool is_special_token(const std::string& s) {
  // Very small heuristic: ignore tokens like <pad>, <unk>, <|...|>, and the explicit "<blank>" if present.
  if (s == "<blank>" || s == "<pad>" || s == "<unk>") return true;
  if (!s.empty() && s.front() == '<' && s.back() == '>') return true;
  return false;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  out.reserve(ids.size() * 2);

  for (int id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) continue;
    const std::string& tok = vocab_[static_cast<size_t>(id)];
    if (is_special_token(tok)) continue;

    // SentencePiece "whitespace" marker U+2581 (▁)
    if (!tok.empty() && (unsigned char)tok[0] == 0xE2) {
      // UTF-8 for '▁' is 3 bytes: E2 96 81
      if (tok.size() >= 3 && (unsigned char)tok[0] == 0xE2 && (unsigned char)tok[1] == 0x96 && (unsigned char)tok[2] == 0x81) {
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
        out.append(tok.substr(3));
        continue;
      }
    }

    out.append(tok);
  }

  // Trim leading spaces.
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

bool Tokenizer::is_punct_only(int id) const {
  if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) return false;
  const std::string& tok = vocab_[static_cast<size_t>(id)];
  if (is_special_token(tok)) return false;

  size_t start = 0;
  if (tok.size() >= 3 && (unsigned char)tok[0] == 0xE2 && (unsigned char)tok[1] == 0x96 &&
      (unsigned char)tok[2] == 0x81) {
    start = 3;
  }
  if (start >= tok.size()) return false;

  bool any_alnum = false;
  bool any_non_space = false;
  for (size_t i = start; i < tok.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(tok[i]);
    if (std::isalnum(c)) {
      any_alnum = true;
      break;
    }
    if (!std::isspace(c)) {
      any_non_space = true;
    }
  }
  return any_non_space && !any_alnum;
}
