#include "greedy_rnnt.h"

#include <iostream>
#include <string>

static std::string join(const std::vector<int32_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i + 1 < v.size()) s += ",";
  }
  return s;
}

int main() {
  constexpr int32_t T = 5;
  constexpr int32_t blank = 0;
  constexpr int32_t max_u = 4;

  // Deterministic oracle:
  // t=0: emit 1 then blank
  // t=1: blank
  // t=2: emit 2,3 then blank
  // t>=3: blank
  auto argmax = [&](int32_t t, int32_t u) -> int32_t {
    if (t == 0) return (u == 0) ? 1 : blank;
    if (t == 1) return blank;
    if (t == 2) {
      if (u == 0) return 2;
      if (u == 1) return 3;
      return blank;
    }
    return blank;
  };

  const auto tokens = greedy_rnnt_decode_tensor_only(T, blank, max_u, argmax);
  const std::vector<int32_t> expected = {1, 2, 3};

  if (tokens != expected) {
    std::cerr << "greedy_decode_smoke FAILED\n";
    std::cerr << "  got:      [" << join(tokens) << "]\n";
    std::cerr << "  expected: [" << join(expected) << "]\n";
    return 1;
  }

  std::cout << "greedy_decode_smoke OK: [" << join(tokens) << "]\n";
  return 0;
}


