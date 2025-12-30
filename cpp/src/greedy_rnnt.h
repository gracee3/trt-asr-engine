#ifndef GREEDY_RNNT_H
#define GREEDY_RNNT_H

#include <cstdint>
#include <functional>
#include <vector>

// Minimal greedy RNN-T decode loop (tensor-only).
//
// This is intentionally engine-agnostic: it validates the control flow / termination behavior
// (emit-until-blank, per-timestep symbol cap) before wiring up TensorRT predictor/joint execution.
//
// The caller provides an argmax oracle:
//   argmax(t, u) -> token_id (including blank_id)
//
// Where:
// - t: encoder timestep index [0..T-1]
// - u: number of non-blank symbols emitted so far at this timestep [0..max_symbols_per_t-1]
//
// The greedy rule:
// - If argmax == blank_id: advance t (next encoder frame)
// - Else: emit token and continue at same t, incrementing u
//
// Returns the emitted token IDs in order.
inline std::vector<int32_t> greedy_rnnt_decode_tensor_only(
    int32_t T,
    int32_t blank_id,
    int32_t max_symbols_per_timestep,
    const std::function<int32_t(int32_t t, int32_t u)>& argmax) {
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(T));

  for (int32_t t = 0; t < T; ++t) {
    for (int32_t u = 0; u < max_symbols_per_timestep; ++u) {
      const int32_t tok = argmax(t, u);
      if (tok == blank_id) {
        break;  // advance encoder time
      }
      out.push_back(tok);
    }
  }

  return out;
}

#endif  // GREEDY_RNNT_H


