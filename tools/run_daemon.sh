#!/usr/bin/env bash
set -euo pipefail

MAGNOLIA_DIR="${MAGNOLIA_DIR:-/home/emmy/git/magnolia}"
TRT_CPP_BUILD="${TRT_CPP_BUILD:-/home/emmy/git/trt-asr-engine/cpp/build}"

export LD_LIBRARY_PATH="${TRT_CPP_BUILD}:${LD_LIBRARY_PATH:-}"

cd "${MAGNOLIA_DIR}"
exec cargo run -p daemon
