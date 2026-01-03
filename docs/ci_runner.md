# CI + Runner Notes (Laptop-Only Dev)

This repo pairs a GitHub-hosted CPU gate with an optional GPU gate that runs on
your laptop when the self-hosted runner is online.

## Required PR Gate (GitHub-Hosted)

CPU-only checks:

```bash
cargo fmt --check
cargo clippy
cargo test
cargo build -p daemon
```

Lightweight config/contract checks (no TRT):
- Validate JSON/TOML files parse and required fields exist.
- Keep these fast and deterministic so they can always run on GitHub-hosted runners.

## Optional GPU Gate (Self-Hosted Runner)

Run only when the laptop runner is online:

- TRT parity smoke (5 chunks), FP16 and/or FP32
- Contract assertions must pass (encoded_lengths=1, time_dim=1, cache_len=0)

Recommendation:
- Keep this gate non-required unless the laptop runner is reliably online.
- Enforce via PR checklist if availability is intermittent.

## Local Run Helper (Magnolia)

Use the repo-local script to avoid the `LD_LIBRARY_PATH` gotcha when running
Magnolia against the TRT runtime.

```bash
tools/run_daemon.sh
```

Env overrides:
- `MAGNOLIA_DIR` (default: `/home/emmy/git/magnolia`)
- `TRT_CPP_BUILD` (default: `/home/emmy/git/trt-asr-engine/cpp/build`)
