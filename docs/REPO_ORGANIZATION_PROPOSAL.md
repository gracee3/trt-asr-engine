# Repo Organization Proposal

Goal: clean separation between runtime code, tooling, contracts, docs, and generated artifacts, without breaking existing tooling.

## Proposed target layout
```
trt-asr-engine/
  contracts/
    parakeet-tdt-0.6b-v3.contract.json
    schema.json
  cpp/
    runtime/
    include/
    tests/
  rust/
    crates/
      features/
      ffi/
      cli/
  tools/
    export_onnx/
    verify_nemo/
    parity/
      ort/
      trt/
    build_trt/
    stt_suite/
  docs/
    RESTART_PLAN.md
    CONTRACT_SOURCES.md
    ARCHITECTURE_RUNTIME.md
    REPO_ORGANIZATION_PROPOSAL.md
    DECISION_LOG.md
    debugging.md
  out/          # generated (gitignored)
    onnx/
    trt/
  artifacts/    # generated (gitignored)
    parity/
    stability/
    taps/
```

## Mapping from current layout
- `cpp/src/` -> `cpp/runtime/` (keep headers in `cpp/include/`).
- `rust/{features,parakeet_trt_sys,parakeet_trt,cli}` -> `rust/crates/*` (crate names unchanged).
- `tools/onnxruntime/` -> `tools/parity/ort/`.
- `tools/tensorrt/` -> `tools/parity/trt/`.
- `tools/stt_suite/` unchanged (or move under `tools/stt_suite/`).
- `tools/export_onnx/` and `tools/verify_nemo/` unchanged.
- `out/` and `artifacts/` remain gitignored with a standardized subpath layout.

## Why this layout
- Explicit separation between runtime (C++/Rust) and Python tooling.
- Parity tooling grouped by backend (ORT vs TRT).
- Contracts and docs become first-class, versioned artifacts.
- Generated outputs in predictable locations for CI and reproducibility.

## Migration steps
1. Create new directories (`cpp/runtime`, `rust/crates`, `tools/parity/*`).
2. Move code with minimal edits; add shim CMake/Cargo paths as needed.
3. Update include paths and Cargo workspace members.
4. Add symlinks or wrapper scripts to preserve old paths during transition.
5. Update docs and scripts to use new paths.
6. Remove shims after two stable releases.

## Keep existing tools working
- Provide wrapper scripts in old locations that forward to new paths.
- Preserve CLI flags and output file names.
- Use `tools/README.md` to document deprecated paths and timeline.

## Standard ladder entrypoint
Add a `justfile` or `Makefile` with canonical tasks:
1. `export` -> `tools/export_onnx/export.py`
2. `ort-parity` -> `tools/parity/ort/onnx_streaming_parity.py`
3. `trt-build` -> `tools/build_trt/build_trt.py`
4. `trt-parity` -> `tools/parity/trt/trt_streaming_parity.py`
5. `bench` -> runtime microbench harness
