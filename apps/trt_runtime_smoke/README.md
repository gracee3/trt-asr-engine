# `trt_runtime_smoke`

Minimal Linux-only **TensorRT runtime smoke test** for Parakeet RNNT-TDT engines.

## Build

From repo root:

```bash
cmake -S apps/trt_runtime_smoke -B apps/trt_runtime_smoke/build
cmake --build apps/trt_runtime_smoke/build -j
```

If you don't have CUDA/TensorRT installed, you can still build a no-op binary:

```bash
cmake -S apps/trt_runtime_smoke -B apps/trt_runtime_smoke/build -DPARAKEET_MOCK=ON
cmake --build apps/trt_runtime_smoke/build -j
```

## Run

```bash
./apps/trt_runtime_smoke/build/trt_runtime_smoke --model_dir models/parakeet-tdt-0.6b-v3 --device 0
```

This will:

- Load `encoder.engine`, `predictor.engine`, `joint.engine`
- Create one execution context per engine
- Allocate device buffers once (opt shapes) and reuse them
- Run one enqueue per engine on a CUDA stream
- Print binding names/indices/dims and simple timings


