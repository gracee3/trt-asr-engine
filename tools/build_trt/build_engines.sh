#!/bin/bash
set -e

MODEL_DIR=$1
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

echo "Building TensorRT engines from ONNX..."

# Encoder
trtexec --onnx=$MODEL_DIR/encoder.onnx \
        --saveEngine=$MODEL_DIR/encoder.engine \
        --fp16 \
        --minShapes=audio_signal:1x80x1 \
        --optShapes=audio_signal:1x80x100 \
        --maxShapes=audio_signal:1x80x500 \
        --workspace=2048

# Predictor
trtexec --onnx=$MODEL_DIR/predictor.onnx \
        --saveEngine=$MODEL_DIR/predictor.engine \
        --fp16 \
        --workspace=1024

# Joint
trtexec --onnx=$MODEL_DIR/joint.onnx \
        --saveEngine=$MODEL_DIR/joint.engine \
        --fp16 \
        --workspace=1024

echo "Engines built successfully in $MODEL_DIR"
