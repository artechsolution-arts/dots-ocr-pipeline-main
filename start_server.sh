#!/bin/bash
set -ex
echo "--- Starting dots.ocr server ---"
echo "Modifying vllm entrypoint..."
export PYTHONPATH=/workspace/dots.ocr:/workspace/dots.ocr/weights:$PYTHONPATH
export hf_model_path=/workspace/dots.ocr/weights/DotsOCR

if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        echo "❌ No GPU detected inside the container!"
        echo "👉 Did you forget to run with: docker run --gpus all ... ?"
        exit 1
    fi
else
    echo "⚠ nvidia-smi not found in PATH. This container may not have GPU support."
    echo "👉 Ensure you are using an NVIDIA-compatible base image and runtime."
    exit 1
fi
sed -i "/^from vllm\\.entrypoints\\.cli\\.main import main/a from DotsOCR import modeling_dots_ocr_vllm" $(which vllm)
echo "Starting server..."
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DEVICE=cuda

vllm serve ${hf_model_path} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format string \
    --served-model-name dotsocr-model \
    --trust-remote-code \
    --enforce-eager &

VLLM_PID=$!

# Optional: wait for VLLM to be ready
echo "Waiting for VLLM to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health; then
        echo "✅ VLLM ready!"
        break
    fi
    sleep 2
done

# Start the RunPod handler in foreground
python3 -u handler.py

# Wait for VLLM if handler exits
kill $VLLM_PID || true