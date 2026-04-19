#!/bin/bash
# 启动 vLLM server - Gemma 4 31B IT (修复版)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate reba

mkdir -p ~/reba-project/logs
LOG_FILE=~/reba-project/logs/vllm_server_$(date +%Y%m%d_%H%M%S).log

MODEL_PATH=~/reba-project/models/gemma-4-31B-it

echo "===== Starting vLLM server ====="
echo "Model: $MODEL_PATH"
echo "Log:   $LOG_FILE"
echo "Time:  $(date)"
echo ""

# 关键参数说明：
# --max-model-len 8192：对 20 秒视频（~20-30 帧）+ prompt + output 足够
# --gpu-memory-utilization 0.90：提高到 90%（从 0.85），解决 KV cache 不足
# --limit-mm-per-prompt：JSON 格式指定每个请求的多模态配额

vllm serve "$MODEL_PATH" \
    --served-model-name gemma-4-31B-it \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 30, "video": 1}' \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee "$LOG_FILE"
