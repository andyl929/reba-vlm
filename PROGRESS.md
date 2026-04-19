# REBA Project Progress

## Environment (as of 2026-04-16)
- Hardware: 2× NVIDIA A100-PCIE-40GB on NCSU VCL (vclvm176-201)
- User: ali35 (shared machine)
- OS: Ubuntu 24.04.4 LTS, CUDA 12.8, driver 570.211.01
- Conda env: reba (Python 3.12.13)

## Stack
- vLLM 0.19.0
- PyTorch 2.10.0+cu128
- Transformers 5.5.4
- ffmpeg (via conda-forge)

## Model
- google/gemma-4-31B-it (BF16, 59GB)
- Path: ~/reba-project/models/gemma-4-31B-it/

## How to Start vLLM Server
conda activate reba
~/reba-project/scripts/start_vllm_server.sh

## Sanity Check (All Passed 2026-04-16)
- Text, single image, multi-image, video: all green
- Key finding: Gemma 4 video = up to 32 frames × 70 tokens + mm:ss timestamps

## Next (2026-04-17)
- [ ] Transfer REBA videos from local → VCL
- [ ] Design REBA scoring prompts (v1 baseline)
- [ ] Code skeleton (client.py, frame_extractor.py)
- [ ] Run first experiment
