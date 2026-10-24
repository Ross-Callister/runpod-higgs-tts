#!/bin/bash
# Create a scale-to-zero RunPod Serverless endpoint for Higgs Audio v3 TTS (needs runpodctl >= 2.4,
# logged in). Prints the endpoint id.
#   ./deploy.sh                      # 24 GB Ampere pool (A5000 / 3090 / L4)
#   GPU="NVIDIA GeForce RTX 4090" ./deploy.sh
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/ross-callister/runpod-higgs-tts:0.1.0}
GPU=${GPU:-NVIDIA RTX A5000}
NAME=${NAME:-higgs-tts}
TPL=$(runpodctl template create --serverless --name "$NAME" --image "$IMAGE" --container-disk-in-gb 20 | jq -r .id)
echo "template $TPL" >&2
# One --model-reference only: RunPod caches one model per endpoint, and a second one stalled new
# workers at "initializing model files" for 30+ minutes. The codec is baked into the image.
runpodctl serverless create --name "$NAME" --template-id "$TPL" --gpu-id "$GPU" \
  --workers-min 0 --workers-max "${WORKERS_MAX:-1}" --idle-timeout "${IDLE_TIMEOUT:-60}" --execution-timeout 600 \
  --model-reference "https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers:main" | jq -r .id
