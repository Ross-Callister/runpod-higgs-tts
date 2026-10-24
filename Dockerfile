# RunPod Serverless worker for Higgs Audio v3 TTS. Packages and the 0.8 GB audio codec are baked in;
# the 8.7 GB model comes from the endpoint's cached model (--model-reference) so the image stays
# small and cold starts skip the download. FROM the stock runpod/pytorch image, whose layers
# RunPod hosts usually already hold.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
LABEL org.opencontainers.image.source="https://github.com/Ross-Callister/runpod-higgs-tts" \
      org.opencontainers.image.description="RunPod Serverless worker: Higgs Audio v3 TTS 4B with zero-shot voice cloning" \
      org.opencontainers.image.licenses="MIT"

ENV PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_CACHE=/opt/hf-cache HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

# A venv over the image's packages: pip can't replace Debian's cryptography 41 in place
# ("uninstall-no-record-file"), which the runpod SDK would otherwise try to upgrade.
RUN PY=$(command -v python || command -v python3) \
 && "$PY" -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/python -c "import torchaudio" \
 && /opt/venv/bin/python -m pip install --no-cache-dir "transformers==5.20.2" "runpod<2" soundfile \
 && /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('bosonai/higgs-audio-v2-tokenizer', revision='815a03f1b57dcb5ae918db415f22cd0c8a6f0f54', allow_patterns=['*.json', '*.safetensors'])"

COPY licenses/ /licenses/
COPY handler.py /handler.py
CMD ["/opt/venv/bin/python", "-u", "/handler.py"]
