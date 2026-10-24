"""RunPod Serverless worker for Higgs Audio v3 TTS (4B): zero-shot voice cloning from a short reference clip.

Job input (all optional except `text`):
    text            str    text to speak (long text is split into sentence chunks)
    ref_audio_b64   str    reference clip, base64 (wav/flac/mp3/ogg; 3-15 s of clean speech works best)
    ref_audio_url   str    ...or a URL to fetch it from
    ref_text        str    transcript of the reference clip (improves cloning noticeably)
    temperature     float  0.7
    top_p           float  0.95
    top_k           int    none
    seed            int    random
    format          str    "wav" (16-bit PCM) | "flac" | "mp3" | "opus"
    max_chunk_chars int    400; longer text is split at sentence ends into chunks of about this size

Output: {"audio_b64", "format", "sample_rate", "duration_s", "render_s", "rtf", "chunks", "gpu"}
        or {"error": "..."}.

Weights: the endpoint's cached model (--model-reference, mounted under /runpod-volume/huggingface-cache)
is used when present; HIGGS_MODEL_PATH points at a local copy instead (e.g. on a network volume);
otherwise the model is downloaded at start-up (slow). The codec is baked into the image.
"""

import base64
import hashlib
import io
import os
import re
import time
import urllib.request

import numpy as np
import runpod
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = os.environ.get("HIGGS_MODEL", "multimodalart/higgs-audio-v3-tts-4b-transformers")
MODEL_REVISION = os.environ.get("HIGGS_REVISION", "main")
HOST_CACHE = "/runpod-volume/huggingface-cache/hub"
FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "mp3": ("MP3", "MPEG_LAYER_III"),
           "opus": ("OGG", "OPUS")}
MAX_TEXT_CHARS = 20000
MAX_REF_SECONDS = 30
CHUNK_GAP_S = 0.12


def resolve_model() -> str:
    if os.environ.get("HIGGS_MODEL_PATH"):
        return os.environ["HIGGS_MODEL_PATH"]
    try:  # RunPod's host-side cache for --model-reference
        return snapshot_download(MODEL_REPO, revision=MODEL_REVISION, cache_dir=HOST_CACHE, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached here: fall back to a (slow) download
        print(f"no cached copy of {MODEL_REPO} in {HOST_CACHE}; downloading", flush=True)
        return snapshot_download(MODEL_REPO, revision=MODEL_REVISION)


t0 = time.time()
MODEL_DIR = resolve_model()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16)
model = model.to("cuda").eval()
model.get_audio_codec()  # load the codec now rather than on the first request
GPU = torch.cuda.get_device_name(0)
SR = model.config.sample_rate
print(f"Higgs ready in {time.time() - t0:.1f} s on {GPU}", flush=True)

_ref_codes: dict[str, torch.Tensor] = {}  # reference clip hash -> codec codes (re-used across jobs)


def reference_codes(raw: bytes) -> torch.Tensor:
    key = hashlib.sha1(raw).hexdigest()
    if key not in _ref_codes:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)[: int(MAX_REF_SECONDS * sr)]
        with torch.inference_mode():
            _ref_codes[key] = model._encode_reference(torch.from_numpy(audio).unsqueeze(0), sr).cpu()
        if len(_ref_codes) > 16:
            _ref_codes.pop(next(iter(_ref_codes)))
    return _ref_codes[key]


def chunks(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        if cur and len(cur) + len(sentence) + 1 > limit:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    return out + ([cur] if cur else [])


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "input.text is required"}
    if len(text) > MAX_TEXT_CHARS:
        return {"error": f"input.text is longer than {MAX_TEXT_CHARS} characters"}
    fmt = (inp.get("format") or "wav").lower()
    if fmt not in FORMATS:
        return {"error": f"format must be one of {', '.join(FORMATS)}"}

    codes = None
    try:
        if inp.get("ref_audio_b64"):
            codes = reference_codes(base64.b64decode(inp["ref_audio_b64"]))
        elif inp.get("ref_audio_url"):
            with urllib.request.urlopen(inp["ref_audio_url"], timeout=30) as r:
                codes = reference_codes(r.read(50_000_000))
    except Exception as e:  # noqa: BLE001 - bad upload/URL: report, don't crash the worker
        return {"error": f"could not read the reference audio: {e}"}

    if "seed" in inp:
        torch.manual_seed(int(inp["seed"]))
    gen = {"temperature": float(inp.get("temperature", 0.7)), "top_p": float(inp.get("top_p", 0.95))}
    if inp.get("top_k"):
        gen["top_k"] = int(inp["top_k"])

    t = time.time()
    parts = []
    pieces = chunks(text, int(inp.get("max_chunk_chars", 400)))
    for piece in pieces:
        with torch.inference_mode():
            wav = model.generate_speech(piece, tok, reference_codes=codes,
                                        reference_text=(inp.get("ref_text") or None) if codes is not None else None,
                                        **gen)
        parts.append(wav.numpy())
    render = time.time() - t
    gap = np.zeros(int(CHUNK_GAP_S * SR), dtype=np.float32)
    audio = np.concatenate([x for p in parts for x in (p, gap)][:-1]) if parts else np.zeros(0, np.float32)
    if len(audio) < 0.2 * SR:
        return {"error": "the model produced no audio; try another seed or reference clip"}

    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format=container, subtype=subtype)
    duration = len(audio) / SR
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode(), "format": fmt, "sample_rate": SR,
            "duration_s": round(duration, 2), "render_s": round(render, 2), "rtf": round(render / duration, 2),
            "chunks": len(pieces), "gpu": GPU}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
