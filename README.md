# Higgs Audio v3 TTS on RunPod Serverless (voice cloning)

A scale-to-zero [RunPod Serverless](https://www.runpod.io/product/serverless) worker for
[Boson AI's Higgs Audio v3 TTS 4B](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b): text to speech in
**any voice cloned from a 3–15 s reference clip**, no fine-tuning. One request = one clip back
(wav, flac, mp3 or opus). It costs nothing while idle and about $0.002 per short line while warm.

```
image:  ghcr.io/ross-callister/runpod-higgs-tts:0.1.0   (public)
model:  multimodalart/higgs-audio-v3-tts-4b-transformers (attached as a RunPod cached model)
GPU:    any 16–24 GB card (peak 8.6 GiB VRAM)
```

## Why this worker

- **It clones well.** In a blind listening test of everyday lines (one listener, two cartoon voices, new lines
  each round), Higgs v3 scored 8–10/10 and beat IndexTTS-2, which had topped an automatic speaker-ID ranking of
  eight zero-shot engines; Echo-TTS, CosyVoice3 and Qwen3-TTS scored 0–4 by ear. Speaker-embedding similarity
  to the reference here: 0.66–0.72 (ECAPA, cartoon voice) and 0.71 (the JFK test clip below).
- **Fast cold starts for its size.** The 8.7 GB model is attached with `--model-reference`, so RunPod keeps it
  on the host (download time isn't billed) and the worker loads it in about 3 s. Packages and the 0.8 GB codec are
  baked into the image, which is built on RunPod's own `runpod/pytorch` base so hosts already have most layers.
- **Measured, not guessed** (2026-10-24):

| case | time to audio |
|---|---|
| warm worker, 5–7 s line | 9–11 s end to end (render 3.3–4.9 s) |
| after 5 min idle (FlashBoot resume) | 8.6–10.5 s (1.1 s start-up) |
| first request on a host that doesn't have the model yet | 146–185 s (mostly model caching, not billed) |
| 32 s of audio, two chunks, cloned from a URL | 24.5 s end to end (render 20.7 s) |

Real-time factor (render ÷ audio): 0.65–0.73 on RTX PRO 6000 Blackwell MIG 24 GB, 0.77–0.96 on L4.

- **No hidden traps.** Things that cost hours to find are handled: a second `--model-reference` stalls new
  workers for 30+ min (RunPod caches one model per endpoint); pip can't upgrade the base image's Debian
  `cryptography` (packages live in a venv instead); the sync `/runsync` route times out (~100 s) on a cold
  start (use `/run` + `/status`).

Alternatives I tried and why they lost: the audio.cpp GGUF build ignores the reference voice on the CPU at every
precision ([issue #471](https://github.com/0xShug0/audio.cpp/issues/471)); torch on the CPU keeps the voice but
needs ~2 min per line; int8 activation quantisation destroys the cloning; SGLang-Omni serves Higgs v3 well but
is a heavy always-on server, poor for scale-to-zero. The other community images I found targeted Higgs v2 or
had no published source.

## Deploy

Needs [`runpodctl`](https://github.com/runpod/runpodctl) ≥ 2.4, logged in.

```bash
./deploy.sh                                   # 24 GB Ampere pool, 0-1 workers, 60 s idle; prints the endpoint id
GPU="NVIDIA GeForce RTX 4090" ./deploy.sh     # or pick another GPU type / pool
```

Or by hand:

```bash
runpodctl template create --serverless --name higgs-tts \
  --image ghcr.io/ross-callister/runpod-higgs-tts:0.1.0 --container-disk-in-gb 20
runpodctl serverless create --name higgs-tts --template-id <template id> --gpu-id "NVIDIA RTX A5000" \
  --workers-min 0 --workers-max 1 --idle-timeout 60 --execution-timeout 600 \
  --model-reference "https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers:main"
```

Without `--model-reference` the worker downloads the model at start-up (minutes, billed). To keep it on a
network volume instead, set `HIGGS_MODEL_PATH` to the folder.

## Call it

```bash
export RUNPOD_API_KEY=...
python examples/client.py <endpoint id> "Hello there, darling." out.wav reference.wav "Transcript of reference.wav"
```

Raw API (use `/run` and poll `/status/<id>`; `/runsync` is fine once a worker is warm):

```json
POST https://api.runpod.ai/v2/<endpoint id>/run
{"input": {
  "text": "Text to speak. Long text is split at sentence ends.",
  "ref_audio_url": "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
  "ref_text": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "format": "mp3",
  "seed": 0
}}
```

| input | default | |
|---|---|---|
| `text` | required | up to 20,000 characters; split into ~`max_chunk_chars` (400) sentence chunks |
| `ref_audio_b64` / `ref_audio_url` | none | reference clip (wav/flac/mp3/ogg), first 30 s used; without one you get a random voice |
| `ref_text` | none | transcript of the reference; noticeably better cloning |
| `temperature`, `top_p`, `top_k` | 0.7, 0.95, off | |
| `seed` | random | |
| `format` | `wav` | `wav` (16-bit), `flac`, `mp3`, `opus` |

Output: `{"audio_b64", "format", "sample_rate": 24000, "duration_s", "render_s", "rtf", "chunks", "gpu"}`, or
`{"error": "..."}`. Encoded reference clips are cached per worker, so repeating a voice skips re-encoding.

**Getting a good clone:** use 5–10 s of clean, expressive (not monotone) speech with an accurate transcript.
The choice of reference clip mattered more than the choice of engine in our tests: pick the clip that sounds
most like the speaker's usual voice. Don't add emotion tags; they lost the voice in 4 of 4 tests.

## Cost

Billed per second while a worker runs, at RunPod's flex rate (24 GB ≈ $0.69/h, 16 GB ≈ $0.58/h, late 2026):
a warm 10 s request ≈ $0.002, plus the 60 s idle timeout. Idle endpoints cost nothing. Keeping a worker
always on (`--workers-min 1`) bills around the clock.

## Licences

The code here is MIT. The model is under the **Boson Higgs Audio v3 Research and Non-Commercial License**
(no commercial use without a licence from Boson AI); the bundled codec is under the Boson Higgs Audio 2
Community License (based on Meta Llama 3). Copies are in [`licenses/`](licenses/), and the required notices are in
[`NOTICE`](NOTICE). Built with Higgs Audio v3 licensed from Boson AI USA, Inc. Cloning a real person's voice
needs their consent.

## Files

- `handler.py`: the worker (RunPod handler)
- `Dockerfile`: image build (`docker build --platform=linux/amd64 .`)
- `deploy.sh`: template + endpoint via runpodctl
- `examples/client.py`: stdlib client
- `.runpod/`: RunPod Hub listing and tests
