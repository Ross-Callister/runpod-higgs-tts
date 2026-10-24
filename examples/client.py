"""Minimal client (stdlib only): speak a line in a cloned voice and save the audio.

usage: RUNPOD_API_KEY=... python client.py <endpoint id> "text" out.wav [reference.wav "reference transcript"]

Uses async /run + /status: a cold start can outlast the ~100 s timeout of the sync /runsync route.
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def call(endpoint: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def speak(endpoint: str, text: str, ref: Path | None = None, ref_text: str | None = None,
          fmt: str = "wav", timeout: float = 600) -> tuple[bytes, dict]:
    job_input = {"text": text, "format": fmt}
    if ref:
        job_input |= {"ref_audio_b64": base64.b64encode(ref.read_bytes()).decode(), "ref_text": ref_text}
    job = call(endpoint, "run", {"input": job_input})
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = call(endpoint, f"status/{job['id']}")
        if st["status"] == "COMPLETED":
            out = st["output"]
            if "error" in out:
                raise RuntimeError(out["error"])
            return base64.b64decode(out.pop("audio_b64")), out | {"delay_s": st.get("delayTime", 0) / 1000}
        if st["status"] in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(f"job {st['status']}: {st.get('error')}")
        time.sleep(1.5)
    call(endpoint, f"cancel/{job['id']}", {})
    raise TimeoutError("no result in time")


if __name__ == "__main__":
    endpoint, text, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    ref = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    audio, info = speak(endpoint, text, ref, sys.argv[5] if len(sys.argv) > 5 else None, fmt=out.suffix.lstrip(".") or "wav")
    out.write_bytes(audio)
    print(json.dumps(info))
