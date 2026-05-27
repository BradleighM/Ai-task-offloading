"""
latency_server.py  —  FastAPI Edge Server with Controllable Latency
-------------------------------------------------------------------
Drop-in replacement for your normal server during the demo.
You can switch latency modes WITHOUT restarting — just hit the
/set-latency endpoint or use the helper commands printed at startup.

Modes
-----
  normal  →  no added delay  (agent should OFFLOAD when CPU is high)
  slow    →  300 ms delay    (agent should stay LOCAL even if CPU is high)
  jitter  →  random 50–400 ms (tests the LSTM / Recurrent PPO)

Usage
-----
  python latency_server.py

Then in a second terminal, switch modes live:
  curl -X POST "http://localhost:8000/set-latency?mode=slow"
  curl -X POST "http://localhost:8000/set-latency?mode=normal"
  curl -X POST "http://localhost:8000/set-latency?mode=jitter"
  curl http://localhost:8000/latency-status      # check current mode
"""

import asyncio
import random
import time
import io
import psutil

from contextlib import asynccontextmanager
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("  Edge Server running on http://localhost:8000")
    print("="*60)
    print("\n  DEMO COMMANDS (run in a second terminal):\n")
    print("  # Switch to SLOW network (forces local execution)")
    print('  curl -X POST "http://localhost:8000/set-latency?mode=slow"\n')
    print("  # Switch back to NORMAL (agent offloads under high CPU)")
    print('  curl -X POST "http://localhost:8000/set-latency?mode=normal"\n')
    print("  # Random jitter (tests LSTM agent)")
    print('  curl -X POST "http://localhost:8000/set-latency?mode=jitter"\n')
    print("  # Check current mode")
    print('  curl http://localhost:8000/latency-status\n')
    print("="*60 + "\n")
    yield

app = FastAPI(title="Edge Server — Demo Mode", lifespan=lifespan)

# ── Latency state (shared, thread-safe for single-process demo) ───────────────
LATENCY_MODE = "normal"          # "normal" | "slow" | "jitter"
LATENCY_MAP  = {
    "normal": (0,   0),          # (min_ms, max_ms)
    "slow":   (300, 300),
    "jitter": (50,  400),
}


async def inject_latency():
    lo, hi = LATENCY_MAP[LATENCY_MODE]
    if lo == 0:
        return
    delay_ms = random.randint(lo, hi) if lo != hi else lo
    print(f"  [latency] mode={LATENCY_MODE}  injecting {delay_ms} ms")
    await asyncio.sleep(delay_ms / 1000)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/set-latency")
async def set_latency(mode: str = Query(..., description="normal | slow | jitter")):
    """Switch latency mode on the fly — no restart needed."""
    global LATENCY_MODE
    if mode not in LATENCY_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use: {list(LATENCY_MAP)}")
    LATENCY_MODE = mode
    lo, hi = LATENCY_MAP[mode]
    msg = f"✅  Latency mode → '{mode}'"
    if lo == hi:
        msg += f"  ({lo} ms fixed delay)" if lo else "  (no delay)"
    else:
        msg += f"  ({lo}–{hi} ms random)"
    print(msg)
    return {"mode": mode, "min_ms": lo, "max_ms": hi, "message": msg}


@app.get("/")
async def root():
    await inject_latency()
    return {
        "message": "Edge Server Running",
        "latency_mode": LATENCY_MODE
    }

@app.get("/latency-status")
async def latency_status():
    lo, hi = LATENCY_MAP[LATENCY_MODE]
    return {"current_mode": LATENCY_MODE, "min_ms": lo, "max_ms": hi}


@app.get("/ping")
async def ping():
    """Used by the client to measure RTT."""
    await inject_latency()
    return {"status": "ok", "latency_mode": LATENCY_MODE}

@app.get("/status")
def status():
    return {
        "status": "online",
        "latency_mode": LATENCY_MODE,
        "cpu_load": psutil.cpu_percent(interval=0.1)
    }

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    filter_type: str = Query("Grayscale", description="Grayscale | Blur | Edge Detection | median")
):
    """
    Receive an image, apply the chosen OpenCV filter, return the result.
    Artificial latency is injected BEFORE processing to simulate a slow network.
    """
    await inject_latency()          # ← simulated network delay

    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    t_start = time.perf_counter()

    ft_lower = filter_type.lower()
    if ft_lower in ["blur", "gaussianblur"]:
        result = cv2.GaussianBlur(img, (51, 51), 0)
    elif ft_lower in ["edges", "edge detection", "edge_detection", "edge"]:
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny  = cv2.Canny(gray, 100, 200)
        result = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    elif ft_lower in ["grayscale", "gray"]:
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif ft_lower == "median":
        result = cv2.medianBlur(img, 15)
    else:
        result = img

    proc_ms = (time.perf_counter() - t_start) * 1000

    _, encoded = cv2.imencode(".jpg", result)
    print(f"  [server] filter={filter_type}  proc={proc_ms:.1f}ms  mode={LATENCY_MODE}")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"X-Processing-Time-Ms": f"{proc_ms:.1f}",
                 "X-Latency-Mode": LATENCY_MODE}
    )


# ── Startup banner ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run("latency_server:app", host="0.0.0.0", port=8000, reload=False)