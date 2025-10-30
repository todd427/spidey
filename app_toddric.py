#!/usr/bin/env python3
# app_toddric.py — Spidey/Toddric server with templated UI, adaptive caps,
# logging that never goes silent, and JSONL transcripts.

from __future__ import annotations

import os
import json
import time
import uuid
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import multiprocessing as mp

# ---------------- Runtime hygiene (before heavy imports) ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------- Web stack ----------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# ---------------- Engine ----------------
import toddric_chat as tc

# =============================================================================
# Paths / App
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

APP_VERSION = os.getenv("APP_VERSION", "dev")
RELOAD = bool(int(os.getenv("RELOAD", "0")))
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "60"))  # seconds

# ----- Logging that won't be muted by Uvicorn -----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
APP_LOG_FILE = os.getenv("APP_LOG_FILE", "logs/app.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

alog = logging.getLogger("app")
alog.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not alog.handlers:  # avoid duplicates on reload
    # Write app logs to file only (not stdout/stderr)
    Path(APP_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    _h = TimedRotatingFileHandler(
        APP_LOG_FILE, when="midnight", backupCount=7, encoding="utf-8"
    )
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    alog.addHandler(_h)
    alog.propagate = False  # keep independent of Uvicorn dictConfig

ulog = logging.getLogger("uvicorn.error")  # available if you want it

# ----- JSONL transcript writer -----
LOG_TRANSCRIPTS = os.getenv("LOG_TRANSCRIPTS", "1").lower() not in ("0", "false", "no")
LOG_DIR = os.getenv("LOG_DIR", "logs")

class TranscriptWriter:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = self.log_dir / f"transcripts-{stamp}.jsonl"
        self._lock = threading.Lock()
    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

_transcripts = TranscriptWriter(LOG_DIR) if LOG_TRANSCRIPTS else None

# ----- App + static/templates -----
app = FastAPI(title="Spidey / Toddric", version="2.1")

# (Removed print()s to avoid stdout noise)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# =============================================================================
# Lazy engine getter
# =============================================================================
_engine: Optional[tc.ChatEngine] = None

def get_engine() -> tc.ChatEngine:
    global _engine
    if _engine is None:
        cfg = tc.EngineConfig.from_env()
        ulog.info(f"[toddric] MODEL_ID={cfg.model}  REV=main")
        _engine = tc.ChatEngine(cfg)
        head = (os.getenv("SYSTEM_PROMPT", "") or _engine.system_prompt or "").strip()[:80]
        if head:
            ulog.info(f"[toddric] SYSTEM head='{head}'")
    return _engine

# =============================================================================
# Routes
# =============================================================================
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(
        "ui.html",
        {"request": request, "APP_VERSION": APP_VERSION},
    )

@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """
    Body:  { "message": str, "session_id": str|null, ...gen_kwargs }
    """
    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "empty message", "req_id": req_id}, status_code=400)

    session_id = payload.get("session_id")
    gen_overrides = {k: v for k, v in payload.items() if k not in {"message", "session_id"}}

    # Adaptive token cap for quick Q&A
    q = message.lower()
    looks_long = q.startswith(("recipe:", "cook:", "make:")) or any(
        k in q for k in ("essay", "story", "code", "step-by-step", "explain in detail", "slides:")
    )
    if not looks_long:
        gen_overrides.setdefault("max_new_tokens", 96)
    gen_overrides.setdefault("temperature", 0.2)

    alog.info(f"[{req_id}] /chat start msg='{message[:80]}{'…' if len(message)>80 else ''}' "
              f"session={session_id} overrides={gen_overrides}")

    eng = get_engine()

    async def _do_chat():
        # thread off so we can await timeout even if CPU-bound
        return await asyncio.to_thread(eng.chat, message=message, session_id=session_id, **gen_overrides)

    try:
        out = await asyncio.wait_for(_do_chat(), timeout=CHAT_TIMEOUT)
    except asyncio.TimeoutError:
        dur = int((time.time() - t0) * 1000)
        alog.error(f"[{req_id}] TIMEOUT after {dur}ms")
        if _transcripts:
            _transcripts.write({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "req_id": req_id,
                "session_id": session_id,
                "message": message,
                "overrides": gen_overrides,
                "error": f"timeout after {CHAT_TIMEOUT}s",
                "latency_ms": dur,
                "model": getattr(eng.cfg, "model", None) if eng else None,
            })
        return JSONResponse(
            {"error": f"chat timed out after {CHAT_TIMEOUT}s", "req_id": req_id, "latency_ms": dur},
            status_code=504,
        )
    except Exception as e:
        dur = int((time.time() - t0) * 1000)
        alog.exception(f"[{req_id}] ERROR after {dur}ms: {e}")
        if _transcripts:
            _transcripts.write({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "req_id": req_id,
                "session_id": session_id,
                "message": message,
                "overrides": gen_overrides,
                "error": str(e),
                "latency_ms": dur,
                "model": getattr(eng.cfg, "model", None) if eng else None,
            })
        return JSONResponse({"error": str(e), "req_id": req_id, "latency_ms": dur}, status_code=500)

    # success
    out.setdefault("latency_ms", int((time.time() - t0) * 1000))
    out.setdefault("model", eng.cfg.model)
    out["req_id"] = req_id

    if _transcripts:
        try:
            _transcripts.write({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "req_id": req_id,
                "session_id": session_id,
                "message": message,
                "overrides": gen_overrides,
                "response": out.get("text", ""),
                "latency_ms": out["latency_ms"],
                "model": out["model"],
            })
        except Exception as e:
            alog.warning(f"[{req_id}] transcript write failed: {e}")

    alog.info(f"[{req_id}] OK {out['latency_ms']}ms")
    return JSONResponse(out)

from starlette.responses import JSONResponse

@app.get("/debug/prompt")
def debug_prompt():
    sys = os.getenv("SYSTEM_PROMPT", "") or getattr(get_engine(), "system_prompt", "")
    head = (sys[:160] + "…") if len(sys) > 160 else sys
    return JSONResponse({"system_prompt_head": head, "len": len(sys)})

@app.get("/debug/static")
def debug_static():
    return JSONResponse(
        {
            "static_exists": STATIC_DIR.exists(),
            "static_files": sorted([p.name for p in STATIC_DIR.glob("*")]),
            "templates_exists": TEMPLATES_DIR.exists(),
            "templates_files": sorted([p.name for p in TEMPLATES_DIR.glob("*")]),
        }
    )

@app.get("/debug/ping")
def debug_ping():
    return {"ok": True, "time": time.time()}

@app.get("/debug/engine")
def debug_engine():
    eng = get_engine()
    return {
        "model": eng.cfg.model,
        "bits": eng.cfg.bits,
        "dtype": eng.cfg.dtype,
        "gen_defaults": eng.gen_kwargs,
        "autodetect_recipe": getattr(eng, "recipe_autodetect", None),
        "warmed": getattr(eng, "_did_warmup", False),
    }

@app.post("/debug/warmup")
def debug_warmup():
    eng = get_engine()
    getattr(eng, "_warmup", lambda: None)()
    return {"ok": True, "warmed": getattr(eng, "_did_warmup", False)}

@app.get("/debug/logtest")
def logtest():
    logging.getLogger("uvicorn.error").info("uvicorn.error INFO alive")
    alog.info("app INFO alive")
    return {"ok": True}

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app_toddric:app",
        host=host,
        port=port,
        reload=RELOAD,
        reload_dirs=[str(BASE_DIR)],
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
