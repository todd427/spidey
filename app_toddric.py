# app_toddric.py — FastAPI app: /ui, /chat, /debug/prompt, static + transcripts logging

import os, json, time, uuid, logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import toddric_chat as tc

log = logging.getLogger("uvicorn.error")
APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
TEMPLATES_DIR = APP_ROOT / "templates"
LOG_DIR = APP_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# simple transcript writer
_start_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
_transcripts_path = LOG_DIR / f"transcripts-{_start_stamp}.jsonl"
def write_transcript(rec: Dict[str, Any]):
    try:
        with _transcripts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"[app] transcript write failed: {e}")

app = FastAPI()

# mount static if present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

def get_engine() -> tc.ChatEngine:
    return tc._get_engine()

@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    if templates and (TEMPLATES_DIR / "ui.html").exists():
        return templates.TemplateResponse("ui.html", {"request": request})
    # fallback minimal page
    html = """<!doctype html><meta charset="utf-8"><title>toddric • Web UI</title>
    <link rel="stylesheet" href="/static/ui.css?v=dev">
    <div id="app"><main id="log"></main>
      <footer>
        <textarea id="msg" rows="3" placeholder="Type a message…"></textarea>
        <button id="send">Send</button>
        <button id="reset">New chat</button>
        <span id="ready">ready</span>
        <span id="system-prompt" style="display:none"></span>
      </footer>
    </div>
    <script src="/static/ui.js?v=dev"></script>"""
    return HTMLResponse(html)

@app.post("/chat")
async def chat_endpoint(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id") or str(uuid.uuid4())
    req_id = str(uuid.uuid4())[:8]

    if not message:
        return JSONResponse({"text":"(empty message)"})

    log.info(f"[app] [{req_id}] /chat start msg={message!r} session={session_id}")
    t0 = time.time()
    try:
        eng = get_engine()
        out = await _to_thread(eng.chat, message=message, session_id=session_id)
        dt = int((time.time() - t0) * 1000)
        out["req_id"] = req_id
        write_transcript({
            "ts": time.time(), "req_id": req_id, "session_id": session_id,
            "message": message, "response": out.get("text",""), "latency_ms": out.get("latency_ms", dt),
            "model": out.get("model")
        })
        return JSONResponse(out)
    except Exception as e:
        log.exception(f"[app] [{req_id}] /chat error: {e}")
        return JSONResponse({"error": str(e), "req_id": req_id}, status_code=500)

@app.get("/debug/prompt")
def debug_prompt():
    sys = os.getenv("SYSTEM_PROMPT","") or getattr(get_engine(), "system_prompt", "")
    head = (sys[:160] + "…") if len(sys) > 160 else sys
    return JSONResponse({"system_prompt_head": head, "len": len(sys)})

@app.get("/debug/static")
def debug_static():
    return JSONResponse({
        "static_exists": STATIC_DIR.exists(),
        "static_files": sorted([p.name for p in STATIC_DIR.glob("*")]) if STATIC_DIR.exists() else [],
        "templates_exists": TEMPLATES_DIR.exists(),
        "templates_files": sorted([p.name for p in TEMPLATES_DIR.glob("*")]) if TEMPLATES_DIR.exists() else [],
    })

# util: thread off blocking HF gen
import asyncio
async def _to_thread(fn, *a, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*a, **kw))
