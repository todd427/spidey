# app_toddric.py
import os, time, json
from typing import Optional, Dict, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="toddric API", version="1.1.0")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "web"

class ChatResponse(BaseModel):
    text: str
    used_rag: bool = False
    provenance: Optional[dict] = None
    latency_ms: Optional[int] = None

# -----------------------------------------------------------------------------
# Auth: TODDRIC_BEARER required unless ALLOW_NO_AUTH=1 (dev)
# Delivered to browser via HttpOnly cookie (hidden from JS).
# -----------------------------------------------------------------------------
API_TOKEN = os.getenv("TODDRIC_BEARER", "").strip()
TOKEN_COOKIE_NAME = os.getenv("TOKEN_COOKIE_NAME", "toddric_token")
TOKEN_COOKIE_SECURE = os.getenv("TOKEN_COOKIE_SECURE", "0") == "1"  # set to 1 in prod (HTTPS)
TOKEN_COOKIE_SAMESITE = os.getenv("TOKEN_COOKIE_SAMESITE", "Lax")   # Lax or Strict
ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "0") == "1"

@app.on_event("startup")
async def _auth_startup_check():
    if not API_TOKEN and not ALLOW_NO_AUTH:
        raise RuntimeError(
            "TODDRIC_BEARER is not set. Refusing to start without auth. "
            "Set ALLOW_NO_AUTH=1 to bypass (dev only)."
        )

@app.middleware("http")
async def inject_auth_cookie(request: Request, call_next):
    """
    For same-origin page/static GET/HEAD, set an HttpOnly cookie with the bearer token.
    JS cannot read it; browser includes it automatically on API calls.
    """
    response: Response = await call_next(request)
    if request.method in ("GET", "HEAD") and API_TOKEN:
        if not request.cookies.get(TOKEN_COOKIE_NAME, ""):
            response.set_cookie(
                key=TOKEN_COOKIE_NAME,
                value=API_TOKEN,
                httponly=True,
                secure=TOKEN_COOKIE_SECURE,
                samesite=TOKEN_COOKIE_SAMESITE,
                path="/",
                max_age=60 * 60 * 24 * 30,  # 30 days
            )
    return response

def bearer_auth(request: Request):
    """Authorize via Authorization header OR HttpOnly cookie."""
    if not API_TOKEN:
        return True  # dev mode
    # Header
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer ") and auth.removeprefix("Bearer ").strip() == API_TOKEN:
        return True
    # Cookie
    if (request.cookies.get(TOKEN_COOKIE_NAME) or "").strip() == API_TOKEN:
        return True
    raise HTTPException(status_code=401, detail="Missing or invalid token")

# -----------------------------------------------------------------------------
# Rate limiting (simple, single-process)
# -----------------------------------------------------------------------------
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "60"))
_rate_state: Dict[str, tuple] = {}

def rate_limiter(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    last_ts, count = _rate_state.get(ip, (0.0, 0))
    if now - last_ts > WINDOW_SEC:
        count = 0
        last_ts = now
    count += 1
    _rate_state[ip] = (last_ts, count)
    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    return True

# -----------------------------------------------------------------------------
# STOP / START / HELP semantics
# -----------------------------------------------------------------------------
OPT_OUT_WORDS = {"stop", "unsubscribe", "cancel", "quit"}
OPT_IN_WORDS  = {"start", "unstop"}
HELP_WORDS    = {"help", "info", "support"}

def normalize(s: str) -> str:
    return (s or "").strip()

def classify_command(s: str) -> Optional[str]:
    low = s.lower().strip()
    if low in OPT_OUT_WORDS: return "STOP"
    if low in OPT_IN_WORDS:  return "START"
    if low in HELP_WORDS:    return "HELP"
    return None

# -----------------------------------------------------------------------------
# Diagnostics + command router (whoami, /diag)
# -----------------------------------------------------------------------------
START_TS = time.time()
TODDRIC_USER = os.getenv("TODDRIC_USER", "Todd J. McCaffrey")
MODEL_PATH = os.getenv("TODDRIC_MODEL", "")

def _diag():
    return {
        "ok": True,
        "uptime_s": int(time.time() - START_TS),
        "rate_limit": RATE_LIMIT,
        "window_s": WINDOW_SEC,
        "model": MODEL_PATH or "(unknown)",
        "has_tc": bool(tc),
    }

def _handle_command(msg: str) -> Optional[dict]:
    m = msg.strip()
    low = m.lower()

    if low in ("whoami", "/whoami"):
        return {"text": f"You’re {TODDRIC_USER}."}

    if low in ("/diag", "diag", "/status"):
        return {"text": json.dumps(_diag(), indent=2)}

    # Add more command hooks here as needed
    return None

# -----------------------------------------------------------------------------
# toddric pipeline (import & shim)
# -----------------------------------------------------------------------------
tc = None
try:
    import toddric_chat as tc  # your shared engine module with top-level chat()
except Exception:
    tc = None

def _call_toddric(message: str, session_id: str) -> dict:
    if tc:
        for fn_name in ("chat", "generate_reply", "handle_message", "handleMessage"):
            fn = getattr(tc, fn_name, None)
            if callable(fn):
                try:
                    res = fn(message, session_id=session_id)
                except TypeError:
                    res = fn(message)
                if isinstance(res, dict):
                    return {
                        "text": str(res.get("text") or res.get("reply") or ""),
                        "used_rag": bool(res.get("used_rag", False)),
                        "provenance": res.get("provenance") or None,
                    }
                return {"text": str(res), "used_rag": False, "provenance": None}
        gen = getattr(tc, "generate", None)
        if callable(gen):
            try:
                out = gen(message, session_id=session_id)
            except TypeError:
                out = gen(message)
            return {"text": str(out), "used_rag": False, "provenance": None}

    return {"text": f"Echo: {message}", "used_rag": False, "provenance": {"source": "fallback"}}

async def _stream_tokens(message: str, session_id: str) -> AsyncGenerator[str, None]:
    if tc:
        for attr in ("stream_generate", "stream", "stream_reply", "generate_stream"):
            gen_fn = getattr(tc, attr, None)
            if callable(gen_fn):
                try:
                    iterator = gen_fn(message, session_id=session_id)
                except TypeError:
                    iterator = gen_fn(message)
                try:
                    async for token in iterator:
                        yield str(token)
                    return
                except TypeError:
                    for token in iterator:
                        yield str(token)
                    return
                except Exception:
                    break
    # Fallback: chunk final answer
    res = _call_toddric(message, session_id)
    text = res["text"]
    for i in range(0, len(text), 20):
        yield text[i:i+20]
        if i % 200 == 0:
            import asyncio; await asyncio.sleep(0)

# -----------------------------------------------------------------------------
# Routes (define BEFORE static mount)
# -----------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest,
               _auth=Depends(bearer_auth),
               _lim=Depends(rate_limiter)):
    start = time.time()
    msg = normalize(req.message)

    # SMS-like controls
    sys_cmd = classify_command(msg)
    if sys_cmd == "STOP":
        return ChatResponse(text="You’ve been opted out. Send START to opt back in.",
                            provenance={"system":"opt-out"},
                            latency_ms=int((time.time()-start)*1000))
    if sys_cmd == "START":
        return ChatResponse(text="You’re opted in. Ask me anything.",
                            provenance={"system":"opt-in"},
                            latency_ms=int((time.time()-start)*1000))
    if sys_cmd == "HELP":
        return ChatResponse(text="toddric help: HELP, STOP, START.",
                            provenance={"system":"help"},
                            latency_ms=int((time.time()-start)*1000))

    # Web commands (whoami, /diag, etc.)
    cmd_reply = _handle_command(msg)
    if cmd_reply is not None:
        return ChatResponse(text=cmd_reply["text"],
                            provenance={"system":"command"},
                            latency_ms=int((time.time()-start)*1000))

    # Model path
    result = _call_toddric(msg, req.session_id)
    latency = int((time.time() - start) * 1000)
    return ChatResponse(text=result["text"],
                        used_rag=bool(result.get("used_rag", False)),
                        provenance=result.get("provenance"),
                        latency_ms=latency)

# SSE: 2-step (POST stores, GET streams) — matches optional streaming UI flow
_pending: Dict[str, str] = {}

@app.post("/chat/stream")
async def prepare_stream(req: ChatRequest,
                         _auth=Depends(bearer_auth),
                         _lim=Depends(rate_limiter)):
    _pending[req.session_id] = normalize(req.message)
    return JSONResponse({"ok": True})

def _sse_event(data: str) -> bytes:
    return f"data: {data}\n\n".encode("utf-8")

@app.get("/chat/stream")
async def stream(session_id: str,
                 request: Request,
                 _auth=Depends(bearer_auth),
                 _lim=Depends(rate_limiter)):
    message = _pending.pop(session_id, "")
    if not message:
        raise HTTPException(status_code=400, detail="No pending message for this session_id")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        try:
            async for token in _stream_tokens(message, session_id):
                yield _sse_event(token)
            yield _sse_event("[DONE]")
        except Exception as e:
            yield _sse_event(f'{{"error":"{str(e)}"}}')

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}

# -----------------------------------------------------------------------------
# Static UI (served LAST). Put index.html in ./public
# -----------------------------------------------------------------------------
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response: Response = await call_next(request)
    path = request.url.path
    # Cache assets; keep index.html uncached for easy deploys
    if any(path.startswith(p) for p in ("/assets", "/static")) or "." in path:
        response.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
    return response

app.mount("/", StaticFiles(directory="public", html=True), name="ui")

