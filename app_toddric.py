# app_toddric.py
import os, time, json, pathlib, datetime, threading, uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse, PlainTextResponse, HTMLResponse, RedirectResponse
)
from pydantic import BaseModel

try:
    from transformers import AutoConfig
except Exception:
    AutoConfig = None

# ──────────────────────────────────────────────────────────────────────────────
# Config / system prompt

def _load_system_prompt() -> str:
    env_p = os.getenv("SYSTEM_PROMPT", "").strip()
    if env_p:
        return env_p
    path = os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_toddric.md")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
    except Exception:
        pass
    return (
        "Always prefix replies with [TODDRIC]. "
        "You are Toddric — pragmatic, nerdy, playful, and wise. Speak plainly, avoid fluff. "
        "Push back gently on falsehoods; explain the correction. Prefer 3–6 tight sentences. "
        "When the user asks a question, answer it directly in 1–3 sentences. "
        "Do not respond with a list of questions. Ask at most one clarifying question only if the request is ambiguous. "
        "If uncertain, say 'Not sure.' Label speculation."
    )

@dataclass
class Cfg:
    MODEL_ID: str = os.getenv("MODEL_ID", "toddie314/toddric-llama-8B-merged-v1")
    MODEL_DIR: Optional[str] = os.getenv("MODEL_DIR") or None
    MODEL_REV: str = os.getenv("MODEL_REV", "main")
    TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "auto")
    SYSTEM_PROMPT: str = _load_system_prompt()
    # decoding defaults (conservative MVP)
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "64"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    DO_SAMPLE: bool = os.getenv("DO_SAMPLE", "0") not in ("0", "false", "False")
    REP_PEN: float = float(os.getenv("REPETITION_PENALTY", "1.15"))

_cfg = Cfg()
print(f"[toddric] MODEL_ID={_cfg.MODEL_ID}  MODEL_DIR={_cfg.MODEL_DIR}  REV={_cfg.MODEL_REV}")
print(f"[toddric] SYSTEM head={_cfg.SYSTEM_PROMPT[:96]!r}")

# ──────────────────────────────────────────────────────────────────────────────
# Auth

API_TOKEN = os.getenv("TODDRIC_BEARER", "").strip()
ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "1") in ("1", "true", "True")  # default open for local dev

def _require_bearer(request: Request):
    if ALLOW_NO_AUTH:
        return True
    if not API_TOKEN:
        raise RuntimeError("TODDRIC_BEARER is not set. Set ALLOW_NO_AUTH=1 to bypass (dev only).")
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if h.split(" ", 1)[1].strip() != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Transcript logging (JSONL with rotation)

_LOG_ON = os.getenv("LOG_TRANSCRIPTS", "0") not in ("0", "false", "False")
_LOG_DIR = os.getenv("LOG_DIR", "./logs")
_LOG_ROTATE_MB = int(os.getenv("LOG_ROTATE_MB", "50"))
_log_lock = threading.Lock()
_log_fp = None
_log_path = None

def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_log_file():
    global _log_fp, _log_path
    if not _LOG_ON:
        return None
    os.makedirs(_LOG_DIR, exist_ok=True)
    if _log_fp:
        try:
            if os.path.getsize(_log_path) >= (_LOG_ROTATE_MB * 1024 * 1024):
                _log_fp.close()
                _log_fp = None
        except FileNotFoundError:
            _log_fp = None
    if _log_fp is None:
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        _log_path = os.path.join(_LOG_DIR, f"transcripts-{stamp}.jsonl")
        _log_fp = open(_log_path, "a", encoding="utf-8")
    return _log_fp

def log_event(event: dict):
    if not _LOG_ON:
        return
    with _log_lock:
        fp = _ensure_log_file()
        if fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            fp.flush()

def log_session_start(session_id: str, system_prompt: str, cfg: Cfg):
    if not _LOG_ON:
        return
    log_event({
        "type": "session_start",
        "time": _now_iso(),
        "session_id": session_id,
        "system_head": (system_prompt or "")[:400],
        "model_id": cfg.MODEL_ID,
        "model_dir": cfg.MODEL_DIR,
        "decode_defaults": {
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
            "temperature": cfg.TEMPERATURE,
            "top_p": cfg.TOP_P,
            "top_k": cfg.TOP_K,
            "do_sample": cfg.DO_SAMPLE,
            "repetition_penalty": cfg.REP_PEN,
        },
        "server_pid": os.getpid(),
        "instance": str(uuid.uuid4()),
    })

def log_turn(session_id: str, user_text: str, assistant_text: str, meta: dict):
    if not _LOG_ON:
        return
    log_event({
        "type": "turn",
        "time": _now_iso(),
        "session_id": session_id,
        "user": user_text,
        "assistant": assistant_text,
        "meta": meta,
    })

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app

app = FastAPI(title="Spidey / Toddric", version="1.9")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("Cache-Control", "private, max-age=5")
    return resp

# ──────────────────────────────────────────────────────────────────────────────
# Chat engine compatibility cfg

def _compat_cfg_for_chatengine():
    class Compat:
        def __init__(self, preset: dict): self.__dict__.update(preset)
        def __getattr__(self, name):
            defaults = {
                "device": "auto", "device_map": "auto",
                "torch_dtype": "auto", "dtype": self.__dict__.get("torch_dtype", "auto"),
                "bits": None, "load_in_4bit": False, "load_in_8bit": False,
                "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "trust_remote_code": True,
                "max_new_tokens": _cfg.MAX_NEW_TOKENS, "temperature": _cfg.TEMPERATURE,
                "top_p": _cfg.TOP_P, "top_k": _cfg.TOP_K, "do_sample": _cfg.DO_SAMPLE,
            }
            return defaults.get(name, None)

    load4 = os.getenv("LOAD_IN_4BIT", "0") not in ("0", "false", "False")
    load8 = os.getenv("LOAD_IN_8BIT", "0") not in ("0", "false", "False")
    bits = 4 if load4 else (8 if load8 else None)

    preset = {
        "model_dir": _cfg.MODEL_DIR,
        "model": _cfg.MODEL_DIR or _cfg.MODEL_ID,
        "revision": _cfg.MODEL_REV,
        "system_prompt": _cfg.SYSTEM_PROMPT,
        "device": _cfg.TORCH_DEVICE,
        "device_map": os.getenv("DEVICE_MAP", "auto"),
        "torch_dtype": os.getenv("TORCH_DTYPE", "auto"),
        "dtype": os.getenv("DTYPE", os.getenv("TORCH_DTYPE", "auto")),
        "load_in_4bit": load4, "load_in_8bit": load8, "bits": bits,
        "trust_remote_code": True,
    }
    return Compat(preset)

# ──────────────────────────────────────────────────────────────────────────────
# Engine adapter (no persona injection here; engines handle it)

def _engine_adapter(obj):
    class Adapter:
        def __init__(self, inner):
            self.inner = inner
            self._has_stream = hasattr(inner, "stream") and callable(getattr(inner, "stream"))

        def _call_with_fallbacks(self, fn, message, session_id, kwargs):
            try:
                return fn(message, session_id=session_id, **kwargs)
            except TypeError:
                try:
                    return fn(message, session_id=session_id)
                except TypeError:
                    return fn(message)

        def chat(self, message: str, session_id: str = "default",
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 do_sample: Optional[bool] = None,
                 repetition_penalty: Optional[float] = None):
            kwargs = {}
            if max_new_tokens is not None: kwargs["max_new_tokens"] = max_new_tokens
            if temperature    is not None: kwargs["temperature"]    = temperature
            if top_p          is not None: kwargs["top_p"]          = top_p
            if top_k          is not None: kwargs["top_k"]          = top_k
            if do_sample      is not None: kwargs["do_sample"]      = do_sample
            if repetition_penalty is not None: kwargs["repetition_penalty"] = repetition_penalty

            o = self.inner
            if hasattr(o, "chat") and callable(getattr(o, "chat")):
                res = self._call_with_fallbacks(o.chat, message, session_id, kwargs)
            elif hasattr(o, "generate") and callable(getattr(o, "generate")):
                res = self._call_with_fallbacks(o.generate, message, session_id, kwargs)
            elif callable(o):
                res = self._call_with_fallbacks(o, message, session_id, kwargs)
            else:
                raise AttributeError("ChatEngine object has no usable interface (.chat/.generate/callable)")

            if isinstance(res, str):
                return {"text": res, "used_rag": False}
            if isinstance(res, dict):
                txt = res.get("text") or res.get("reply") or ""
                return {**res, "text": str(txt)}
            txt = getattr(res, "text", None)
            return {"text": str(txt if txt is not None else res), "used_rag": False}

        def stream(self, message: str, session_id: str = "default", **gen_kwargs):
            if self._has_stream:
                try:
                    return self.inner.stream(message, session_id=session_id, **gen_kwargs)
                except TypeError:
                    pass
            yield self.chat(message, session_id=session_id, **gen_kwargs)["text"]

    return Adapter(obj)

# ──────────────────────────────────────────────────────────────────────────────
# Engine builder (toddric_chat preferred → Minimal fallback)

_engine = None

def _build_minimal_engine():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = _cfg.MODEL_DIR or _cfg.MODEL_ID
    print(f"[toddric] MinimalEngine loading: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lm  = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() and _cfg.TORCH_DEVICE != "cpu" else "cpu"
    dtype  = torch.bfloat16 if (device == "cuda") else torch.float32
    lm.to(device=device, dtype=dtype).eval()

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    system = _cfg.SYSTEM_PROMPT

    class MinimalEngine:
        def chat(self, message: str, session_id: str = "default",
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 do_sample: Optional[bool] = None,
                 repetition_penalty: Optional[float] = None):
            # Resolve decoding params
            M = max_new_tokens if max_new_tokens is not None else _cfg.MAX_NEW_TOKENS
            T = temperature    if temperature    is not None else _cfg.TEMPERATURE
            P = top_p          if top_p          is not None else _cfg.TOP_P
            K = top_k          if top_k          is not None else _cfg.TOP_K
            S = do_sample      if do_sample      is not None else _cfg.DO_SAMPLE
            R = repetition_penalty if repetition_penalty is not None else _cfg.REP_PEN

            # Build prompt (chat template if available)
            try:
                msgs = [{"role": "system", "content": system},
                        {"role": "user",   "content": message}]
                ids = tok.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    return_tensors="pt", padding=True
                )
            except Exception:
                prompt = f"{system}\n\nUser: {message}\nAssistant:"
                ids = tok(prompt, return_tensors="pt").input_ids

            attn = (ids != tok.pad_token_id).long()
            ids, attn = ids.to(device), attn.to(device)

            # Generate (eos stop + deterministic by default)
            with torch.no_grad():
                out = lm.generate(
                    ids,
                    attention_mask=attn,
                    max_new_tokens=M,
                    do_sample=S,
                    temperature=T,
                    top_p=P,
                    top_k=K,
                    repetition_penalty=R,
                    eos_token_id=tok.eos_token_id,
                )

            text = tok.decode(out[0], skip_special_tokens=True)

            # Tidy: remove echoed system or role tags
            sys_head = system.strip()[:120]
            if sys_head and text.strip().startswith(sys_head):
                text = text.split("\n", 1)[-1].strip()
            for tag in ("<|assistant|>", "Assistant:", "assistant\n"):
                if tag in text:
                    text = text.split(tag)[-1].strip()

            # Questionnaire clamp: if it devolves into a list of questions, keep only the first sentence
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if sum(ln.endswith("?") for ln in lines) >= 3:
                first = lines[0]
                for end in (". ", "! ", "? "):
                    if end in first:
                        first = first.split(end, 1)[0] + end.strip()
                        break
                text = first

            # Hard brevity cap (MVP)
            if len(text) > 360:
                text = text[:360].rstrip() + "…"

            return {"text": text, "used_rag": False}

        def stream(self, message: str, session_id: str = "default", **_):
            yield self.chat(message, session_id=session_id)["text"]

    print("[toddric] Using MinimalEngine (built-in)")
    return MinimalEngine()

def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    try:
        from toddric_chat import ChatEngine as _ChatEngine  # type: ignore
        compat = _compat_cfg_for_chatengine()
        raw = _ChatEngine(compat)
        _engine = _engine_adapter(raw)
        print("[toddric] Using ChatEngine from toddric_chat (adapter)")
        return _engine
    except Exception as e:
        print(f"[toddric] ChatEngine not available/compatible: {e!r}")
    _engine = _engine_adapter(_build_minimal_engine())
    return _engine

# ──────────────────────────────────────────────────────────────────────────────
# Schemas

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "web"
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None

class ChatResponse(BaseModel):
    text: str
    used_rag: bool = False
    provenance: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    truncated: Optional[bool] = None

# ──────────────────────────────────────────────────────────────────────────────
# Routes

@app.get("/", response_class=RedirectResponse)
def root_redirect():
    return RedirectResponse(url="/ui")

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html><meta charset="utf-8"><title>toddric • Web UI</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0c10;--panel:#12151d;--panel2:#1b1f2a;--txt:#e8eaf0;--muted:#9aa3b2;--acc:#3b65ff;--bd:#2a2e39}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:17px/1.55 system-ui,Segoe UI,Roboto,Ubuntu}
header{display:flex;align-items:center;gap:8px;padding:12px 16px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:rgba(11,12,16,.9);backdrop-filter:saturate(120%) blur(6px)}
header .dot{width:8px;height:8px;border-radius:50%;background:#26d07c;box-shadow:0 0 0 2px #153,0 0 10px #26d07c}
header h1{font-size:16px;margin:0 6px 0 0;font-weight:600}
header .flag{margin-left:auto;color:#8fc;opacity:.9}
main{max-width:980px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;height:70vh}
.msg{display:flex}
.msg .bubble{max-width:78ch;padding:12px 14px;border-radius:14px;border:1px solid var(--bd);white-space:pre-wrap;font-size:16px;line-height:1.55}
.msg.me{justify-content:flex-end}
.msg.me .bubble{background:var(--panel2)}
.msg.bot .bubble{background:var(--panel)}
.system{color:var(--muted);font-size:13px}
footer{position:sticky;bottom:0;background:rgba(11,12,16,.95);border-top:1px solid var(--bd)}
form{display:flex;gap:10px;padding:12px;max-width:980px;margin:0 auto;align-items:flex-end}
textarea,button{border-radius:12px;border:1px solid var(--bd);background:#0b0e14;color:var(--txt)}
textarea{flex:1;resize:none;height:110px;padding:12px 14px;font-size:17px;line-height:1.5}
button{padding:12px 16px;min-width:96px;background:var(--acc);border-color:#365df0;color:#fff;font-weight:600}
.meta{color:var(--muted);font-size:12px;margin-top:4px}
kbd{background:#111;border:1px solid #333;border-bottom-color:#222;color:#ddd;border-radius:6px;padding:1px 6px;font-size:.85em}
#reset{background:#2e7d32;border-color:#246428}
</style>
<header>
  <div class=dot></div><h1>toddric • Web UI</h1><span class=meta id=ready>ready</span>
  <button id=reset type=button style="margin-left:8px">New chat</button>
  <span class="flag meta">🇮🇪 cloudflared-ready</span>
</header>
<main id=log></main>
<footer>
  <form id=f>
    <textarea id=msg placeholder="Type a message…  (Enter=send,  Shift+Enter=newline)"></textarea>
    <button id=send type=submit>Send</button>
  </form>
  <div class=meta style="max-width:980px;margin:0 auto 10px auto;padding:0 12px">
    Tip: press <kbd>Enter</kbd> to send, <kbd>Shift</kbd>+<kbd>Enter</kbd> for newline.
  </div>
</footer>
<script>
const $=s=>document.querySelector(s), log=$("#log"), f=$("#f"), T=$("#msg"), resetBtn=$("#reset");
function scrollToBottom(){ log.scrollTo({ top: log.scrollHeight, behavior: "smooth" }); }
function add(role, text){
  const d=document.createElement("div"); d.className="msg "+role;
  const b=document.createElement("div"); b.className="bubble"; b.textContent=text;
  d.appendChild(b); log.appendChild(d); scrollToBottom();
}
function sys(text){ const d=document.createElement("div"); d.className="system"; d.textContent=text; log.appendChild(d); scrollToBottom(); }

// rotate session id for true fresh starts
let session = crypto.randomUUID();
resetBtn.addEventListener("click", ()=>{ session = crypto.randomUUID(); log.innerHTML=""; sys("new chat started"); });

async function send(ev){ ev&&ev.preventDefault(); const txt=T.value.trim(); if(!txt) return; T.value=""; add("me", txt);
  const res=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:txt,session_id:session})});
  if(!res.ok){ add("bot","[error "+res.status+"] "+await res.text()); return; }
  const data=await res.json(); let out=data.text||""; if(data.truncated){ out+=" …"; } add("bot", out);
}
T.addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){ send(e) }});
f.addEventListener("submit",send);

(async()=>{ try{ const r=await fetch("/debug/prompt"); if(r.ok){ const j=await r.json(); sys("system prompt: "+(j.system_prompt_head||"")+(j.len>240?" …":"")); } }catch{} })();
</script>
"""

@app.get("/ui-lite", response_class=HTMLResponse)
def ui_lite():
    return "<p>Use <a href='/ui'>/ui</a>.</p>"

# ──────────────────────────────────────────────────────────────────────────────
# Chat

class Req(BaseModel):
    message: str
    session_id: Optional[str] = "web"
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None

class Resp(BaseModel):
    text: str
    used_rag: bool = False
    provenance: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    truncated: Optional[bool] = None

@app.post("/chat", response_model=Resp)
def chat(req: Req, _auth=Depends(_require_bearer)):
    start = time.time()

    # Log session start once
    if not hasattr(app.state, "seen_sessions"):
        app.state.seen_sessions = set()
    if req.session_id not in app.state.seen_sessions:
        app.state.seen_sessions.add(req.session_id)
        log_session_start(req.session_id or "web", _cfg.SYSTEM_PROMPT, _cfg)

    eng = _get_engine()
    try:
        res = eng.chat(
            req.message, session_id=req.session_id,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
            do_sample=req.do_sample,
            repetition_penalty=_cfg.REP_PEN,
        )
        if isinstance(res, dict):
            text = str(res.get("text") or res.get("reply") or "")
            used_rag = bool(res.get("used_rag", False))
            provenance = res.get("provenance")
        else:
            text, used_rag, provenance = str(res), False, None

        truncated = len(text) > 0 and text[-1] not in ".!?”’\""
        latency = int((time.time() - start) * 1000)

        log_turn(req.session_id or "web", req.message, text, {
            "latency_ms": latency,
            "used_rag": used_rag,
            "truncated": truncated,
            "engine": "toddric_chat-or-minimal",
            "model_resolved": _cfg.MODEL_DIR or _cfg.MODEL_ID,
        })

        return Resp(text=text, used_rag=used_rag, provenance=provenance,
                    latency_ms=latency, truncated=truncated)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"error": "chat_failed", "detail": repr(e)})

# ──────────────────────────────────────────────────────────────────────────────
# Debug

@app.get("/debug/model")
def debug_model():
    mid = _cfg.MODEL_DIR or _cfg.MODEL_ID
    out: Dict[str, Any] = {"env_MODEL_ID": _cfg.MODEL_ID, "env_MODEL_DIR": _cfg.MODEL_DIR, "resolved": mid}
    try:
        if AutoConfig is not None:
            conf = AutoConfig.from_pretrained(mid, trust_remote_code=True)
            out.update({
                "model_type": getattr(conf, "model_type", None),
                "architectures": getattr(conf, "architectures", None),
                "hidden_size": getattr(conf, "hidden_size", None),
                "vocab_size": getattr(conf, "vocab_size", None),
            })
    except Exception as e:
        out["config_error"] = repr(e)
    try:
        p = pathlib.Path(mid)
        out["source"] = "local_dir" if p.exists() else "hub"
        if p.exists():
            out["files_present"] = sorted([q.name for q in p.glob("*")][:12])
    except Exception:
        pass
    return out

@app.get("/debug/prompt")
def debug_prompt():
    s = _cfg.SYSTEM_PROMPT or ""
    return {"system_prompt_head": s[:240], "len": len(s)}

