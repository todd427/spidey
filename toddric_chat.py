# toddric_chat.py — routed decoding (rank/howto/creative/general),
# on-topic guard, answer contract shaping, warmup, SDPA, and file-based system prompt.

from __future__ import annotations

import os, re, time, json, logging, unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

log = logging.getLogger("uvicorn.error")

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------- small helpers ----------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand
    return p

def _has_meta_tensors(model) -> bool:
    try:
        for p in model.parameters():
            if getattr(p, "device", None) and str(p.device) == "meta":
                return True
        for b in getattr(model, "buffers", lambda: [])():
            if getattr(b, "device", None) and str(b.device) == "meta":
                return True
    except Exception:
        pass
    return False

def _env_bool(name: str, default: bool=True) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() not in ("0","false","no","off")

def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, default))
    except Exception: return default

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, default))
    except Exception: return default

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()

# ---------- boundary helpers ----------
SENT_END_RE = re.compile(r'(?s)(.*?[.!?…](?:[)"\]’”]+)?)(\s+|$)')

def _trim_to_sentence(text: str) -> str:
    t = text.strip()
    m = SENT_END_RE.match(t)
    return m.group(1).strip() if m else t

def _looks_complete(text: str) -> bool:
    return bool(SENT_END_RE.match(text.strip()))

# ---------- intent routing ----------
def intent_of(q: str) -> str:
    ql = q.lower()
    if any(p in ql for p in ["best ", " top ", "which is the best", "rank ", " vs ", "compare "]):
        return "rank"
    if any(p in ql for p in ["how to", "how do i", "steps", "recipe", "ingredients", "make "]):
        return "howto"
    if any(p in ql for p in ["ideas", "brainstorm", "alternatives", "taglines", "slogans", "creative"]):
        return "creative"
    return "general"

def gen_params_for(intent: str) -> Dict[str, Any]:
    if intent in ("rank", "general"):
        return dict(do_sample=False, max_new_tokens=192)
    if intent == "howto":
        return dict(do_sample=False, max_new_tokens=320)
    # creative
    return dict(do_sample=True, temperature=0.8, top_p=0.9, top_k=50, max_new_tokens=220)

TOPIC_WORDS = [
    "champagne","sparkling","garlic","beef","windows","domain","azure","ireland",
    "pricing","security","donegal","letterkenny","python","django","huggingface"
]

def enforce_topic(user_q: str, answer: str, engine) -> str:
    lower_q = user_q.lower()
    must = [w for w in TOPIC_WORDS if w in lower_q]
    if must and not any(w in answer.lower() for w in must):
        fix = f"Answer directly about: {', '.join(must)}. Do not change the topic."
        tiny = engine._build_prompt(fix + "\n\n" + user_q)
        return _trim_to_sentence(engine.generate(tiny, do_sample=False, max_new_tokens=64))
    return answer

def enforce_contract(ans: str, kind: str) -> str:
    ans = ans.strip()
    for tag in ("User:", "Assistant:", "assistant:", "<|assistant|>"):
        if ans.startswith(tag): ans = ans[len(tag):].lstrip()
    if kind == "rank":
        lines = [ln for ln in ans.splitlines() if ln.strip()]
        if len(lines) <= 1:
            parts = re.split(r"[;•]\s*", _trim_to_sentence(ans))
            lines = [p for p in parts if p.strip()]
        lines = lines[:5]
        return "\n".join(f"- {ln.lstrip('-• ').strip()}" for ln in lines)
    return _trim_to_sentence(ans)

# ---------- recipe JSON (optional) ----------
class Ingredient(BaseModel):
    qty: Optional[str] = None
    item: str
    notes: Optional[str] = None

class Recipe(BaseModel):
    title: str
    servings: Optional[str] = None
    total_time: Optional[str] = None
    ingredients: List[Ingredient]
    steps: List[str]
    notes: Optional[List[str]] = None

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: raise ValueError("No JSON object found.")
    return m.group(0)

def _render_recipe_md(rec: Recipe) -> str:
    out = [rec.title]
    meta = []
    if rec.servings: meta.append(f"Serves {rec.servings}")
    if rec.total_time: meta.append(rec.total_time)
    if meta: out.append(" • ".join(meta))
    out.append("")
    out.append("Ingredients")
    for it in rec.ingredients:
        q = (it.qty + " ") if it.qty else ""
        n = f" ({it.notes})" if it.notes else ""
        out.append(f"- {q}{it.item}{n}")
    out.append("")
    out.append("Method")
    for i, step in enumerate(rec.steps, 1):
        out.append(f"{i}) {step}")
    if rec.notes:
        out.append("")
        out.append("Notes")
        out += [f"- {n}" for n in rec.notes]
    return "\n".join(out)

# ---------- config ----------
def _resolve_model() -> str:
    for k in ("MODEL_ID","MODEL_NAME","TODDRIC_MODEL","MODEL"):
        v = os.getenv(k)
        if v and v.strip(): return v.strip()
    return "toddie314/toddric-1_5b-merged-v1"

@dataclass
class EngineConfig:
    model: str
    trust_remote_code: bool = True
    bits: Optional[int] = None
    device_map: str = "auto"
    dtype: Optional[str] = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    @classmethod
    def from_env(cls) -> "EngineConfig":
        bits = None
        b = os.getenv("BITS") or os.getenv("TODDRIC_BITS")
        if b and b.isdigit(): bits = int(b)
        return cls(
            model=_resolve_model(),
            trust_remote_code=_env_bool("TRUST_REMOTE_CODE", True),
            bits=bits if bits in (4,8) else None,
            device_map="auto",
            dtype=os.getenv("DTYPE") or os.getenv("TORCH_DTYPE") or "auto",
            max_new_tokens=_env_int("MAX_NEW_TOKENS", 256),
            temperature=_env_float("TEMPERATURE", 0.7),
            top_p=_env_float("TOP_P", 0.95),
            top_k=_env_int("TOP_K", 50),
            repetition_penalty=_env_float("REPETITION_PENALTY", 1.1),
            do_sample=_env_bool("DO_SAMPLE", True),
        )

# ---------- engine ----------
class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        log.info(f"[toddric_chat] Resolving model: {cfg.model}")

        # system prompt: prefer file, else env
        sp_file = _resolve_path(os.getenv("SYSTEM_PROMPT_FILE","").strip())
        sys_prompt = _read_text_file(sp_file).strip() if sp_file else ""
        if not sys_prompt:
            sys_prompt = (os.getenv("SYSTEM_PROMPT","") or "").strip()
        self.system_prompt = sys_prompt

        # tokenizer/model
        _ = AutoConfig.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = dict(trust_remote_code=cfg.trust_remote_code, device_map=cfg.device_map, low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            load_kwargs["attn_implementation"] = os.getenv("ATTN_IMPL","sdpa")

        resolved_dtype = None
        if cfg.dtype and cfg.dtype != "auto":
            resolved_dtype = getattr(torch, cfg.dtype, None)

        if cfg.bits in (4,8) and torch.cuda.is_available():
            try:
                import bitsandbytes as _  # noqa
                if cfg.bits == 4: load_kwargs["load_in_4bit"]=True
                if cfg.bits == 8: load_kwargs["load_in_8bit"]=True
            except Exception as e:
                log.warning(f"[toddric_chat] bnb unavailable: {e}")

        # allow env overrides for device map / low_cpu_mem
        device_map_env = os.getenv("DEVICE_MAP", "").strip()
        if device_map_env:
            load_kwargs["device_map"] = device_map_env
        low_cpu_mem = os.getenv("LOW_CPU_MEM", None)
        if low_cpu_mem is not None:
            load_kwargs["low_cpu_mem_usage"] = str(low_cpu_mem).strip().lower() not in ("0","false","no","off")

        if resolved_dtype is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=resolved_dtype, **load_kwargs)
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=resolved_dtype, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)
        self.model.eval()

        self.gen_base = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self._pad_id(),
            eos_token_id=self._eos_id(),
            top_k=self.cfg.top_k,
        )
        self.sample_defaults = dict(temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k)

        self._did_warmup = False
        if _env_bool("WARMUP", True):
            import threading
            threading.Thread(target=self._warmup, daemon=True).start()

        log.info(f"[toddric_chat] Loaded model ok. bits={cfg.bits} dtype={cfg.dtype} max_new={cfg.max_new_tokens} do_sample={self.cfg.do_sample}")

    # --- low-level helpers
    def _pad_id(self) -> int:
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
    def _eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id
    def _sdp_ctx(self):
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)
        except Exception:
            from contextlib import nullcontext
            return nullcontext()

    def _build_prompt(self, message: str, history: Optional[List[Dict[str,str]]]=None) -> str:
        try:
            msgs = []
            if self.system_prompt:
                msgs.append({"role":"system","content":self.system_prompt})
            msgs.append({"role":"system","content":"Answer directly. No role-play. No 'User:' prefixes."})
            if history: msgs.extend(history[-6:])
            msgs.append({"role":"user","content":message})
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            sys = f"System: {self.system_prompt}\n" if self.system_prompt else ""
            return f"{sys}User: {message}\nAssistant:"

    # --- warmup (meta-safe)
    def _warmup(self):
        if self._did_warmup:
            return
        for attempt in range(5):
            if _has_meta_tensors(self.model):
                wait = 0.8 * (attempt + 1)
                log.warning(f"[toddric_chat] Warmup deferred: meta tensors (attempt {attempt+1}); sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            try:
                tiny = self._build_prompt("Say ok.")
                _ = self.generate(tiny, do_sample=False, max_new_tokens=4)
                med = self._build_prompt("List three Irish counties.")
                _ = self.generate(med, do_sample=False, max_new_tokens=24)
                self._did_warmup = True
                log.info("[toddric_chat] Warmup complete.")
                return
            except Exception as e:
                log.warning(f"[toddric_chat] Warmup failed on attempt {attempt+1}: {e}")
                time.sleep(0.6 * (attempt + 1))
        log.warning("[toddric_chat] Warmup skipped after retries.")

    # --- unified generate (CLASS-LEVEL: not inside _warmup)
    @torch.inference_mode()
    def generate(self, prompt: str, **overrides) -> str:
        t0 = time.time()
        gen = dict(self.gen_base)
        gen.update({k:v for k,v in overrides.items() if v is not None})
        do_sample = bool(gen.get("do_sample", False))
        temp = gen.get("temperature", None)
        if temp is not None and temp <= 0:
            do_sample = False

        if do_sample:
            for k,v in self.sample_defaults.items():
                gen.setdefault(k, v)
            if gen.get("temperature",1.0) <= 0:
                gen["temperature"]=1.0
        else:
            gen["do_sample"]=False
            for k in ("temperature","top_p","top_k"):
                gen.pop(k, None)

        max_new = int(gen.get("max_new_tokens", 256))
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        in_len = int(inputs["input_ids"].shape[-1])

        with self._sdp_ctx():
            out = self.model.generate(**inputs, **gen)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        for tag in ("Assistant:","assistant:","<|assistant|>"):
            i = text.find(tag)
            if i != -1:
                text = text[i+len(tag):].lstrip()
                break
        text = text.strip()

        gen_tokens = int(out.shape[-1]) - in_len
        near_cap = gen_tokens >= max_new - 2
        if near_cap and not _looks_complete(text):
            cont_inputs = self.tokenizer(prompt + text, return_tensors="pt")
            cont_inputs = {k:v.to(self.model.device) for k,v in cont_inputs.items()}
            cont_kwargs = dict(gen)
            cont_kwargs["do_sample"]=False
            cont_kwargs["max_new_tokens"]=min(24, max(8, max_new//10))
            for k in ("temperature","top_p","top_k"):
                cont_kwargs.pop(k, None)
            with self._sdp_ctx():
                out2 = self.model.generate(**cont_inputs, **cont_kwargs)
            more = self.tokenizer.decode(out2[0], skip_special_tokens=True)
            if more.startswith(prompt+text):
                more = more[len(prompt+text):].lstrip()
            text = (text + " " + more.strip()).strip()

        if not _looks_complete(text):
            text = _trim_to_sentence(text)

        try:
            toks = len(self.tokenizer(text, return_tensors=None)["input_ids"])
            dt = time.time()-t0
            if dt>0:
                log.info(f"[toddric_chat] gen≈{max(0,toks-in_len)} tok in {dt:.2f}s")
        except Exception:
            pass
        return text

    # --- public chat
    def chat(self, message: str, session_id: Optional[str]=None, history: Optional[List[Dict[str,str]]]=None, **gen_overrides) -> Dict[str,Any]:
        t0 = time.time()
        kind = intent_of(message)
        params = gen_params_for(kind)
        params.update(gen_overrides or {})
        prompt = self._build_prompt(message, history=history)
        reply = self.generate(prompt, **params)
        reply = enforce_topic(message, reply, self)
        reply = enforce_contract(reply, kind)
        return {
            "text": reply,
            "used_rag": False,
            "provenance": {"intent": kind},
            "latency_ms": int((time.time()-t0)*1000),
            "model": self.cfg.model,
            "session_id": session_id,
        }

# module-level singleton
_engine: Optional[ChatEngine] = None
def _get_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        cfg = EngineConfig.from_env()
        log.info(f"[toddric_chat] Final resolved MODEL: {cfg.model}")
        _engine = ChatEngine(cfg)
    return _engine

def chat(message: str, session_id: Optional[str]=None) -> Dict[str,Any]:
    return _get_engine().chat(message=message, session_id=session_id)
