# toddric_chat.py
# Chat engine wrapper for spidey/Toddric.
# - Env precedence for model: MODEL_NAME > TODDRIC_MODEL > MODEL > default HF id
# - Uses new `dtype=` kwarg (no deprecated torch_dtype warning)
# - Supports HF private/gated repos via HUGGINGFACE_HUB_TOKEN/HF_TOKEN
# - Optional 4/8-bit loading (requires bitsandbytes when on GPU)
# - Simple prompt formatting; configurable gen params via env

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger("uvicorn.error")


# -----------------------------
# Env helpers
# -----------------------------

def _resolve_model() -> str:
    for key in ("MODEL_NAME", "TODDRIC_MODEL", "MODEL"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()
    return "toddie314/toddric-1_5b-merged-v1"


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip() not in ("0", "false", "False", "no", "NO")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


@dataclass
class EngineConfig:
    model: str
    trust_remote_code: bool = True
    bits: Optional[int] = None         # 4, 8, or None
    device_map: str = "auto"
    dtype: Optional[str] = "auto"      # "auto" or "bfloat16"/"float16"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    @classmethod
    def from_env(cls) -> "EngineConfig":
        model = _resolve_model()
        trust = _env_bool("TRUST_REMOTE_CODE", True)

        bits_env = os.getenv("BITS") or os.getenv("TODDRIC_BITS")
        bits = None
        if bits_env and bits_env.strip().isdigit():
            b = int(bits_env)
            if b in (4, 8):
                bits = b

        # dtype: prefer DTYPE, fallback to TORCH_DTYPE, default "auto"
        dtype = (os.getenv("DTYPE") or os.getenv("TORCH_DTYPE") or "auto").strip()

        max_new = _env_int("MAX_NEW_TOKENS", 256)
        temp = _env_float("TEMPERATURE", 0.7)
        top_p = _env_float("TOP_P", 0.95)
        top_k = _env_int("TOP_K", 50)
        rep = _env_float("REPETITION_PENALTY", 1.1)
        do_sample = _env_bool("DO_SAMPLE", True)

        return cls(
            model=model,
            trust_remote_code=trust,
            bits=bits,
            device_map="auto",
            dtype=dtype,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=rep,
            do_sample=do_sample,
        )


# -------------
# Chat Engine
# -------------

class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        log.info(f"[toddric_chat] Resolving model: {cfg.model}")

        # Token for private/gated repos
        self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        # Fetch config/tokenizer with token
        acfg = AutoConfig.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
            token=self.hf_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=True,
            token=self.hf_token,
        )

        # Prepare load kwargs
        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            low_cpu_mem_usage=True,
        )

        # dtype (new name replacing torch_dtype in Transformers)
        if cfg.dtype == "auto":
            load_kwargs["dtype"] = "auto"
        elif cfg.dtype:
            try:
                load_kwargs["dtype"] = getattr(torch, cfg.dtype)
            except Exception:
                load_kwargs["dtype"] = "auto"

        # Optional 4/8-bit loading
        if cfg.bits in (4, 8):
            if cfg.bits == 4:
                load_kwargs.update(dict(load_in_4bit=True))
            elif cfg.bits == 8:
                load_kwargs.update(dict(load_in_8bit=True))

        # Pass token explicitly if present
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)
        self.model.eval()

        # Generation defaults
        self.gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self._pad_id(),
            eos_token_id=self._eos_id(),
        )

        # Prompt template
        self.chat_template = os.getenv("CHAT_TEMPLATE") or None
        if self.chat_template and "{user}" not in self.chat_template:
            self.chat_template = None

        log.info(
            f"[toddric_chat] Loaded model ok. bits={cfg.bits} trust_remote_code={cfg.trust_remote_code} "
            f"dtype={cfg.dtype} max_new={cfg.max_new_tokens} temp={cfg.temperature} "
            f"top_p={cfg.top_p} top_k={cfg.top_k} do_sample={cfg.do_sample}"
        )

    def _pad_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def _eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def _format_prompt(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        if self.chat_template:
            return self.chat_template.replace("{user}", message)
        if history:
            convo = ""
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                convo += f"{role}: {content}\n"
            convo += f"user: {message}\nassistant:"
            return convo
        return f"User: {message}\nAssistant:"

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text.strip()


# ----------------
# Public interface
# ----------------

_engine: Optional[ChatEngine] = None


def _get_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        cfg = EngineConfig.from_env()
        log.info(f"[toddric_chat] Final resolved MODEL: {cfg.model}")
        _engine = ChatEngine(cfg)
    return _engine


def chat(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    eng = _get_engine()
    prompt = eng._format_prompt(message)
    reply = eng.generate(prompt)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "text": reply,
        "used_rag": False,
        "provenance": None,
        "latency_ms": latency_ms,
        "model": eng.cfg.model,
        "session_id": session_id,
    }
