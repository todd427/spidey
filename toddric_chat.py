# toddric_chat.py
# Chat engine wrapper for spidey/Toddric.
# - Prefers chat() with real chat formatting; generate() also works.
# - Model env precedence: MODEL_ID > MODEL_NAME > TODDRIC_MODEL > MODEL > default HF id
# - HF auth: uses HUGGINGFACE_HUB_TOKEN / HF_TOKEN; only "login" if HF_TOKEN isn't already set
# - dtype compatibility: prefers dtype=... (new), falls back to torch_dtype=... (old); if auto, passes nothing
# - Optional 4/8-bit quant (requires CUDA + bitsandbytes; otherwise ignored)
# - System prompt pulled from SYSTEM_PROMPT (optional)

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
    # Be generous about env names we accept
    for key in ("MODEL_ID", "MODEL_NAME", "TODDRIC_MODEL", "MODEL"):
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
        dtype = (os.getenv("DTYPE") or os.getenv("TORCH_DTYPE") or "auto").strip().lower()

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

        # Make token globally visible for hub utilities ONLY if HF_TOKEN isn't already set.
        if self.hf_token and not os.getenv("HF_TOKEN"):
            try:
                from huggingface_hub import login, HfFolder
                HfFolder.save_token(self.hf_token)
                login(self.hf_token, add_to_git_credential=False)
                log.info("[toddric_chat] Hugging Face token saved and session logged in.")
            except Exception as e:
                log.warning(f"[toddric_chat] HF login/save token failed: {e}")

        # Fetch config/tokenizer with token
        _ = AutoConfig.from_pretrained(
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

        # Be explicit about padding to avoid odd left/right truncation and echoes
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # System prompt (optional)
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "").strip()

        # Prepare load kwargs
        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            low_cpu_mem_usage=True,
        )

        # ---------- dtype compatibility shim ----------
        resolved_dtype = None
        if cfg.dtype and cfg.dtype != "auto":
            try:
                resolved_dtype = getattr(torch, cfg.dtype)
            except Exception:
                resolved_dtype = None
        # ---------- end dtype shim ----------

        # Optional 4/8-bit loading (only when CUDA & bitsandbytes available)
        if cfg.bits in (4, 8) and torch.cuda.is_available():
            try:
                import bitsandbytes as _  # noqa: F401
                if cfg.bits == 4:
                    load_kwargs.update(dict(load_in_4bit=True))
                elif cfg.bits == 8:
                    load_kwargs.update(dict(load_in_8bit=True))
            except Exception as e:
                log.warning(f"[toddric_chat] bitsandbytes not available; ignoring BITS={cfg.bits} ({e})")

        # Pass token explicitly if present
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        # Load model with dtype shim handling
        if resolved_dtype is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model, dtype=resolved_dtype, **load_kwargs
                )
            except TypeError as e:
                if "dtype" in str(e):
                    # Older transformers: retry with torch_dtype
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model, torch_dtype=resolved_dtype, **load_kwargs
                    )
                else:
                    raise
        else:
            # No explicit dtype requested; let transformers decide
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

        # Optional plain string template fallback (rarely needed now)
        self.chat_template = os.getenv("CHAT_TEMPLATE") or None
        if self.chat_template and "{user}" not in self.chat_template:
            self.chat_template = None

        log.info(
            f"[toddric_chat] Loaded model ok. bits={cfg.bits} trust_remote_code={cfg.trust_remote_code} "
            f"dtype={cfg.dtype} max_new={cfg.max_new_tokens} temp={cfg.temperature} "
            f"top_p={cfg.top_p} top_k={cfg.top_k} do_sample={cfg.do_sample}"
        )

    # ---- internals ----

    def _pad_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def _eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def _build_prompt(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Prefer real chat templates if tokenizer supports them; else use a robust string fallback.
        """
        # Try native chat template
        try:
            msgs: List[Dict[str, str]] = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            if history:
                msgs.extend(history[-6:])
            msgs.append({"role": "user", "content": message})
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # Fallback path
            if self.chat_template:
                return self.chat_template.replace("{user}", message)
            sys = f"System: {self.system_prompt}\n" if self.system_prompt else ""
            return f"{sys}User: {message}\nAssistant:"

    # ---- public generation methods ----

    @torch.inference_mode()
    def generate(self, prompt: str, **overrides) -> str:
        """
        Text generation on a prepared prompt. Supports override of gen kwargs.
        """
        gen = dict(self.gen_kwargs)
        # allow adapter to pass decoding knobs through
        for k, v in overrides.items():
            if v is not None:
                gen[k] = v

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **gen)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip echoed prompt (handles both raw and "User:/Assistant:" forms)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        for tag in ("Assistant:", "assistant:", "<|assistant|>"):
            i = text.find(tag)
            if i != -1:
                text = text[i + len(tag):].lstrip()
                break
        return text.strip()

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **gen_overrides,
    ) -> Dict[str, Any]:
        """
        Primary entry point expected by the FastAPI app.
        """
        t0 = time.time()
        prompt = self._build_prompt(message, history=history)
        reply = self.generate(prompt, **gen_overrides)
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "text": reply,
            "used_rag": False,
            "provenance": None,
            "latency_ms": latency_ms,
            "model": self.cfg.model,
            "session_id": session_id,
        }


# ----------------
# Module-level API
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
    # Back-compat shim if something imports chat() directly
    eng = _get_engine()
    return eng.chat(message=message, session_id=session_id)

