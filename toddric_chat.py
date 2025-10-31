# toddric_chat.py
# Lightweight chat engine for toddric-spidey with sane intent routing.
# - Default intent = "facts" (concise, direct answers; greedy decoding)
# - Other intents ("recipe", "howto", "rank", "creative") use sampling presets
# - Warmup exercises the same path as "facts" so early latencies are lower
# - Sampling knobs are only passed when do_sample=True (no more warnings)

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger("toddric_chat")
if not LOG.handlers:
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class EngineConfig:
    # Prefer MODEL_ID, fall back to MODEL_DIR, otherwise HF repo
    model: str = os.getenv("MODEL_ID", os.getenv("MODEL_DIR", "toddie314/toddric_v2_merged"))
    dtype: str = os.getenv("TORCH_DTYPE", "bfloat16")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "1") != "0"
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "160"))
    # Use the prompts/ folder by default
    system_file: str = os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_toddric.md")

    def torch_dtype(self):
        d = (self.dtype or "").lower()
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }.get(d, torch.bfloat16)


# -----------------------------
# Chat Engine
# -----------------------------
class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

        LOG.info(
            "[toddric_chat] Resolving model: %s  dtype=%s  device_map=%s",
            cfg.model, cfg.dtype, cfg.device_map
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            # Prevent generate() warnings and ensure clean decoding
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (let Accelerate decide placement; tolerate meta/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=cfg.torch_dtype(),
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )

        # System prompt (optional)
        self.system_prompt = self._load_system_prompt(cfg.system_file)

        # Generation defaults used for ALL intents (merged beneath per-intent presets)
        self.base_gen: Dict[str, Any] = dict(
            max_new_tokens=cfg.max_new_tokens,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Per-intent generation presets
        self.presets: Dict[str, Dict[str, Any]] = {
            # Concise, deterministic answers for general Q&A
            "facts": dict(do_sample=False, max_new_tokens=min(140, cfg.max_new_tokens)),
            # Explanations/steps; a touch of sampling but still tight
            "howto": dict(do_sample=True, temperature=0.4, top_p=0.9, max_new_tokens=min(220, cfg.max_new_tokens)),
            # Cook mode (lists + steps). Slight sampling for formatting, still grounded.
            "recipe": dict(do_sample=True, temperature=0.45, top_p=0.9, max_new_tokens=min(260, cfg.max_new_tokens)),
            # Ranked/short bullet outputs; slight sampling so ties break naturally
            "rank": dict(do_sample=True, temperature=0.3, top_p=0.85, max_new_tokens=min(160, cfg.max_new_tokens)),
            # Brainstormy / creative; let it wander (within reason)
            "creative": dict(do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=min(280, cfg.max_new_tokens)),
        }

        # Default routing
        self.default_intent = "facts"

        # Warmup on the EXACT path used by default answers
        try:
            _ = self.generate_text(self._wrap_facts("1+1=?"), **self.presets["facts"])
        except Exception as e:
            LOG.warning(" [toddric_chat] Warmup skipped: %s", e)
        else:
            LOG.info(" [toddric_chat] Warmup OK.")

    # ---------- public API ----------

    def chat(self, message: str, session_id: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """
        Main entry used by FastAPI. Routes to an intent, builds a prompt wrapper,
        merges gen kwargs (base + intent + overrides), and returns text + meta.
        """
        t0 = time.time()
        intent = (overrides or {}).get("intent") or self._classify_intent(message)
        LOG.info("[toddric_chat] intent=%s  msg=%r", intent, message[:160])

        text = self._handle_intent(message, intent, **overrides)
        dt = int((time.time() - t0) * 1000)

        return {
            "text": text.strip(),
            "meta": {
                "intent": intent,
                "latency_ms": dt,
                "model": self.cfg.model,
            },
        }

    # ---------- intent routing ----------

    def _handle_intent(self, msg: str, intent: str, **overrides) -> str:
        intent = intent or self.default_intent
        preset = self.presets.get(intent, self.presets[self.default_intent])

        if intent == "recipe":
            prompt = self._wrap_recipe(msg)
        elif intent == "howto":
            prompt = self._wrap_howto(msg)
        elif intent == "rank":
            prompt = self._wrap_rank(msg)
        elif intent == "creative":
            prompt = self._wrap_creative(msg)
        else:
            prompt = self._wrap_facts(msg)

        params = self._merged(preset, overrides)
        return self.generate_text(prompt, **params)

    # Extremely lightweight heuristic classifier
    def _classify_intent(self, msg: str) -> str:
        m = msg.lower()
        if any(k in m for k in ("recipe", "cook", "bake", "stew", "bread", "soda bread", "sauce")):
            return "recipe"
        if m.startswith(("how do i", "how can i")) or "steps" in m or "tutorial" in m:
            return "howto"
        if any(k in m for k in ("rank", "top ", "best ", "list ", "compare")):
            return "rank"
        if any(k in m for k in ("story", "poem", "creative", "imagine")):
            return "creative"
        return self.default_intent

    # ---------- prompt wrappers ----------

    def _with_system(self, body: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt.strip()}\n\nUser: {body.strip()}\nAssistant:"
        return f"User: {body.strip()}\nAssistant:"

    def _wrap_facts(self, q: str) -> str:
        # Short, direct 1–3 sentence answer. No rambling; no follow-ups unless asked.
        return self._with_system(
            f"Answer the question directly in 1–3 sentences, or a very short list if clearer.\n\nQuestion: {q}"
        )

    def _wrap_howto(self, q: str) -> str:
        return self._with_system(
            "Give a tight step-by-step set of instructions (3–8 steps). "
            "Use numbered steps; keep each step under ~20 words.\n\nTask: " + q
        )

    def _wrap_recipe(self, q: str) -> str:
        return self._with_system(
            "Return a compact recipe with this structure:\n"
            "Title\nServes: N | Total time: H:MM\n\nIngredients:\n- item\n- item\n\nMethod:\n1) step\n2) step\n3) step\n"
            "Keep it realistic; no hallucinated measurements. If the request is vague, pick a sensible style.\n\nRequest: " + q
        )

    def _wrap_rank(self, q: str) -> str:
        return self._with_system(
            "Return a short ranked list (3–7 bullets). One line per item. "
            "Start each line with a number and a concise label.\n\nPrompt: " + q
        )

    def _wrap_creative(self, q: str) -> str:
        return self._with_system(
            "Be vivid but economical. Keep output under ~180 words.\n\nCreative prompt: " + q
        )

    # ---------- generation ----------

    def _merged(self, preset: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
        merged = dict(self.base_gen)
        merged.update(preset or {})
        if overrides:
            merged.update(overrides)

        # If we're not sampling, strip sampling knobs so HF won't warn.
        if not merged.get("do_sample"):
            for k in ("temperature", "top_p", "top_k"):
                merged.pop(k, None)
        else:
            # Sampling guards
            if merged.get("temperature", 0) <= 0:
                merged["temperature"] = 0.2
            if "top_p" in merged and not (0 < float(merged["top_p"]) <= 1):
                merged.pop("top_p", None)

        return merged

    @torch.inference_mode()
    def generate_text(self, prompt: str, **gen_kwargs) -> str:
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Return only the assistant segment if we prefixed with "Assistant:"
        cut = text.rfind("Assistant:")
        return text[cut + len("Assistant:") :] if cut != -1 else text

    # ---------- utils ----------

    def _load_system_prompt(self, path: str) -> str:
        p = Path(path)
        if p.is_file():
            try:
                s = p.read_text(encoding="utf-8").strip()
                if s:
                    LOG.info("[toddric_chat] Loaded system prompt: %s (%d chars)", path, len(s))
                    return s
            except Exception as e:
                LOG.warning("[toddric_chat] Failed reading %s: %s", path, e)
        # Fallback minimal instruction
        return (
            "You are a concise expert. Answer the user directly. Use 1–3 sentences or a very short list "
            "when appropriate. Ask at most one clarifying question only if it is essential."
        )


# -----------------------------
# Singleton Accessor (used by FastAPI)
# -----------------------------
_engine_singleton: Optional[ChatEngine] = None

def _get_engine() -> ChatEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ChatEngine(EngineConfig())
    return _engine_singleton
