# spidey/toddric_chat.py
from __future__ import annotations
import os, re, time, json, logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

LOG = logging.getLogger("toddric_chat")
LOG.setLevel(logging.INFO)

# ------------------------------
# Engine configuration
# ------------------------------

@dataclass
class EngineConfig:
    model: str = os.getenv("MODEL_ID", os.getenv("HF_MODEL_ID", "toddie314/toddric_v2_merged"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = os.getenv("TORCH_DTYPE", "auto")  # "auto" | "bfloat16" | "float16"
    trust_remote_code: bool = True
    use_safetensors: bool = True

# Singleton cache
_engine_singleton: Optional["ChatEngine"] = None

def _get_engine() -> "ChatEngine":
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ChatEngine(EngineConfig())
    return _engine_singleton

# Simple recipe schema detector
RE_ING = re.compile(r"(?i)\bingredients?:")
RE_METHOD = re.compile(r"(?i)\b(method|directions?|steps?):")

# ------------------------------
# Chat Engine
# ------------------------------

class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        LOG.info("[toddric_chat] Loading model=%s on %s dtype=%s",
                 cfg.model, cfg.device, cfg.dtype)

        load_kwargs: Dict[str, Any] = dict(
            torch_dtype=getattr(torch, cfg.dtype) if cfg.dtype != "auto" else "auto",
            trust_remote_code=cfg.trust_remote_code,
            use_safetensors=cfg.use_safetensors,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)
        self.model.eval()

        # Default low-level gen kwargs (overridden per-intent below)
        self.base_gen = dict(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=180,
            repetition_penalty=1.02,
        )

        # Intent presets (from our discussion)
        self.presets = {
            "rank": dict(do_sample=True,  temperature=0.4, top_p=0.9,  max_new_tokens=220),
            "howto": dict(do_sample=False, temperature=0.0,            max_new_tokens=300),
            "recipe_pass1": dict(do_sample=False, temperature=0.0,     max_new_tokens=520),
            "recipe_pass2": dict(do_sample=True,  temperature=0.6, top_p=0.9, max_new_tokens=520),
            "creative": dict(do_sample=True,  temperature=0.8, top_p=0.95, max_new_tokens=360),
            "facts": dict(do_sample=False, temperature=0.0,            max_new_tokens=200),
        }

        # Tiny warmup to build CUDA graphs / lazy init
        try:
            _ = self.generate_text("Hello.", **self.presets["facts"])
        except Exception as e:
            LOG.warning("[toddric_chat] Warmup skipped: %s", e)

    # ------------- public API -------------

    def chat(self, message: str, session_id: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """Main entry: returns dict with 'text' and 'meta'."""
        t0 = time.time()
        intent = self._classify_intent(message)
        LOG.info("[toddric_chat] intent=%s", intent)

        text = self._handle_intent(message, intent, **overrides)
        dt = int((time.time() - t0) * 1000)
        return {
            "text": text.strip(),
            "meta": {"intent": intent, "latency_ms": dt}
        }

    # ------------- intent handling -------------

    def _classify_intent(self, msg: str) -> str:
        m = msg.lower().strip()
        # Recipe-ish
        if any(w in m for w in ("recipe", "how do i cook", "how to cook", "give me a recipe")):
            return "recipe"
        # Procedural how-to
        if any(w in m for w in ("how do i", "how to", "steps", "walk me through")) and "recipe" not in m:
            return "howto"
        # Ranking / best-of
        if any(w in m for w in ("best ", "top ", "rank ", "vs ", "versus ", "compare ")):
            return "rank"
        # Creative writing-ish
        if any(w in m for w in ("story", "poem", "joke", "outline", "pitch", "tagline")):
            return "creative"
        # Default factual
        return "facts"

    def _handle_intent(self, msg: str, intent: str, **overrides) -> str:
        if intent == "recipe":
            return self._recipe_two_pass(msg, **overrides)
        elif intent == "howto":
            return self.generate_text(self._wrap_howto(msg), **self._merged(self.presets["howto"], overrides))
        elif intent == "rank":
            return self.generate_text(self._wrap_rank(msg), **self._merged(self.presets["rank"], overrides))
        elif intent == "creative":
            return self.generate_text(msg, **self._merged(self.presets["creative"], overrides))
        else:  # facts
            return self.generate_text(msg, **self._merged(self.presets["facts"], overrides))

    # ------------- intent helpers -------------

    def _wrap_rank(self, msg: str) -> str:
        return (
            "Give a concise ranked answer. If multiple reasonable answers exist, list 3–5 with 1-line reasons.\n"
            f"Question: {msg}\nAnswer:\n"
        )

    def _wrap_howto(self, msg: str) -> str:
        return (
            "Answer with clear numbered steps (1., 2., 3.). Keep it practical and compact.\n"
            f"Task: {msg}\nSteps:\n"
        )

    def _recipe_two_pass(self, msg: str, **overrides) -> str:
        sys = (
            "Return a real recipe with sections:\n"
            "Title\nServes and total time\n\nIngredients:\n- item\n- item\n\nMethod:\n1) step\n2) step\n\nNotes (optional):\n- note\n"
        )
        prompt = f"{sys}\nRequest: {msg}\n\nRecipe:\n"

        # Pass 1: greedy structural
        out1 = self.generate_text(prompt, **self._merged(self.presets["recipe_pass1"], overrides))
        if RE_ING.search(out1) and RE_METHOD.search(out1):
            return out1

        # Pass 2: gentle sampled repair
        fix_prompt = (
            "The previous output missed sections (Ingredients/Method)."
            " Rewrite it as a proper recipe with those sections."
            "\n---\nOriginal request: "
            f"{msg}\nOriginal output:\n{out1}\n\nRewritten recipe:\n"
        )
        out2 = self.generate_text(fix_prompt, **self._merged(self.presets["recipe_pass2"], overrides))
        return out2

    # ------------- low-level generation -------------

    def _merged(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.base_gen, **base}
        if overrides:
            # Only allow safe overrides
            for k in ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"):
                if k in overrides:
                    merged[k] = overrides[k]
        return merged

    def generate_text(self, prompt: str, **gen_kwargs) -> str:
        # Ensure do_sample/temperature coherence
        if not gen_kwargs.get("do_sample", False):
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["top_p"] = 1.0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Heuristic: return only the continuation after the prompt if it’s clearly prefixed
        if text.startswith(prompt):
            return text[len(prompt):]
        return text

# ------------- module-level tiny façade -------------

def chat(message: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    eng = _get_engine()
    return eng.chat(message=message, session_id=session_id, **kwargs)
