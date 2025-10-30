# toddric_chat.py — Chat engine with recipe mode, strict validation, warmup,
# SDPA (new API), greedy-fast recipes, and warning-free generation args.

from __future__ import annotations

import os
import re
import time
import json
import unicodedata
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")

# Prefer fast matmul (TF32 on Ampere+ / modern GPUs)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -----------------------------
# Recipe schema + helpers
# -----------------------------
class Ingredient(BaseModel):
    qty: Optional[str] = Field(None, description="Human-friendly, e.g., '1 cup' or '450 g'")
    item: str
    notes: Optional[str] = None

class Recipe(BaseModel):
    title: str
    servings: Optional[str] = None
    total_time: Optional[str] = None
    ingredients: List[Ingredient]
    steps: List[str]
    notes: Optional[List[str]] = None

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s.lower()).strip()

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)

def _ingredients_appear_in_steps(rec: Recipe) -> List[str]:
    steps_blob = _norm(" ".join(rec.steps))
    return [i.item for i in rec.ingredients if _norm(i.item) not in steps_blob]

def _render_recipe_md(rec: Recipe) -> str:
    out: List[str] = [rec.title]
    meta = []
    if rec.servings: meta.append(f"Serves {rec.servings}")
    if rec.total_time: meta.append(rec.total_time)
    if meta:
        out.append(" • ".join(meta))
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
        for n in rec.notes:
            out.append(f"- {n}")
    return "\n".join(out)

JSON_GUARD = (
    "You are a careful chef-bot. Output ONLY valid JSON that conforms to the schema. "
    "No markdown fences or commentary—JSON ONLY."
)

RECIPE_INSTR = """Create a complete, coherent recipe as JSON with fields:
- title (string)
- servings (string, optional)
- total_time (string, optional)
- ingredients (array of objects: qty (string|nullable), item (string), notes (string|nullable))
- steps (array of strings)
- notes (array of strings, optional)

Rules (strict):
- Match the user's request exactly: if the user specifies a dish or ingredient (e.g., "garlic beef"),
  the recipe TITLE must include those words and the INGREDIENTS must include them.
- Do NOT change the primary protein; never substitute chicken for beef, etc.
- Every food mentioned in steps MUST appear in ingredients (same wording).
- No nutrition panels, no brands, home-kitchen amounts only (metric or US).
"""

# -----------------------------
# Required-term extraction & validation
# -----------------------------
PROTEINS = {
    "beef","chicken","pork","lamb","turkey","tofu","tempeh","mushroom","fish","salmon","shrimp"
}

def _required_terms_from_request(text: str) -> List[str]:
    t = _norm(text)
    words = {w.strip(",.!?;:") for w in t.split()}
    req: List[str] = []
    req += [p for p in PROTEINS if p in words]
    for w in ("garlic","ginger","onion","pepper","rice","noodles","stir-fry","curry","tacos","soup","beef","chicken"):
        if w in t:
            req.append(w)
    seen=set(); out=[]
    for w in req:
        if w not in seen:
            out.append(w); seen.add(w)
    return out

def _validate_terms(rec: Recipe, terms: List[str]) -> None:
    title = _norm(rec.title)
    ing_blob = _norm(" ".join(i.item for i in rec.ingredients))
    for term in terms:
        if term in PROTEINS:
            if term not in title:
                raise ValueError(f"title must include protein '{term}'")
            if term not in ing_blob:
                raise ValueError(f"ingredients must include protein '{term}'")
        else:
            if term in {"garlic","ginger","onion","pepper","rice","noodles"} and term not in ing_blob:
                raise ValueError(f"ingredients must include '{term}'")
    req_proteins = [t for t in terms if t in PROTEINS]
    if req_proteins:
        banned = (PROTEINS - set(req_proteins))
        for b in banned:
            if b in title or b in ing_blob:
                raise ValueError(f"conflicting protein '{b}' present")

# -----------------------------
# Env helpers
# -----------------------------
def _resolve_model() -> str:
    for key in ("MODEL_ID", "MODEL_NAME", "TODDRIC_MODEL", "MODEL"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()
    return "toddie314/toddric-1_5b-merged-v1"

def _env_bool(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in ("0", "false", "no")

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

def _env_default(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default

@dataclass
class EngineConfig:
    model: str
    trust_remote_code: bool = True
    bits: Optional[int] = None        # 4, 8, or None
    device_map: str = "auto"
    dtype: Optional[str] = "auto"     # "auto" or "bfloat16"/"float16"
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

        dtype = (os.getenv("DTYPE") or os.getenv("TORCH_DTYPE") or "auto").strip().lower()

        return cls(
            model=model,
            trust_remote_code=trust,
            bits=bits,
            device_map="auto",
            dtype=dtype,
            max_new_tokens=_env_int("MAX_NEW_TOKENS", 256),
            temperature=_env_float("TEMPERATURE", 0.7),
            top_p=_env_float("TOP_P", 0.95),
            top_k=_env_int("TOP_K", 50),
            repetition_penalty=_env_float("REPETITION_PENALTY", 1.1),
            do_sample=_env_bool("DO_SAMPLE", True),
        )

# -------------
# Chat Engine
# -------------
class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        log.info(f"[toddric_chat] Resolving model: {cfg.model}")

        self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if self.hf_token and not os.getenv("HF_TOKEN"):
            try:
                from huggingface_hub import login, HfFolder
                HfFolder.save_token(self.hf_token)
                login(self.hf_token, add_to_git_credential=False)
                log.info("[toddric_chat] Hugging Face token saved and session logged in.")
            except Exception as e:
                log.warning(f"[toddric_chat] HF login/save token failed: {e}")

        _ = AutoConfig.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code, token=self.hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code, use_fast=True, token=self.hf_token
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ---- prompts / modes (set BEFORE warmup) ----
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "").strip()
        self.chat_template = os.getenv("CHAT_TEMPLATE") or None
        if self.chat_template and "{user}" not in self.chat_template:
            self.chat_template = None
        self.recipe_autodetect = _env_bool("RECIPE_AUTODETECT", True)
        self.recipe_max_new = int(_env_default("RECIPE_MAX_NEW", "240"))  # fast recipes

        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            low_cpu_mem_usage=True,
        )
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        # dtype shim
        resolved_dtype = None
        if cfg.dtype and cfg.dtype != "auto":
            try:
                resolved_dtype = getattr(torch, cfg.dtype)
            except Exception:
                resolved_dtype = None

        # bitsandbytes if available + CUDA
        if cfg.bits in (4, 8) and torch.cuda.is_available():
            try:
                import bitsandbytes as _  # noqa
                if cfg.bits == 4:
                    load_kwargs.update(dict(load_in_4bit=True))
                elif cfg.bits == 8:
                    load_kwargs.update(dict(load_in_8bit=True))
            except Exception as e:
                log.warning(f"[toddric_chat] bitsandbytes not available; ignoring BITS={cfg.bits} ({e})")

        # --- Choose attention backend safely ---
        attn_env = os.getenv("ATTN_IMPL")  # "flash_attention_2", "sdpa", "eager"
        if attn_env:
            load_kwargs["attn_implementation"] = attn_env
            log.info(f"[toddric_chat] attn_implementation set via env: {attn_env}")
        else:
            if torch.cuda.is_available():
                load_kwargs["attn_implementation"] = "sdpa"
                log.info("[toddric_chat] Using attn_implementation='sdpa'")

        # Try BetterTransformer fast-path (optional)
        self._bt_enabled = False
        try:
            from optimum.bettertransformer import BetterTransformer
            self._bt_transform = BetterTransformer.transform
        except Exception:
            self._bt_transform = None

        # ---- Load model (respect dtype if provided) ----
        if resolved_dtype is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=resolved_dtype, **load_kwargs)
            except TypeError as e:
                if "dtype" in str(e):
                    self.model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=resolved_dtype, **load_kwargs)
                else:
                    raise
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)

        # BetterTransformer transform (optional)
        if self._bt_transform is not None:
            try:
                self.model = self._bt_transform(self.model, keep_original_model=False)
                self._bt_enabled = True
                log.info("[toddric_chat] BetterTransformer enabled")
            except Exception as e:
                log.info(f"[toddric_chat] BetterTransformer not used: {e}")

        self.model.eval()

        # ---- generation defaults (set BEFORE warmup) ----
        # Keep base (greedy-safe) defaults separate from sampling-only knobs.
        self.gen_base = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self._pad_id(),
            eos_token_id=self._eos_id(),
            top_k=self.cfg.top_k,  # harmless for greedy, pruned when off
        )
        self.sample_defaults = dict(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )

        # Warmup thread (speeds up first responses)
        self._did_warmup = False
        if _env_bool("WARMUP", True):
            import threading
            threading.Thread(target=self._warmup, name="warmup", daemon=True).start()

        log.info(
            f"[toddric_chat] Loaded model ok. bits={cfg.bits} trust_remote_code={cfg.trust_remote_code} "
            f"dtype={cfg.dtype} max_new={cfg.max_new_tokens} temp={self.cfg.temperature} "
            f"top_p={self.cfg.top_p} top_k={self.cfg.top_k} do_sample={self.cfg.do_sample}"
        )

    # ---- warmup --------------------------------------------------------------
    def _warmup(self):
        if self._did_warmup:
            return
        try:
            tiny = self._build_prompt("Say 'ok'.")
            _ = self.generate(tiny, max_new_tokens=4, temperature=0.0, do_sample=False)
            med = self._build_prompt("List three Irish counties.")
            _ = self.generate(med, max_new_tokens=24, temperature=0.0, do_sample=False)
            self._did_warmup = True
            log.info("[toddric_chat] Warmup complete.")
        except Exception as e:
            log.warning(f"[toddric_chat] Warmup failed: {e}")

    # ---- internals -----------------------------------------------------------
    def _pad_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def _eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def _sdp_ctx(self):
        """
        Prefer the new torch.nn.attention.sdpa_kernel() API (PyTorch 2.5+).
        If unavailable, do nothing (avoid deprecated torch.backends.cuda.sdp_kernel()).
        """
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            return sdpa_kernel(
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            )
        except Exception:
            from contextlib import nullcontext
            return nullcontext()

    def _build_prompt(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        try:
            msgs: List[Dict[str, str]] = []
            #if self.system_prompt:
            #    msgs.append({"role": "system", "content": self.system_prompt})
            if history:
                msgs.extend(history[-6:])
            style = "Answer directly. No role-play. No 'User:'/'Assistant:' tags."
            msgs.append({"role": "system", "content": style})

            msgs.append({"role": "user", "content": message})
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            chat_tmpl = getattr(self, "chat_template", None)
            if chat_tmpl:
                return chat_tmpl.replace("{user}", message)
            sys = f"System: {self.system_prompt}\n" if getattr(self, "system_prompt", "") else ""
            return f"{sys}User: {message}\nAssistant:"

    # ---- recipe intent -------------------------------------------------------
    def _looks_like_recipe(self, message: str) -> bool:
        if not self.recipe_autodetect:
            return False
        m = message.strip().lower()
        if m.startswith(("recipe:", "cook:", "make:")):
            return True
        hot = ("recipe", "how do i make", "how to make", "ingredients for")
        return any(h in m for h in hot)

    def _generate_recipe_json(self, user_prompt: str) -> Recipe:
        terms = _required_terms_from_request(user_prompt)
        first = (
            f"{JSON_GUARD}\n\n{RECIPE_INSTR}\n\n"
            f"User request: {user_prompt}\n"
            f"Required terms (must be present exactly as words): {terms}\n"
            f"Return JSON:"
        )
        p1 = self._build_prompt(first)
        # greedy + capped length => fast and deterministic
        raw = self.generate(p1, temperature=0.0, max_new_tokens=self.recipe_max_new, do_sample=False)

        try:
            obj = json.loads(_extract_json(raw))
            rec = Recipe.model_validate(obj)
            _validate_terms(rec, terms)
            unused = _ingredients_appear_in_steps(rec)
            if unused:
                raise ValueError(f"Unused ingredients: {unused}")
            return rec
        except Exception as e1:
            fix = (
                f"{JSON_GUARD}\nThe previous JSON did not validate.\n"
                f"User request: {user_prompt}\n"
                f"Required terms: {terms}\n"
                f"Error: {e1}\n\nReturn corrected JSON only."
            )
            p2 = self._build_prompt(fix)
            raw2 = self.generate(p2, temperature=0.0, max_new_tokens=self.recipe_max_new, do_sample=False)
            obj2 = json.loads(_extract_json(raw2))
            rec2 = Recipe.model_validate(obj2)
            _validate_terms(rec2, terms)
            unused2 = _ingredients_appear_in_steps(rec2)
            if unused2:
                raise ValueError(f"Unused ingredients after fix: {unused2}")
            return rec2

    # ---- public generation ---------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt: str, **overrides) -> str:
        t_start = time.time()

        # Start from base defaults and apply overrides
        gen: Dict[str, Any] = dict(self.gen_base)
        for k, v in overrides.items():
            if v is not None:
                gen[k] = v

        # Decide final sampling mode
        do_sample = bool(gen.get("do_sample", False))
        temp = gen.get("temperature", None)

        # If user asked for temp<=0, force greedy
        if temp is not None and temp <= 0:
            do_sample = False

        # Compose final kwargs:
        # - if sampling: ensure temperature/top_p present (from overrides or defaults)
        # - if greedy: remove sampling-only keys so Transformers doesn't warn
        if do_sample:
            for k, v in self.sample_defaults.items():
                gen.setdefault(k, v)
            if gen.get("temperature", 1.0) <= 0:
                gen["temperature"] = 1.0
        else:
            gen["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                gen.pop(k, None)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Fresh SDPA context each call (uses new API if present; otherwise no-op)
        with self._sdp_ctx():
            outputs = self.model.generate(**inputs, **gen)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # strip echoed prompt / tags
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        for tag in ("Assistant:", "assistant:", "<|assistant|>"):
            i = text.find(tag)
            if i != -1:
                text = text[i + len(tag):].lstrip()
                break

        # perf note: tokens/sec
        try:
            out_ids = self.tokenizer(text, return_tensors=None)["input_ids"]
            in_len = int(inputs["input_ids"].shape[-1])
            gen_tokens = max(0, len(out_ids) - in_len)
            dt = time.time() - t_start
            if dt > 0:
                log.info(f"[toddric_chat] gen tokens={gen_tokens}  dt={dt:.2f}s  rate={gen_tokens/dt:.1f} tok/s")
        except Exception:
            pass

        return text.strip()

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **gen_overrides,
    ) -> Dict[str, Any]:
        t0 = time.time()

        if self._looks_like_recipe(message):
            try:
                msg = re.sub(r"^(recipe:|cook:|make:)\s*", "", message.strip(), flags=re.I)
                rec = self._generate_recipe_json(msg)
                text = _render_recipe_md(rec)
                return {
                    "text": text,
                    "used_rag": False,
                    "provenance": {"mode": "recipe", "json": rec.model_dump()},
                    "latency_ms": int((time.time() - t0) * 1000),
                    "model": self.cfg.model,
                    "session_id": session_id,
                }
            except Exception as e:
                prompt_fb = self._build_prompt(message, history=history)
                reply_fb = self.generate(prompt_fb, **gen_overrides)
                return {
                    "text": reply_fb + f"\n\n[Note: recipe struct failed: {e}]",
                    "used_rag": False,
                    "provenance": None,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "model": self.cfg.model,
                    "session_id": session_id,
                }

        prompt = self._build_prompt(message, history=history)
        reply = self.generate(prompt, **gen_overrides)
        return {
            "text": reply,
            "used_rag": False,
            "provenance": None,
            "latency_ms": int((time.time() - t0) * 1000),
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
    eng = _get_engine()
    return eng.chat(message=message, session_id=session_id)
