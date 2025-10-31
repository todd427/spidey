# spidey/toddric_chat.py
# Chat engine for toddric-spidey:
# - Router: rank / howto / recipe / creative / general
# - Stop sequences to prevent "User:" / "Assistant:" bleed
# - Recipe mode with structured retries
# - Memory-safe loader: optional 4bit/8bit, GPU budget, CPU offload, OOM retry
# - Warmup optional (WARMUP=1)

from __future__ import annotations
import os, re, time, json, logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

LOG = logging.getLogger("toddric_chat")
LOG.setLevel(logging.INFO)

# ------------------------- helpers -------------------------

def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1","true","yes","on"}

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

# ------------------------- config -------------------------

@dataclass
class EngineConfig:
    model: str = os.getenv("MODEL_ID", os.getenv("MODEL_DIR", "toddie314/toddric_v2_merged"))
    revision: Optional[str] = os.getenv("REV") or os.getenv("REVISION") or None
    attn_impl: str = os.getenv("ATTN_IMPL", "sdpa")              # sdpa|eager
    torch_dtype: Optional[str] = os.getenv("TORCH_DTYPE")        # "bfloat16"|"float16"|None
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    trust_remote_code: bool = env_bool("TRUST_REMOTE_CODE", True)
    system_prompt_file: Optional[str] = os.getenv("SYSTEM_PROMPT_FILE")
    system_prompt: Optional[str] = os.getenv("SYSTEM_PROMPT")
    max_ctx: int = env_int("MAX_CONTEXT", 4096)
    warmup: bool = env_bool("WARMUP", False)  # default off for tight VRAM

# ------------------------- stopping -------------------------

class StopOnSeq(StoppingCriteria):
    """Stop when the last tokens match any stop sequence."""
    def __init__(self, tokenizer, stops: List[str]):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[1] == 0:
            return False
        row = input_ids[0].tolist()
        for sid in self.stop_ids:
            n = len(sid)
            if n and len(row) >= n and row[-n:] == sid:
                return True
        return False

# ------------------------- engine -------------------------

class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        LOG.info("[toddric_chat] Resolving model: %s", cfg.model)

        # --- tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
        )

        # --- quantization (optional via QUANT)
        quant = (os.getenv("QUANT", "") or "").strip().lower()
        quant_config = None
        if quant in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig
                if quant == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                else:  # 8bit
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                LOG.warning("[toddric_chat] BitsAndBytes unavailable; proceeding without quant: %s", e)
                quant = ""

        # --- dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            None: "auto",
            "auto": "auto",
        }
        dtype = dtype_map.get(cfg.torch_dtype, "auto")

        # --- memory budget (helps prevent OOM)
        def _gpu_budget(fraction=0.90):
            try:
                prop = torch.cuda.get_device_properties(0)
                return {0: int(prop.total_memory * fraction), "cpu": "48GiB"}
            except Exception:
                return {"cpu": "48GiB"}

        offload_dir = os.path.join(os.getcwd(), "offload")
        os.makedirs(offload_dir, exist_ok=True)

        # --- loader with OOM fallbacks
        def _load_with(budget_fraction: float):
            return AutoModelForCausalLM.from_pretrained(
                cfg.model,
                revision=cfg.revision,
                trust_remote_code=cfg.trust_remote_code,
                low_cpu_mem_usage=True,
                torch_dtype=("auto" if quant_config else dtype),
                device_map=cfg.device_map,               # "auto"
                max_memory=_gpu_budget(budget_fraction), # cap VRAM
                offload_folder=offload_dir,
                attn_implementation=cfg.attn_impl,       # "sdpa"
                quantization_config=quant_config,
            )

        try:
            self.model = _load_with(0.90)
        except torch.cuda.OutOfMemoryError:
            LOG.warning("[toddric_chat] OOM at 90%% budget; retrying with 75%% and more CPU offload…")
            torch.cuda.empty_cache()
            self.model = _load_with(0.75)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                LOG.warning("[toddric_chat] OOM at 90%%; retrying with 75%%…")
                torch.cuda.empty_cache()
                self.model = _load_with(0.75)
            else:
                raise

        self.system_prompt = self._load_system_prompt()
        LOG.info(
            "[toddric_chat] Loaded model ok. quant=%s dtype=%s attn=%s device_map=%s",
            quant or "none", str(dtype), cfg.attn_impl, cfg.device_map
        )

        if cfg.warmup:
            self._warmup()

    # ----------------- system prompt -----------------

    def _load_system_prompt(self) -> str:
        sp = None
        if self.cfg.system_prompt_file:
            sp = read_text(self.cfg.system_prompt_file)
            if sp:
                return sp.strip()
        if self.cfg.system_prompt:
            return self.cfg.system_prompt.strip()
        # default: concise & permissive for recipes/how-to
        return (
            "You are a concise, helpful expert. When the user asks a factual or procedural question, "
            "answer directly in 1–3 sentences (or a short numbered list). When they ask for a recipe for a specific dish, "
            "produce a complete, safe recipe with a clear 'Ingredients' list and a numbered 'Method'. "
            "Avoid adding new chat role headers. Keep responses grounded and practical."
        )

    # ----------------- warmup -----------------

    def _warmup(self):
        try:
            prompts = [
                "System: You answer concisely.\nUser: Say OK.\nAssistant:",
                "User: 2+2?\nAssistant:",
            ]
            for p in prompts:
                _ = self._gen(p, max_new_tokens=6, do_sample=False, temperature=0.0, top_p=1.0)
                time.sleep(0.02)
        except Exception as e:
            LOG.warning("[toddric_chat] Warmup skipped: %s", e)

    # ----------------- routing -----------------

    _rank_re = re.compile(r"\b(best|top\s+\d+|rank|ranking|recommend|vs\.?|compare|which.*(should|is better))\b", re.I)
    _howto_re = re.compile(r"\b(how\s+to|how\s+do\s+i|steps?|method|instructions?)\b", re.I)
    _recipe_re = re.compile(r"\b(recipe|how\s+to\s+make|ingredients|bake|cook|stew|soup|bread|curry|roast|grill)\b", re.I)

    def _intent_of(self, text: str) -> str:
        t = text.strip().lower()
        if self._recipe_re.search(t):
            return "recipe"
        if self._howto_re.search(t):
            return "howto"
        if self._rank_re.search(t):
            return "rank"
        if re.search(r"\b(write|outline|poem|story|joke|pitch|blurb)\b", t):
            return "creative"
        return "general"

    # ----------------- generation core -----------------

    def _stops(self) -> StoppingCriteriaList:
        return StoppingCriteriaList([
            StopOnSeq(self.tokenizer, ["\nUser:", "\nAssistant:", "\nuser:", "\nassistant:"])
        ])

    def _gen(self,
             prompt: str,
             max_new_tokens: int = 200,
             temperature: float = 0.7,
             top_p: float = 0.9,
             do_sample: bool = True) -> str:
        """Low-level generate wrapper with stop sequences."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(1e-5, float(temperature)) if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            stopping_criteria=self._stops(),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    # ----------------- prompt builders -----------------

    def _chat_template(self, history: List[Dict[str,str]], new_user: str) -> str:
        """Simple conversation template that avoids leaking role headers mid-turn."""
        parts = [self.system_prompt.strip(), ""]
        for m in history or []:
            if m.get("role") == "user":
                parts.append(f"User: {m['content'].strip()}\nAssistant: {m.get('reply','').strip()}")
            elif m.get("role") == "assistant":
                parts.append(m["content"].strip())
        parts.append(f"User: {new_user.strip()}\nAssistant:")
        return "\n".join(parts).strip()

    def _recipe_prompt_strict(self, dish: str) -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"User: Give me a complete, safe recipe for {dish}.\n"
            "Assistant: Provide exactly this structure:\n"
            "Title\n"
            "Serves: N | Total Time: X minutes\n\n"
            "Ingredients:\n"
            "* qty item (notes)\n"
            "* qty item\n"
            "* qty item\n"
            "* qty item\n\n"
            "Method:\n"
            "1) step\n"
            "2) step\n"
            "3) step\n"
            "4) step\n"
        )

    def _rank_prompt(self, q: str) -> str:
        return f"{self.system_prompt}\n\nUser: {q}\nAssistant:"

    def _howto_prompt(self, q: str) -> str:
        return f"{self.system_prompt}\n\nUser: {q}\nAssistant:"

    # ----------------- validators -----------------

    def _looks_like_recipe(self, text: str) -> bool:
        t = text.lower()
        has_ing = "ingredients" in t
        has_method = ("method" in t) or re.search(r"\b(step|steps|directions|instructions)\b", t)
        return has_ing and has_method

    # ----------------- public API -----------------

    def chat(self, message: str, session_id: Optional[str]=None, **overrides) -> Dict[str, Any]:
        """
        Main entry: routes the question and returns {'text': reply}
        """
        q = message.strip()
        intent = self._intent_of(q)
        LOG.info("[chat] intent=%s  msg=%r", intent, q)

        # Default decoding controls (can be overridden from app)
        max_new = int(overrides.get("max_new_tokens", 180))
        temperature = float(overrides.get("temperature", 0.7))
        top_p = float(overrides.get("top_p", 0.9))

        try:
            if intent == "rank":
                prompt = self._rank_prompt(q)
                reply = self._gen(prompt, max_new_tokens=max_new, do_sample=False, temperature=0.0, top_p=1.0)
                return {"text": reply}

            if intent == "howto":
                prompt = self._howto_prompt(q)
                reply = self._gen(prompt, max_new_tokens=max(300, max_new), do_sample=False, temperature=0.0, top_p=1.0)
                return {"text": reply}

            if intent == "recipe":
                dish = q
                p1 = self._recipe_prompt_strict(dish)
                r1 = self._gen(p1, max_new_tokens=max(420, max_new), do_sample=False, temperature=0.0, top_p=1.0)
                if self._looks_like_recipe(r1):
                    return {"text": r1}

                # lightly sampled second try
                p2 = self._recipe_prompt_strict(dish) + "\nKeep it concise but complete."
                r2 = self._gen(p2, max_new_tokens=max(420, max_new), do_sample=True, temperature=0.6, top_p=0.9)
                if self._looks_like_recipe(r2):
                    return {"text": r2}

                # final fallback
                r3 = (
                    "Here is a basic, safe recipe outline:\n\n"
                    "Ingredients:\n- (list primary ingredients)\n\n"
                    "Method:\n1) Prep the ingredients.\n2) Cook main element.\n3) Combine and season.\n4) Rest and serve.\n"
                )
                return {"text": r3}

            if intent == "creative":
                prompt = self._chat_template([], q)
                reply = self._gen(prompt, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p)
                return {"text": reply}

            # general
            prompt = self._chat_template([], q)
            reply = self._gen(prompt, max_new_tokens=max_new, do_sample=False, temperature=0.0, top_p=1.0)
            return {"text": reply}

        except Exception as e:
            LOG.exception("[chat] generation error: %s", e)
            return {"text": f"[error] {e}"}

# ------------------------- convenience factory & legacy singleton -------------------------

def build_engine() -> ChatEngine:
    return ChatEngine(EngineConfig())

# Back-compat for app_toddric.py
_engine_singleton = None
def _get_engine():
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ChatEngine(EngineConfig())
    return _engine_singleton

def _reset_engine():
    global _engine_singleton
    _engine_singleton = None
