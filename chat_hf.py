#!/usr/bin/env python3
"""
chat_hf.py — Interactive CLI for local/HF models with fast SMS + bench.

New flags:
  --fast-sms                  Use greedy + 2-sentence early stop (fast path)
  --bench "PROMPT"            One-shot timing report on PROMPT (fast-sms path)
  --attn eager|sdpa           Attention impl (default eager)
  --device-map '{"":0}'       Device map (default {"":0})

Example (bf16 merged):
  python chat_hf.py --model /home/todd/training/ckpts/toddric-3b-merged-v3 --fast-sms

Example bench:
  python chat_hf.py --model /home/todd/training/ckpts/toddric-3b-merged-v3 --bench "What do you know about Jimi Hendrix?"
"""
import argparse, threading, json, os, re, time
from typing import Optional, List, Dict, Any
import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)

def build_bnb(bits: Optional[int]):
    if not bits: return None
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if bits == 8: return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError("--bits must be 4, 8, or omitted")

def _first_word(name: Optional[str], default="Assistant") -> str:
    if not name: return default
    return re.sub(r"[^A-Za-z0-9_.\-]", "", name.strip().split()[0]) or default

def main():
    ap = argparse.ArgumentParser(description="Chat with a HF model (local or hub).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, choices=[4,8], default=None)
    ap.add_argument("--device-map", default='{"":0}')
    ap.add_argument("--attn", default="eager", choices=["eager","sdpa"])
    ap.add_argument("--trust-remote-code", action="store_true", default=True)

    ap.add_argument("--persona", default=None)
    ap.add_argument("--system", default="You are a helpful assistant. Be concise.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--assistant-label", default=None)

    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)

    ap.add_argument("--fast-sms", action="store_true", help="Greedy + 2-sentence early stop.")
    ap.add_argument("--bench", default=None, help="Run fast-sms bench on this prompt and exit.")
    args = ap.parse_args()

    # persona/system
    persona = {}
    if args.persona:
        if os.path.isfile(args.persona): persona = json.load(open(args.persona,"r",encoding="utf-8"))
        else: persona = json.loads(args.persona)
    system_text = (args.system or "").strip()
    if (args.name or persona.get("name")) and not system_text.lower().startswith("your name is"):
        system_text = f"Your name is {(args.name or persona.get('name'))}. " + system_text
    asst_label = args.assistant_label or _first_word(args.name or persona.get("name"), "Assistant")

    # model
    device_map = args.device_map
    try: device_map = eval(device_map, {"__builtins__": {}})
    except Exception: pass

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    has_qconf = getattr(cfg, "quantization_config", None) is not None
    bnb = build_bnb(args.bits)
    dtype = None if (has_qconf or bnb) else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    qkwargs = {} if (has_qconf or not bnb) else {"quantization_config": bnb}

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device_map, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn, low_cpu_mem_usage=True, **qkwargs
    ).eval()
    try: mdl.config.use_cache = True
    except Exception: pass

    def _sms(prompt: str, max_new=48, max_sents=2):
        class _Stop(StoppingCriteria):
            def __init__(self, tok, n): self.tok, self.n, self.s = tok, n, ""
            def __call__(self, input_ids, scores, **kw):
                tid = int(input_ids[0, -1]); self.s += self.tok.decode([tid], skip_special_tokens=True)
                return (self.s.count(".") + self.s.count("?") + self.s.count("!")) >= self.n
        inputs = tok([prompt], return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k,v in inputs.items()}
        with torch.inference_mode():
            out = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=tok.eos_token_id, use_cache=True,
                               stopping_criteria=StoppingCriteriaList([_Stop(tok, max_sents)]))
        return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    if args.bench:
        msgs = [{"role":"system","content":"Reply in 2–3 sentences, SMS-friendly."},
                {"role":"user","content":args.bench}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.time(); out = _sms(prompt, max_new=48, max_sents=2); dt = time.time()-t0
        ntok = len(tok(out)["input_ids"])
        print(json.dumps({"new_tokens": ntok, "latency_s": round(dt,3), "tok_per_s": round(ntok/max(dt,1e-9),2), "output": out}, indent=2))
        return

    print("Type your message. Commands: /reset, /exit\n")
    history: List[Dict[str,str]] = []
    while True:
        try: user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nExiting."); break
        if not user: continue
        if user in ("/exit","/quit"): break
        if user == "/reset": history.clear(); print("🔄 history cleared."); continue

        history.append({"role":"user","content":user})
        if args.fast_sms:
            msgs = [{"role":"system","content":"Reply in 2–3 sentences, SMS-friendly."},{"role":"user","content":user}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text = _sms(prompt, max_new=48, max_sents=2)
            print(f"{asst_label}: {text}")
            history.append({"role":"assistant","content":text})
            continue

        # normal path (few-shot transcript)
        parts = [system_text, ""]
        for m in history:
            if m["role"] == "user":
                parts += [f"You: {m['content']}", f"{asst_label}:"]
            else:
                parts += [m["content"], ""]
        if not parts[-1].endswith(f"{asst_label}:"): parts.append(f"{asst_label}:")
        prompt = "\n".join(parts).rstrip()

        inputs = tok([prompt], return_tensors="pt"); inputs = {k:v.to(mdl.device) for k,v in inputs.items()}
        do_sample = args.temperature > 0
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                          eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True)
        if do_sample:
            gen_kwargs["temperature"]=args.temperature; gen_kwargs["top_p"]=args.top_p
            if args.top_k is not None: gen_kwargs["top_k"]=args.top_k
        if args.repetition_penalty and args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        print(f"{asst_label}: ", end="", flush=True)
        t = threading.Thread(target=lambda: mdl.generate(**inputs, streamer=streamer, **gen_kwargs), daemon=True)
        t.start()
        full = ""
        try:
            for piece in streamer:
                full += piece
                print(piece, end="", flush=True)
        except KeyboardInterrupt: print("\n⛔️ Interrupted.")
        print()
        history.append({"role":"assistant","content":full.strip() or "(no output)"})

if __name__ == "__main__":
    main()
