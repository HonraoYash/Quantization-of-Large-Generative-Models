#!/usr/bin/env python3
import argparse, json, os, time, math, gc
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)

def tik(): return time.perf_counter()

def pretty_mem(bytes_):
    if bytes_ is None: return "n/a"
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_ < 1024: return f"{bytes_:,.1f}{unit}"
        bytes_ /= 1024
    return f"{bytes_:,.1f}PB"

def load_model(model_dir, backend, device):
    """
    backends: fp16 | bnb-int8 | bnb-nf4 | ao-int8
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    kwargs = dict(
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )

    if backend == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map="auto", **kwargs
        )
    elif backend in ("bnb-int8","bnb-nf4"):
        if backend == "bnb-int8":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_cfg, device_map="auto", **kwargs
        )
    elif backend == "ao-int8":
        # Load on CPU first to avoid big VRAM spike, then apply torchao
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map={"": "cpu"}, **kwargs
        )
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())  # weight-only INT8
        except Exception as e:
            print(f"[WARN] torchao not available or failed: {e}")
        model = model.to(device)
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model

@torch.inference_mode()
def generate(tok, model, prompt, max_new_tokens=128, temperature=0.0):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    # warmup (stable timings)
    _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    torch.cuda.reset_peak_memory_stats(model.device) if torch.cuda.is_available() else None
    t0 = tik()
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature or None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    dt = tik() - t0
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    # simple speed metric
    gen_tokens = out_ids[0].size(0) - inputs["input_ids"].size(1)
    toks_per_s = gen_tokens / dt if dt > 0 else float("inf")

    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated(model.device)

    return {
        "latency_s": dt,
        "gen_tokens": int(gen_tokens),
        "tokens_per_s": toks_per_s,
        "gpu_peak_mem": gpu_mem,
        "output": text
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local HF model dir (prefetched).")
    ap.add_argument("--out_dir", required=True, help="Where to save results.")
    ap.add_argument("--prompt", default="Explain quantization in 3 bullets.")
    ap.add_argument("--backends", nargs="+", default=["fp16","bnb-int8","bnb-nf4","ao-int8"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print(f"[info] device={device}  model_dir={args.model_dir}")
    print(f"[info] backends={args.backends}")

    for be in args.backends:
        print(f"\n=== Backend: {be} ===")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        tok, model = load_model(args.model_dir, be, device)
        r = generate(tok, model, args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)

        # Add model footprint if available
        try:
            r["model_footprint"] = pretty_mem(model.get_memory_footprint())
        except Exception:
            r["model_footprint"] = "n/a"

        results[be] = r
        # write each output for easy diff
        with open(os.path.join(args.out_dir, f"out_{be}.txt"), "w") as f:
            f.write(r["output"])
        # lightweight JSON row
        with open(os.path.join(args.out_dir, f"stats_{be}.json"), "w") as f:
            json.dump({k: (v if k != "gpu_peak_mem" else pretty_mem(v)) for k, v in r.items()}, f, indent=2)

        # Safe cleanup between backends
        del model, tok
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
    print(f"\n✔ Done. Results in: {args.out_dir}")

if __name__ == "__main__":
    main()
