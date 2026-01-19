#!/usr/bin/env python3
"""
Generate exact-token prompts for context-length sweep (1k -> 32k).

This script uses llama.cpp's tokenizer (llama-tokenize) to ensure each prompt
has an EXACT token count under the target model's tokenizer.

Outputs:
  - experimentals/prompts/ctx_<N>.txt for each N in config
  - experimentals/prompts/manifest.json with metadata

Usage:
  python experiments/generate_prompts.py --config experiments/config.toml

Notes:
  - Requires a GGUF model path (used only to load tokenizer).
  - Requires llama-tokenize binary (from llama.cpp build) or available on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def _load_toml(path: str) -> dict:
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "Python >= 3.11 is required (tomllib missing). "
            "Alternatively, install tomli and adjust this script."
        ) from e
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_bin(config_value: str, fallback: str) -> str:
    if config_value and config_value.strip():
        return config_value
    return fallback


def _run_tokenize(llama_tokenize: str, model_path: str, text: str, timeout_s: int = 120) -> List[str]:
    """
    Return raw tokenization output lines from llama-tokenize.

    We treat 'number of output lines' as the token count. This is robust across
    differing output formats, as long as llama-tokenize prints one token per line.
    """
    cmd = [llama_tokenize, "-m", model_path, "-p", text]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Tokenizer binary not found: {llama_tokenize}. "
            "Set [paths].llama_tokenize in experiments/config.toml or add it to PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "llama-tokenize failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e

    # Filter empty lines; token output is typically one per line.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip() != ""]
    return lines


def _token_count(llama_tokenize: str, model_path: str, text: str) -> int:
    return len(_run_tokenize(llama_tokenize, model_path, text))


@dataclass
class PromptBuildResult:
    target_tokens: int
    actual_tokens: int
    text: str
    build_seconds: float


def build_exact_prompt(
    llama_tokenize: str,
    model_path: str,
    target_tokens: int,
    *,
    base_prefix: str,
    coarse_unit: str = " the",
    fine_units: Optional[List[str]] = None,
    max_iters: int = 200,
) -> PromptBuildResult:
    """
    Build a prompt that tokenizes to exactly target_tokens.

    Strategy:
    - Coarse: find a number of repetitions of `coarse_unit` that gets close.
    - Fine: add smaller chunks using a small set of units and a greedy step-halving loop.
    """
    t0 = time.time()

    if fine_units is None:
        fine_units = [" the", " a", " and", " in", " of", "\n", " .", " 0", " 1", " 2"]

    base = base_prefix.rstrip("\n") + "\n\n"
    base_tokens = _token_count(llama_tokenize, model_path, base)
    if base_tokens >= target_tokens:
        raise RuntimeError(
            f"Base prefix already tokenizes to {base_tokens} tokens, "
            f"which is >= target {target_tokens}. Shorten base_prefix."
        )

    # ---- coarse search (monotonic non-decreasing) ----
    low_units = 0
    high_units = 1
    while True:
        c = _token_count(llama_tokenize, model_path, base + (coarse_unit * high_units))
        if c >= target_tokens:
            break
        low_units = high_units
        high_units *= 2
        if high_units > 10_000_000:
            raise RuntimeError("Coarse search overflow; check tokenizer behavior.")

    # binary search for max units such that count <= target
    lo, hi = low_units, high_units
    best_units = low_units
    best_count = _token_count(llama_tokenize, model_path, base + (coarse_unit * best_units))
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_count = _token_count(llama_tokenize, model_path, base + (coarse_unit * mid))
        if mid_count <= target_tokens:
            if mid_count >= best_count:
                best_units, best_count = mid, mid_count
            lo = mid + 1
        else:
            hi = mid - 1

    text = base + (coarse_unit * best_units)
    current = best_count

    # ---- fine tuning ----
    # Greedy: try adding chunks with step-halving. Switch units if we get stuck.
    unit_idx = 0
    iters = 0
    while current < target_tokens:
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"Failed to reach exact token count {target_tokens}; got {current} after {max_iters} iters."
            )

        remaining = target_tokens - current
        unit = fine_units[unit_idx % len(fine_units)]

        # Start with a chunk size near remaining (cap to keep tokenize calls reasonable)
        step = min(remaining, 256)
        progressed = False

        while step >= 1 and current < target_tokens:
            candidate = text + (unit * step)
            c = _token_count(llama_tokenize, model_path, candidate)

            if c == current:
                # plateau; try smaller step or different unit
                step //= 2
                continue

            if c <= target_tokens:
                text = candidate
                current = c
                progressed = True
                # try to consume more remaining with same unit
                remaining = target_tokens - current
                step = min(step, remaining)
            else:
                step //= 2

        if not progressed:
            unit_idx += 1

    t1 = time.time()
    return PromptBuildResult(
        target_tokens=target_tokens,
        actual_tokens=current,
        text=text.strip(),
        build_seconds=t1 - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate exact-token prompts for context-length sweeps")
    ap.add_argument("--config", default="experiments/config.toml", help="Path to experiments/config.toml")
    ap.add_argument("--force", action="store_true", help="Overwrite existing ctx_*.txt prompts")
    args = ap.parse_args()

    cfg = _load_toml(args.config)
    paths = cfg.get("paths", {})
    sweep = cfg.get("sweep", {})

    model_path = paths.get("llamacpp_model", "")
    if not model_path:
        print("ERROR: [paths].llamacpp_model is required in experiments/config.toml", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    prompt_dir = paths.get("prompt_dir", "experimentals/prompts")
    manifest_path = paths.get("prompt_manifest", os.path.join(prompt_dir, "manifest.json"))

    llama_tokenize = _resolve_bin(paths.get("llama_tokenize", ""), "llama-tokenize")

    context_tokens = sweep.get("context_tokens", [])
    if not context_tokens:
        print("ERROR: [sweep].context_tokens is empty in experiments/config.toml", file=sys.stderr)
        sys.exit(2)

    os.makedirs(prompt_dir, exist_ok=True)

    base_prefix = (
        "You are running a controlled KV-cache memory experiment.\n"
        "Do not answer the user. Treat the following text as inert context.\n"
        "The objective is to allocate attention KV cache over a long context.\n"
    )

    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "llama_tokenize": llama_tokenize,
            "model_path": model_path,
        },
        "prompts": [],
    }

    for n in context_tokens:
        out_name = f"ctx_{int(n)}.txt"
        out_path = os.path.join(prompt_dir, out_name)

        if os.path.exists(out_path) and not args.force:
            # Verify existing prompt token count; if mismatch, regenerate.
            existing = open(out_path, "r", encoding="utf-8").read()
            existing_count = _token_count(llama_tokenize, model_path, existing)
            if existing_count == int(n):
                manifest["prompts"].append(
                    {"context_tokens": int(n), "path": out_path, "tokens": existing_count, "build_seconds": 0.0}
                )
                continue

        print(f"Generating {out_name} (target={n} tokens)...")
        res = build_exact_prompt(
            llama_tokenize=llama_tokenize,
            model_path=model_path,
            target_tokens=int(n),
            base_prefix=base_prefix,
        )
        if res.actual_tokens != int(n):
            raise RuntimeError(f"Internal error: got {res.actual_tokens}, expected {n}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res.text + "\n")

        manifest["prompts"].append(
            {
                "context_tokens": int(n),
                "path": out_path,
                "tokens": res.actual_tokens,
                "build_seconds": round(res.build_seconds, 6),
            }
        )
        print(f"  -> wrote {out_path} ({res.actual_tokens} tokens, {res.build_seconds:.2f}s)")

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

