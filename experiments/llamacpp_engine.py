"""
llama.cpp experiment helpers:
- Invoke llama-cli with Metal offload and KV-cache quantization flags
- Parse timings for prefill (prompt eval) and generation (eval) to compute TPS
- Sample process RSS/VMS during runtime (unified memory proxy)
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from profiling.process_sampler import ProcessSampler, filter_by_elapsed, median_rss_mb, peak_rss_mb


_RE_LOAD = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
_RE_PROMPT = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens", re.IGNORECASE)
_RE_EVAL = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens", re.IGNORECASE)
_RE_TOTAL = re.compile(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens", re.IGNORECASE)


def _bin(path_or_name: str, fallback: str) -> str:
    return path_or_name if path_or_name and path_or_name.strip() else fallback


def parse_timings(text: str) -> Dict[str, object]:
    """
    Parse llama.cpp timing lines from stderr/stdout.
    Returns best-effort timing dict (may contain None if not found).
    """
    load_ms = None
    m = _RE_LOAD.search(text)
    if m:
        load_ms = float(m.group(1))

    prompt_ms = prompt_tokens = None
    m = _RE_PROMPT.search(text)
    if m:
        prompt_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))

    eval_ms = eval_tokens = None
    m = _RE_EVAL.search(text)
    if m:
        eval_ms = float(m.group(1))
        eval_tokens = int(m.group(2))

    total_ms = total_tokens = None
    m = _RE_TOTAL.search(text)
    if m:
        total_ms = float(m.group(1))
        total_tokens = int(m.group(2))

    prompt_tps = (prompt_tokens / (prompt_ms / 1000.0)) if (prompt_ms and prompt_tokens) else None
    gen_tps = (eval_tokens / (eval_ms / 1000.0)) if (eval_ms and eval_tokens) else None

    return {
        "load_duration_s": (load_ms / 1000.0) if load_ms is not None else None,
        "prompt_eval_duration_s": (prompt_ms / 1000.0) if prompt_ms is not None else None,
        "prompt_eval_count": prompt_tokens,
        "eval_duration_s": (eval_ms / 1000.0) if eval_ms is not None else None,
        "eval_count": eval_tokens,
        "total_duration_s": (total_ms / 1000.0) if total_ms is not None else None,
        "total_count": total_tokens,
        "prefill_tps": prompt_tps,
        "gen_tps": gen_tps,
    }


def run_llama_cli(
    *,
    llama_cli: str,
    model_path: str,
    prompt: str,
    n_predict: int,
    temperature: float,
    n_ctx: int,
    ngl: int,
    cache_type_k: str,
    cache_type_v: str,
    timeout_s: float = 1800.0,
) -> Tuple[subprocess.CompletedProcess, float]:
    """
    Run llama-cli and return (CompletedProcess, elapsed_wall_s).
    """
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    cmd = [
        _bin(llama_cli, "llama-cli"),
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(int(n_predict)),
        "-c",
        str(int(n_ctx)),
        "--temp",
        str(float(temperature)),
        "-ngl",
        str(int(ngl)),
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    )
    t1 = time.time()
    return proc, (t1 - t0)


def measure_run_with_memory(
    *,
    label: str,
    llama_cli: str,
    model_path: str,
    prompt: str,
    n_predict: int,
    temperature: float,
    n_ctx: int,
    ngl: int,
    cache_type_k: str,
    cache_type_v: str,
    interval_s: float,
    baseline_window_s: float,
    timeout_s: float = 1800.0,
) -> Dict[str, object]:
    """
    Run llama-cli while sampling memory; compute baseline + peak metrics.

    Windowing (best-effort, based on llama.cpp timings):
      - baseline: median RSS over [max(0, load_s - baseline_window_s), load_s]
      - prefill: peak RSS over [load_s, load_s + prompt_eval_s]
      - total: peak RSS over [0, elapsed_wall_s]
    """
    # Start llama-cli as a subprocess so we can sample its PID while it runs.
    cmd = [
        _bin(llama_cli, "llama-cli"),
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(int(n_predict)),
        "-c",
        str(int(n_ctx)),
        "--temp",
        str(float(temperature)),
        "-ngl",
        str(int(ngl)),
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
    ]

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = ProcessSampler(pid=popen.pid, interval_s=interval_s, label=label, include_children=True)
    sampler.start()

    t0 = time.monotonic()
    try:
        stdout, stderr = popen.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        popen.kill()
        stdout, stderr = popen.communicate()
        raise RuntimeError(f"llama-cli timed out after {timeout_s}s")
    finally:
        sampler.stop()
        sampler.join(timeout=10)

    t1 = time.monotonic()
    elapsed_wall_s = t1 - t0

    if popen.returncode != 0:
        raise RuntimeError(f"llama-cli failed (exit={popen.returncode}).\nstderr:\n{stderr}\n")

    timings = parse_timings((stderr or "") + "\n" + (stdout or ""))
    load_s = float(timings["load_duration_s"]) if timings["load_duration_s"] is not None else 0.0
    prompt_s = float(timings["prompt_eval_duration_s"]) if timings["prompt_eval_duration_s"] is not None else 0.0

    baseline_start = max(0.0, load_s - baseline_window_s)
    baseline_end = load_s
    prefill_start = load_s
    prefill_end = load_s + prompt_s

    baseline_samples = filter_by_elapsed(sampler.samples, baseline_start, baseline_end)
    prefill_samples = filter_by_elapsed(sampler.samples, prefill_start, prefill_end)
    total_samples = filter_by_elapsed(sampler.samples, 0.0, elapsed_wall_s)

    metrics: Dict[str, object] = {
        "engine": "llamacpp",
        "label": label,
        "pid": popen.pid,
        "baseline_idle_rss_mb": median_rss_mb(baseline_samples),
        "peak_prefill_rss_mb": peak_rss_mb(prefill_samples),
        "peak_total_rss_mb": peak_rss_mb(total_samples),
        "baseline_window_s": (baseline_start, baseline_end),
        "prefill_window_s": (prefill_start, prefill_end),
        "request_window_s": (0.0, elapsed_wall_s),
        "elapsed_wall_s": elapsed_wall_s,
        "stdout": stdout,
        "stderr": stderr,
        "samples": sampler.samples,
    }
    metrics.update(timings)
    return metrics

