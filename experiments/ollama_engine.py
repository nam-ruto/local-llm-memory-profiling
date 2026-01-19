"""
Ollama experiment helpers:
- Start/stop `ollama serve` with KV-cache environment settings
- Run a deterministic /api/generate request and extract timing metrics

We treat process RSS (unified memory) as the VRAM proxy on Apple Silicon.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests

from profiling.process_sampler import ProcessSampler, median_rss_mb, peak_rss_mb, filter_by_elapsed


def _healthcheck(base_url: str, timeout_s: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_ready(base_url: str, timeout_s: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _healthcheck(base_url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Ollama not ready at {base_url} after {timeout_s:.1f}s")


@dataclass
class OllamaServer:
    base_url: str
    process: subprocess.Popen

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def stop(self, timeout_s: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout_s)


def start_server(
    *,
    base_url: str,
    kv_cache_type: str,
    flash_attention: bool,
    log_path: Optional[str] = None,
) -> OllamaServer:
    """
    Start `ollama serve` in a subprocess with KV-cache configuration.

    Important:
      - KV cache type is global; requires server restart to take effect.
      - If another Ollama server is already running on base_url, this will likely fail.
    """
    env = os.environ.copy()
    if flash_attention:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = kv_cache_type

    stdout = stderr = subprocess.DEVNULL
    f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        f = open(log_path, "w", encoding="utf-8")
        stdout = f
        stderr = f

    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    except FileNotFoundError as e:
        if f:
            f.close()
        raise RuntimeError("`ollama` binary not found on PATH") from e

    try:
        wait_for_ready(base_url, timeout_s=30.0)
    except Exception:
        # ensure we cleanup if startup failed
        try:
            proc.terminate()
        except Exception:
            pass
        if f:
            f.close()
        raise

    return OllamaServer(base_url=base_url, process=proc)


def generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_gen_tokens: int,
    temperature: float,
    timeout_s: float = 300.0,
) -> Dict[str, object]:
    """
    Run a single non-streaming Ollama generate request and return raw fields.
    """
    api_url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": int(max_gen_tokens),
            "temperature": float(temperature),
        },
    }

    t0 = time.time()
    r = requests.post(api_url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    t1 = time.time()

    data = r.json()
    # durations are in nanoseconds in Ollama responses
    out = {
        "model": model,
        "elapsed_wall_s": t1 - t0,
        "total_duration_s": float(data.get("total_duration", 0)) / 1e9,
        "prompt_eval_count": int(data.get("prompt_eval_count", 0)),
        "prompt_eval_duration_s": float(data.get("prompt_eval_duration", 0)) / 1e9,
        "eval_count": int(data.get("eval_count", 0)),
        "eval_duration_s": float(data.get("eval_duration", 0)) / 1e9,
        "response_text": data.get("response", ""),
    }

    # Derived TPS (avoid div-by-zero)
    pe_d = out["prompt_eval_duration_s"]
    ev_d = out["eval_duration_s"]
    out["prefill_tps"] = (out["prompt_eval_count"] / pe_d) if pe_d > 0 else None
    out["gen_tps"] = (out["eval_count"] / ev_d) if ev_d > 0 else None
    return out


def measure_generate_with_memory(
    *,
    server: OllamaServer,
    label: str,
    model: str,
    prompt: str,
    max_gen_tokens: int,
    temperature: float,
    interval_s: float,
    baseline_window_s: float,
    warmup: bool,
    warmup_prompt: str,
    warmup_max_gen_tokens: int,
) -> Dict[str, object]:
    """
    Measure one Ollama request with memory sampling.

    Returns a dict containing:
      - baseline_idle_rss_mb
      - peak_prefill_rss_mb
      - peak_total_rss_mb
      - prefill_tps / gen_tps
      - raw durations/counts
      - samples (ProcessSampler.samples) for writing trace CSV
      - window boundaries (elapsed_s) for baseline/prefill/total
    """
    # Optional warmup to ensure model is loaded.
    if warmup:
        _ = generate(
            base_url=server.base_url,
            model=model,
            prompt=warmup_prompt,
            max_gen_tokens=warmup_max_gen_tokens,
            temperature=temperature,
        )

    # Baseline sampling window (engine idle).
    baseline_sampler = ProcessSampler(
        pid=server.pid,
        interval_s=interval_s,
        label=label,
        include_children=True,
    )
    baseline_sampler.start()
    time.sleep(baseline_window_s)
    baseline_sampler.stop()
    baseline_sampler.join(timeout=5)
    baseline_value = median_rss_mb(baseline_sampler.samples)

    # Sample for the duration of a single request.
    sampler = ProcessSampler(
        pid=server.pid,
        interval_s=interval_s,
        label=label,
        include_children=True,
    )
    sampler.start()
    t_req0 = time.monotonic()

    result = generate(
        base_url=server.base_url,
        model=model,
        prompt=prompt,
        max_gen_tokens=max_gen_tokens,
        temperature=temperature,
    )

    t_req1 = time.monotonic()
    sampler.stop()
    sampler.join(timeout=10)

    t0 = sampler.start_monotonic or t_req0
    req_start = t_req0 - t0
    req_end = t_req1 - t0

    prefill_end = req_start + float(result.get("prompt_eval_duration_s") or 0.0)

    # Extract peaks from the request samples.
    prefill_samples = filter_by_elapsed(sampler.samples, req_start, prefill_end)
    total_samples = filter_by_elapsed(sampler.samples, req_start, req_end)

    metrics: Dict[str, object] = {
        "engine": "ollama",
        "label": label,
        "pid": server.pid,
        "baseline_idle_rss_mb": baseline_value,
        "peak_prefill_rss_mb": peak_rss_mb(prefill_samples),
        "peak_total_rss_mb": peak_rss_mb(total_samples),
        "baseline_window_s": baseline_window_s,
        "request_window_s": (req_start, req_end),
        "prefill_window_s": (req_start, prefill_end),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "prompt_eval_duration_s": result.get("prompt_eval_duration_s"),
        "eval_count": result.get("eval_count"),
        "eval_duration_s": result.get("eval_duration_s"),
        "total_duration_s": result.get("total_duration_s"),
        "elapsed_wall_s": result.get("elapsed_wall_s"),
        "prefill_tps": result.get("prefill_tps"),
        "gen_tps": result.get("gen_tps"),
        "samples": sampler.samples,
        "baseline_samples": baseline_sampler.samples,
    }
    return metrics

