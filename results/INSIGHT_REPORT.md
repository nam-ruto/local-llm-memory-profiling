# KV-Cache Memory Behavior: Insight Report

**Experiment:** Empirical study of KV-cache memory and throughput for on-device LLM inference (Llama 3.2 3B) on macOS Apple Silicon.  
**Runtimes:** llama.cpp (llama-completion) and Ollama.  
**Data:** Process RSS as unified-memory proxy; prefill/gen TPS from engine timings.

---

## 1. Setup

| Item | Value |
|------|--------|
| Model | Llama 3.2 3B Instruct (GGUF) |
| KV-cache types | f16, q8_0, q4_0 |
| Context lengths | 1,024 → 32,768 tokens (6 points) |
| Max gen tokens | 256, temperature 0 |
| Platform | Apple Silicon (M4), unified memory |

**Result runs used:**

- **llama.cpp:** `results/20260217_165615/` — full sweep (18 runs).
- **Ollama:** `results/20260217_205443/` — full sweep (18 runs). Server started by the suite with `OLLAMA_KV_CACHE_TYPE` and default `OLLAMA_CONTEXT_LENGTH=4096`.

---

## 2. Findings

### 2.1 llama.cpp: Memory scales with context; quantization reduces RSS

- **Peak total RSS** increases with context length for all KV types, as expected (larger KV cache + activations).
- **At 32k context:** f16 ≈ **5,817 MB**, q8_0 ≈ **4,111 MB**, q4_0 ≈ **3,201 MB** — strong reduction with quantization.
- **At 1k context:** f16 ≈ 2,213 MB, q8_0 ≈ 2,134 MB, q4_0 ≈ 2,093 MB (smaller spread; KV cache is a smaller share of total).
- **Takeaway:** For long-context workloads, q4_0 or q8_0 KV cache substantially lowers peak memory (e.g. ~45% less at 32k for q4_0 vs f16).

### 2.2 llama.cpp: Throughput (TPS)

- **Prefill TPS** decreases as context length grows (more tokens per request, similar hardware). For f16: ~498 at 1k → ~149 at 32k.
- **Gen TPS** (tokens/s during decode) is high when reported together with prefill in the same timing window for some runs; where reported separately, gen TPS is in the ~40–108 range depending on context and KV type.
- Quantized KV (q8_0, q4_0) can trade some prefill speed for lower memory; exact tradeoff depends on context length and batch.

### 2.3 Ollama: Valid RSS only when suite starts the server

- When the suite starts its own `ollama serve` (no other Ollama on port 11434), process RSS is sampled correctly: **baseline and peak RSS are non-zero** (e.g. ~2.3–2.7 GB for the runs in `20260217_205443`).
- If another Ollama instance is already running, the suite may connect to it and sample a different process; then RSS can be reported as 0. **For comparable Ollama memory data, ensure no other Ollama is running so the suite starts the server.**

### 2.4 Ollama: Flat RSS due to 4096 context cap

- In `20260217_205443`, the Ollama server was started with **default `OLLAMA_CONTEXT_LENGTH=4096`** (see server logs). The backend allocates a fixed **4,096-token KV cache** regardless of requested context length.
- For 8k, 16k, and 32k prompts, the server **truncates** to 4,096 tokens (logs: `"truncating input prompt" limit=4096`). So **prompt_eval_count** is 4,096 for those runs, not 8k/16k/32k.
- **Peak RSS is therefore flat** (~2.3–2.7 GB) across all context lengths: it reflects “model + 4096-token KV cache,” not true 8k/16k/32k usage.
- **Implication:** To observe Ollama memory scaling with context (and compare fairly to llama.cpp), the suite or environment must set **`OLLAMA_CONTEXT_LENGTH`** to at least the maximum context used (e.g. 32768) when starting the server.

### 2.5 Ollama vs llama.cpp (where comparable)

- **At effective 4096 context**, Ollama peak total RSS is in the same ballpark as llama.cpp at 4096: e.g. Ollama f16 ~2,683 MB vs llama.cpp f16 ~2,562 MB; Ollama q4_0 ~2,356 MB vs llama.cpp q4_0 ~2,200 MB. Small differences can be due to runtime overhead, allocation strategy, and measurement timing.
- **Throughput:** Ollama prefill TPS (when reported) is in a similar range to llama.cpp for the same effective prompt length (e.g. ~420–480 prefill TPS at 4k). Gen TPS is also in a comparable band (~40 tokens/s).

---

## 3. Limitations

- **Single machine, single model:** Results are for one Apple Silicon machine and one 3B model; scaling and absolute numbers will vary with hardware and model size.
- **Ollama context cap:** Without raising `OLLAMA_CONTEXT_LENGTH`, Ollama memory does not scale with 8k/16k/32k in this dataset; only throughput and behavior at 4096 are comparable.
- **RSS as proxy:** Process RSS is a proxy for “VRAM” use on unified memory; it does not separate GPU vs CPU residency.

---

## 4. Recommendations

1. **Long-context, memory-constrained:** Prefer **q4_0** (or q8_0) KV cache in both llama.cpp and Ollama to cut peak memory significantly at 16k–32k context.
2. **Ollama context sweep:** For a fair context-length vs memory comparison with llama.cpp, set `OLLAMA_CONTEXT_LENGTH` (e.g. to 32768) when starting the Ollama server and re-run the sweep.
3. **Reproducibility:** Run with no other Ollama process bound to the same port so the suite starts the server and RSS is attributed to the correct process.

---

## 5. Data and figures

- **Runs:** `results/20260217_165615/runs.csv` (llama.cpp), `results/20260217_205443/runs.csv` (Ollama).
- **Traces:** `results/<timestamp>/traces/*.csv` — per-run RSS time series.
- **Logs:** `results/<timestamp>/logs/` — server and stderr logs (include `OLLAMA_CONTEXT_LENGTH` and truncation messages for Ollama).
- **Figures:** Combine runs and plot with  
  `python experiments/visualize_runs.py --runs results/20260217_165615 results/20260217_205443 --out results/figures`  
  to get `peak_rss_mb.png`, `prefill_tps.png`, `gen_tps.png`.
