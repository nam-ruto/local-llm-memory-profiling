# Memory Profiling Framework for On-Device LLM Runtimes

An Empirical Study of KV-Cache Memory Behavior in On-Device LLM Runtimes: This repo is a lightweight and simple memory profiling framework for measuring memory usage during LLM inference across different runtimes on macOS (Apple Silicon).

## Overview

This repo is currently focused on empirically studying **KV-cache memory behavior** for **local inference engines**:
- **Ollama** (API-driven, server process)
- **llama.cpp** (raw CLI)

**WebLLM is temporarily out of scope** for this refactor.

We treat **process RSS (unified memory)** as the practical “VRAM utilization” proxy on Apple Silicon.

## KV-cache experiment suite

This is the main workflow for the study:
- **KV cache types**: `f16`, `q8_0`, `q4_0`
- **Context sweep**: 1k → 32k **exact tokens** (generated with `llama-tokenize`)
- **Metrics**: baseline idle RSS, peak RSS during prefill, peak RSS overall, prefill TPS, gen TPS

### 1) Configure paths + sweep

Edit `experiments/config.toml`:
- Set `[paths].llamacpp_model` to your **llama3.2:3b GGUF** path
- Set `[paths].llama_cli` and `[paths].llama_tokenize` (or ensure they’re on PATH)
- Set `[ollama].model` to your Ollama tag for llama3.2:3b (and pull it)

### 2) Generate exact-token prompts

```bash
uv sync
source .venv/bin/activate
python experiments/generate_prompts.py --config experiments/config.toml
```

This writes:
- `inputs/prompts/ctx_<N>.txt` for each context length
- `inputs/prompts/manifest.json`

### 3) Run the suite

**Important (Ollama):** `experiments/run_suite.py` starts its own `ollama serve` and restarts it per KV-cache type.
If you already have Ollama running as a background service, stop it first so the port isn’t in use.
```bash
python experiments/run_suite.py --config experiments/config.toml
```

Outputs:
- `results/<timestamp>/runs.csv` (one row per run)
- `results/<timestamp>/traces/*.csv` (memory time-series per run)
- `results/<timestamp>/logs/*` (stderr / server logs)

## Installation

### Prerequisites

- Python 3.8+
- macOS (Apple Silicon)
- For Ollama: [Ollama installed](https://ollama.ai)
- For llama.cpp: [llama.cpp built](https://github.com/ggml-org/llama.cpp) with Metal support (`LLAMA_METAL=1`), plus `llama-cli` and `llama-tokenize`

### Setup

#### Using `uv` (Recommended)

```bash
# Sync project (creates venv and installs dependencies)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

#### Using standard `pip`

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: This project uses `pyproject.toml` for dependency management. The `requirements.txt` file is kept for backward compatibility.

## Legacy (deprecated)

The scripts below are kept for ad-hoc debugging only. The recommended workflow is the **KV-cache experiment suite** above.

### 1. Memory Profiler (legacy)

The memory profiler monitors any process by name or command substring and samples memory at fixed intervals.

```bash
python legacy/memory_profiler.py --name "ollama" --label "ollama-run1" --output results.csv
```

**Options:**
- `--name`: Process name or command substring (e.g., "ollama", "llama-cli")
- `--label`: Label for this run (e.g., "ollama-run1", "llamacpp-baseline")
- `--interval`: Sampling interval in seconds (default: 0.1 = 100ms)
- `--output`: Output CSV file path
- `--duration`: Maximum profiling duration in seconds (optional)
- `--max-samples`: Maximum number of samples (optional)

**Output CSV format:**
```csv
timestamp,label,pid,rss_mb,vms_mb
2024-01-15T10:30:00.123456,ollama-run1,12345,512.34,2048.67
```

The profiler automatically detects when the target process starts and stops sampling when it exits.

### 2. Ollama Runner (legacy)

Runs deterministic inference using Ollama's HTTP API.

```bash
# Basic usage
python legacy/run_ollama.py --model llama3.2:latest --prompt "The quick brown fox"

# With custom parameters
python legacy/run_ollama.py \
  --model llama3.2:latest \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.0
```

**Prerequisites:**
- Ollama running: `ollama serve`
- Model pulled: `ollama pull llama3.2:1b`

**Options:**
- `--model`: Model name (e.g., "llama3.2:1b", "mistral:7b")
- `--prompt`: Input prompt (default: "The quick brown fox jumps over the lazy dog.")
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.0 for deterministic)
- `--base-url`: Ollama API URL (default: http://localhost:11434)

### 3. llama.cpp Runner (legacy)

Runs deterministic inference using llama.cpp CLI.

```bash
# Basic usage (auto-detects llama-cli)
python legacy/run_llamacpp.py --model model.gguf --prompt "The quick brown fox"

# With custom executable path
python legacy/run_llamacpp.py \
  --model model.gguf \
  --prompt "Once upon a time" \
  --n-predict 100 \
  --llama-cli-path /path/to/llama-cli
```

**Prerequisites:**
- llama.cpp built (produces `llama-cli` or `main` executable)
- GGUF model file available

**Options:**
- `--model`: Path to GGUF model file
- `--prompt`: Input prompt (default: "The quick brown fox jumps over the lazy dog.")
- `--n-predict`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.0)
- `--llama-cli-path`: Path to llama-cli executable (auto-detected if not specified)
- `--n-threads`: Number of threads (default: auto)
- `--n-ctx`: Context window size (default: 512)

### 4. WebLLM Profiler (deferred)

> WebLLM is temporarily out of scope for the KV-cache quantization study refactor.
> The primary workflow is the experiment suite under `experiments/`.
> The section below is kept for reference and may be revisited later.
Memory profiling for browser-based LLM inference.

#### Quick Start

1. **Open the profiler page:**
   ```bash
   # Option 1: Open directly in Chrome
   open -a "Google Chrome" legacy/webllm/webllm_profiler.html
   
   # Option 2: Serve via local server (recommended)
   python -m http.server 8000
   # Then open http://localhost:8000/legacy/webllm/webllm_profiler.html
   ```

2. **Enable memory API in Chrome:**
   ```bash
   # Chrome needs --enable-precise-memory-info flag
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
     --enable-precise-memory-info \
     --enable-memory-info \
     legacy/webllm/webllm_profiler.html
   ```

3. **Use the interface:**
   - Select model
   - Enter prompt
   - Set max tokens
   - Click "Start Profiling"
   - Click "Export CSV" when done

#### Advanced Usage

For programmatic control, use the `WebLLMProfiler` class from `legacy/webllm/webllm_profiler_advanced.js`:

```javascript
const profiler = new WebLLMProfiler('Llama-3.2-1B-Instruct-q4f16_1', {
    samplingInterval: 100, // ms
    onSample: (sample) => console.log('Memory:', sample),
    onToken: (count) => console.log('Tokens:', count),
});

await profiler.initialize();
const result = await profiler.complete("The quick brown fox", {
    max_gen_len: 50
});

profiler.downloadCSV();
```

## Complete Workflow Examples

### Profiling Ollama

```bash
# Terminal 1: Start memory profiler
python memory_profiler.py \
  --name "ollama" \
  --label "ollama-llama3.2-1b-50tokens" \
  --interval 0.1 \
  --output ollama_memory.csv

# Terminal 2: Run inference (profiler will detect the process)
python run_ollama.py \
  --model llama3.2:1b \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-tokens 50

# Profiler will automatically stop when process exits
```

### Profiling llama.cpp

```bash
# Terminal 1: Start memory profiler
python memory_profiler.py \
  --name "llama-cli" \
  --label "llamacpp-llama3.2-1b-50tokens" \
  --interval 0.05 \
  --output llamacpp_memory.csv

# Terminal 2: Run inference
python run_llamacpp.py \
  --model models/llama-3.2-1b.gguf \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --n-predict 50
```

### Profiling WebLLM

1. Launch Chrome with memory API enabled (see WebLLM Profiler section above)
2. Open `webllm_profiler.html`
3. Configure model, prompt, and max tokens
4. Click "Start Profiling" and wait for inference to complete
5. Click "Export CSV" to download results

## Data Analysis

All CSV files use the same format, making comparison straightforward:

```python
import pandas as pd

# Load data
ollama = pd.read_csv('ollama_memory.csv')
llamacpp = pd.read_csv('llamacpp_memory.csv')
webllm = pd.read_csv('webllm_memory.csv')

# Compare peak memory
print(f"Ollama peak RSS: {ollama['rss_mb'].max():.2f} MB")
print(f"llama.cpp peak RSS: {llamacpp['rss_mb'].max():.2f} MB")
print(f"WebLLM peak heap: {webllm['heap_used_mb'].max():.2f} MB")
```

## Common Pitfalls & Solutions

### macOS Apple Silicon Specific

1. **Process name variations:**
   - Ollama might appear as `ollama` or `Ollama` (case-insensitive matching handles this)
   - llama.cpp might be `llama-cli`, `main`, or `llama` depending on build
   - Check process name with: `ps aux | grep ollama`

2. **Short-lived processes:**
   - The profiler waits up to 10 seconds for processes to appear
   - For very short runs, use smaller intervals (e.g., `--interval 0.05`)
   - Consider using `--duration` to limit profiling time

3. **Permission issues:**
   - macOS may require Terminal/IDE to have "Full Disk Access" for process monitoring
   - If you see "AccessDenied" errors, grant permissions in System Settings → Privacy & Security

4. **Memory measurement accuracy:**
   - RSS (Resident Set Size) is the most relevant for physical memory
   - VMS (Virtual Memory Size) includes memory-mapped files
   - On Apple Silicon, memory compression may affect measurements

### WebLLM Specific

1. **Memory API not available:**
   - Chrome requires `--enable-precise-memory-info` flag
   - Alternative: Use Chrome DevTools Performance tab and export manually
   - Firefox doesn't support `performance.memory`

2. **Token counting:**
   - WebLLM token counting is approximate (word-based)
   - For accurate token counts, check WebLLM's internal metrics if available
   - Consider aligning with actual token generation events

3. **Browser overhead:**
   - Browser memory includes rendering, extensions, etc.
   - Baseline measurement (before inference) helps isolate LLM memory
   - Use incognito mode to minimize extension interference

### General Issues

1. **Process not found:**
   - Ensure process is running before starting profiler (or start profiler first, it will wait)
   - Check process name with: `ps aux | grep <process-name>`
   - Use command substring if process name varies

2. **Sampling rate:**
   - Too high (e.g., 0.01s) may impact performance
   - Too low (e.g., 1.0s) may miss memory spikes
   - Recommended: 0.05-0.1s for most use cases

3. **Reproducibility:**
   - Use fixed prompts and token limits
   - Set temperature to 0.0 for deterministic output
   - Close other applications to minimize interference
   - Run multiple times and average results

## Project Structure

```
kv-caching/
├── experiments/
│   ├── config.toml              # Sweep grid + paths
│   ├── generate_prompts.py      # Exact-token prompt generator (via llama-tokenize)
│   ├── run_suite.py             # Main experiment runner (writes results/)
│   ├── ollama_engine.py         # Ollama server + request timing helpers
│   └── llamacpp_engine.py       # llama.cpp invocation + timing parser
├── profiling/
│   └── process_sampler.py       # RSS/VMS sampler + window metric helpers
├── inputs/
│   ├── prompts/                 # ctx_<N>.txt + manifest.json (generated)
│   └── README.md                # Prompt-file usage notes
├── results/                     # outputs (timestamped) from run_suite.py
├── legacy/                      # deprecated one-off scripts / WebLLM artifacts
├── pyproject.toml               # Project configuration (uv/pip standard)
├── uv.lock                      # Lock file for reproducible builds (uv)
├── requirements.txt             # Python dependencies (backward compatibility)
├── .venv/                       # Virtual environment (created by uv)
└── README.md                    # This file
```

### Project Structure Notes

This project follows the **uv project structure**:
- **`pyproject.toml`**: Modern Python project configuration with dependencies
- **`uv.lock`**: Lock file ensuring reproducible dependency versions
- **Standalone scripts**: Scripts are run directly (not installed as packages)
- **Virtual environment**: Managed by `uv sync` in `.venv/`

The `requirements.txt` file is kept for backward compatibility, but `pyproject.toml` is the source of truth for dependencies.

## Research Notes

### Memory Metrics

- **RSS (Resident Set Size)**: Physical memory currently in RAM
- **VMS (Virtual Memory Size)**: Total virtual memory (includes swapped/mapped)
- **VRAM proxy (Apple Silicon)**: We treat **process RSS** as a practical unified-memory proxy for “VRAM utilization”

### Alignment with Token Generation

To align memory samples with token generation:
1. Record timestamps for each token generation event
2. Interpolate memory samples to token timestamps
3. Or: Use token count as x-axis instead of time

### Reproducibility Checklist

- [ ] Same model (size, quantization) across runtimes
- [ ] Same prompt
- [ ] Same max tokens
- [ ] Temperature = 0.0 (deterministic)
- [ ] Minimal background processes
- [ ] Multiple runs for statistical significance
- [ ] Document system state (macOS version, runtime versions)

## License

Research use only. See project license for details.
