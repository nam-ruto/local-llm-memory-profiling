# Memory Profiling Framework for On-Device LLM Runtimes

A lightweight, reproducible memory profiling framework for measuring memory usage during LLM inference across different runtimes on macOS (Apple Silicon).

**Research Project**: "An Empirical Study of KV-Cache Memory Behavior in On-Device LLM Runtimes"

## Overview

This framework enables systematic memory profiling of:
- **Ollama** (via HTTP API)
- **llama.cpp** (via CLI)
- **WebLLM** (via browser JavaScript)

All profilers output CSV data in a consistent format for easy comparison.

## Installation

### Prerequisites

- Python 3.8+
- macOS (Apple Silicon)
- For Ollama: [Ollama installed](https://ollama.ai)
- For llama.cpp: [llama.cpp built](https://github.com/ggerganov/llama.cpp)
- For WebLLM: Chrome browser

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

## Usage

### 1. Memory Profiler (Core Tool)

The memory profiler monitors any process by name or command substring and samples memory at fixed intervals.

```bash
python memory_profiler.py --name "ollama" --label "ollama-run1" --output results.csv
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

### 2. Ollama Runner

Runs deterministic inference using Ollama's HTTP API.

```bash
# Basic usage
python run_ollama.py --model llama3.2:1b --prompt "The quick brown fox"

# With custom parameters
python run_ollama.py \
  --model llama3.2:1b \
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

### 3. llama.cpp Runner

Runs deterministic inference using llama.cpp CLI.

```bash
# Basic usage (auto-detects llama-cli)
python run_llamacpp.py --model models/llama-3.2-1b.gguf --prompt "The quick brown fox"

# With custom executable path
python run_llamacpp.py \
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

### 4. WebLLM Profiler

Memory profiling for browser-based LLM inference.

#### Quick Start

1. **Open the profiler page:**
   ```bash
   # Option 1: Open directly in Chrome
   open -a "Google Chrome" webllm_profiler.html
   
   # Option 2: Serve via local server (recommended)
   python -m http.server 8000
   # Then open http://localhost:8000/webllm_profiler.html
   ```

2. **Enable memory API in Chrome:**
   ```bash
   # Chrome needs --enable-precise-memory-info flag
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
     --enable-precise-memory-info \
     --enable-memory-info \
     webllm_profiler.html
   ```

3. **Use the interface:**
   - Select model
   - Enter prompt
   - Set max tokens
   - Click "Start Profiling"
   - Click "Export CSV" when done

#### Advanced Usage

For programmatic control, use the `WebLLMProfiler` class from `webllm_profiler_advanced.js`:

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
├── memory_profiler.py          # Core memory profiler
├── run_ollama.py                # Ollama runner script
├── run_llamacpp.py              # llama.cpp runner script
├── webllm_profiler.html         # WebLLM profiler UI
├── webllm_profiler_advanced.js  # Advanced WebLLM profiler class
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
- **Heap (WebLLM)**: JavaScript heap usage (closest to RSS equivalent)

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
