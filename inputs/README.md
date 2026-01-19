# Prompt inputs

This folder contains text files for prompt inputs.

- `prompts/`: **generated** exact-token prompts for the KV-cache suite (`ctx_<N>.txt`)
- `manual/`: **hand-written** long prompts for ad-hoc debugging

## Usage

### Ollama

```bash
python legacy/run_ollama.py --model llama3.2:latest --prompt-file manual/long_prompt.txt --max-tokens 200
```

Or specify the full path:
```bash
python legacy/run_ollama.py --model llama3.2:latest --prompt-file inputs/manual/long_prompt.txt --max-tokens 200
```

### llama.cpp

```bash
python legacy/run_llamacpp.py --model model.gguf --prompt-file manual/long_prompt.txt --n-predict 200
```

## Creating Your Own Prompts

1. Create a `.txt` file in this folder
2. Write your prompt (can be multi-line)
3. Use `--prompt-file filename.txt` to load it

The legacy scripts will automatically look in the `inputs/` folder if the file isn't found in the current directory.
