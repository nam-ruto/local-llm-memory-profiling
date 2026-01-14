#!/usr/bin/env python3
"""
llama.cpp Runner Script

Runs a deterministic inference using llama.cpp CLI.
Designed for reproducible memory profiling experiments.

Prerequisites:
    - llama.cpp built (make or cmake)
    - Model file available (GGUF format)

Usage:
    python run_llamacpp.py --model model.gguf --prompt "The quick brown fox" --n-predict 50
"""

import argparse
import subprocess
import sys
import time
import json
import os


def find_llama_cli():
    """
    Try to find llama.cpp CLI executable.
    
    Checks common locations:
    - ./llama-cli (current directory)
    - ./main (current directory, common build name)
    - llama-cli in PATH
    
    Returns:
        Path to executable or None
    """
    # Check current directory first
    for exe in ['llama-cli', 'main', './llama-cli', './main']:
        if os.path.exists(exe) and os.access(exe, os.X_OK):
            return exe
    
    # Check PATH
    import shutil
    for exe in ['llama-cli', 'main']:
        path = shutil.which(exe)
        if path:
            return path
    
    return None


def run_llamacpp_inference(
    model_path: str,
    prompt: str,
    n_predict: int = 50,
    temperature: float = 0.0,
    llama_cli_path: str = None,
    n_threads: int = None,
    n_ctx: int = 512,
):
    """
    Run inference using llama.cpp CLI.
    
    Args:
        model_path: Path to GGUF model file
        prompt: Input prompt text
        n_predict: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        llama_cli_path: Path to llama-cli executable (None = auto-detect)
        n_threads: Number of threads (None = default)
        n_ctx: Context window size
        
    Returns:
        dict with response data and timing information
    """
    # Find executable
    if llama_cli_path is None:
        llama_cli_path = find_llama_cli()
        if llama_cli_path is None:
            print("ERROR: Could not find llama-cli executable.")
            print("Please build llama.cpp or specify --llama-cli-path")
            sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Executable: {llama_cli_path}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {n_predict}")
    print(f"Temperature: {temperature}")
    print(f"\nRunning inference...\n")
    
    # Build command
    cmd = [
        llama_cli_path,
        '-m', model_path,
        '-p', prompt,
        '-n', str(n_predict),
        '-t', str(n_threads) if n_threads else '0',  # 0 = auto
        '-c', str(n_ctx),
        '--temp', str(temperature),
        '--log-disable',  # Disable verbose logging for cleaner output
    ]
    
    start_time = time.time()
    
    try:
        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        output = result.stdout
        stderr = result.stderr
        
        # Parse output (llama.cpp outputs generated text to stdout)
        # The prompt is typically echoed, then the generated text follows
        generated_text = output.strip()
        
        # Try to extract timing info from stderr if available
        tokens_generated = n_predict  # Approximate, may need parsing
        
        print(f"{'='*60}")
        print(f"Completed in {elapsed:.2f}s")
        print(f"\nGenerated text:")
        print(f"{generated_text}")
        if stderr:
            print(f"\nStderr output:")
            print(f"{stderr}")
        print(f"{'='*60}\n")
        
        return {
            'model': model_path,
            'prompt': prompt,
            'n_predict': n_predict,
            'generated_text': generated_text,
            'tokens_generated': tokens_generated,  # Approximate
            'elapsed_time': elapsed,
            'stderr': stderr,
        }
        
    except subprocess.TimeoutExpired:
        print("ERROR: Inference timed out after 5 minutes.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: llama.cpp exited with code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Executable not found: {llama_cli_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run deterministic inference with llama.cpp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_llamacpp.py --model models/llama-3.2-1b.gguf --prompt "The quick brown fox"
  
  # With custom token limit
  python run_llamacpp.py --model model.gguf --prompt "Once upon a time" --n-predict 100
  
  # Specify custom executable path
  python run_llamacpp.py --model model.gguf --llama-cli-path /path/to/llama-cli
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to GGUF model file'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        default="The quick brown fox jumps over the lazy dog.",
        help='Input prompt (default: "The quick brown fox jumps over the lazy dog.")'
    )
    
    parser.add_argument(
        '--n-predict', '-n',
        type=int,
        default=50,
        help='Maximum tokens to generate (default: 50)'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )
    
    parser.add_argument(
        '--llama-cli-path',
        default=None,
        help='Path to llama-cli executable (default: auto-detect)'
    )
    
    parser.add_argument(
        '--n-threads',
        type=int,
        default=None,
        help='Number of threads (default: auto)'
    )
    
    parser.add_argument(
        '--n-ctx',
        type=int,
        default=512,
        help='Context window size (default: 512)'
    )
    
    args = parser.parse_args()
    
    result = run_llamacpp_inference(
        model_path=args.model,
        prompt=args.prompt,
        n_predict=args.n_predict,
        temperature=args.temperature,
        llama_cli_path=args.llama_cli_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
    )
    
    # Save metadata to JSON for later analysis
    model_name = os.path.basename(args.model).replace('.gguf', '')
    output_file = f"llamacpp_{model_name}_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Metadata saved to: {output_file}")


if __name__ == '__main__':
    main()
