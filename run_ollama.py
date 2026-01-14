#!/usr/bin/env python3
"""
Ollama Runner Script

Runs a deterministic inference request using Ollama's local HTTP API.
Designed for reproducible memory profiling experiments.

Prerequisites:
    - Ollama installed and running (ollama serve)
    - Model pulled (e.g., ollama pull llama3.2:1b)

Usage:
    python run_ollama.py --model llama3.2:1b --prompt "The quick brown fox" --max-tokens 50
"""

import argparse
import json
import requests
import sys
import time


def run_ollama_inference(
    model: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434"
):
    """
    Run inference using Ollama's HTTP API.
    
    Args:
        model: Model name (e.g., "llama3.2:1b")
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        base_url: Ollama API base URL
        
    Returns:
        dict with response data and timing information
    """
    api_url = f"{base_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Non-streaming for cleaner profiling
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    }
    
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"\nSending request to {api_url}...")
    
    start_time = time.time()
    
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        result = response.json()
        
        generated_text = result.get('response', '')
        tokens_generated = result.get('eval_count', 0)
        total_duration = result.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
        eval_duration = result.get('eval_duration', 0) / 1e9
        
        print(f"\n{'='*60}")
        print(f"Response received in {elapsed:.2f}s")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Total duration: {total_duration:.3f}s")
        print(f"Eval duration: {eval_duration:.3f}s")
        print(f"Tokens/second: {tokens_generated / eval_duration:.2f}" if eval_duration > 0 else "")
        print(f"\nGenerated text:")
        print(f"{generated_text}")
        print(f"{'='*60}\n")
        
        return {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated_text': generated_text,
            'tokens_generated': tokens_generated,
            'total_duration': total_duration,
            'eval_duration': eval_duration,
            'elapsed_time': elapsed,
        }
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Ollama API.")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error: {e}")
        if response.status_code == 404:
            print(f"Model '{model}' not found. Pull it first: ollama pull {model}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run deterministic inference with Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_ollama.py --model llama3.2:1b --prompt "The quick brown fox"
  
  # With custom token limit
  python run_ollama.py --model llama3.2:1b --prompt "Once upon a time" --max-tokens 100
  
  # With custom temperature
  python run_ollama.py --model llama3.2:1b --prompt "Hello" --temperature 0.7
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Model name (e.g., llama3.2:1b, mistral:7b)'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        default="The quick brown fox jumps over the lazy dog.",
        help='Input prompt (default: "The quick brown fox jumps over the lazy dog.")'
    )
    
    parser.add_argument(
        '--max-tokens', '-n',
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
        '--base-url',
        default='http://localhost:11434',
        help='Ollama API base URL (default: http://localhost:11434)'
    )
    
    args = parser.parse_args()
    
    result = run_ollama_inference(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url
    )
    
    # Save metadata to JSON for later analysis
    output_file = f"ollama_{args.model.replace(':', '_')}_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Metadata saved to: {output_file}")


if __name__ == '__main__':
    main()
