#!/usr/bin/env python3
"""
Memory Profiler for On-Device LLM Runtimes

A lightweight tool to monitor memory usage (RSS and VMS) of processes
during LLM inference. Designed for research reproducibility.

Usage:
    python memory_profiler.py --name "ollama" --label "ollama-run1" --interval 0.1 --output results.csv
"""

import argparse
import csv
import time
import sys
from datetime import datetime
from typing import Optional, List
import psutil


class MemoryProfiler:
    """Monitors memory usage of a process over time."""
    
    def __init__(self, process_identifier: str, label: str, interval: float = 0.1):
        """
        Initialize the memory profiler.
        
        Args:
            process_identifier: Process name or command substring to match
            label: Label for this profiling run (e.g., "ollama-run1")
            interval: Sampling interval in seconds (default: 0.1 = 100ms)
        """
        self.process_identifier = process_identifier
        self.label = label
        self.interval = interval
        self.samples = []
        
    def find_process(self) -> Optional[psutil.Process]:
        """
        Find a process by name or command substring.
        
        Returns:
            psutil.Process if found, None otherwise
        """
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check process name
                if self.process_identifier.lower() in proc.info['name'].lower():
                    return proc
                
                # Check command line
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if self.process_identifier.lower() in cmdline.lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or we don't have permission
                continue
        
        return None
    
    def sample_memory(self, proc: psutil.Process) -> Optional[dict]:
        """
        Sample memory usage for a process.
        
        Args:
            proc: psutil.Process to sample
            
        Returns:
            dict with timestamp, label, pid, rss_mb, vms_mb, or None if process is gone
        """
        try:
            mem_info = proc.memory_info()
            return {
                'timestamp': datetime.now().isoformat(),
                'label': self.label,
                'pid': proc.pid,
                'rss_mb': mem_info.rss / (1024 * 1024),  # Convert bytes to MB
                'vms_mb': mem_info.vms / (1024 * 1024),  # Convert bytes to MB
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    
    def profile(self, duration: Optional[float] = None, max_samples: Optional[int] = None):
        """
        Profile memory usage over time.
        
        Args:
            duration: Maximum duration in seconds (None = until process exits)
            max_samples: Maximum number of samples (None = unlimited)
        """
        print(f"Looking for process matching: '{self.process_identifier}'")
        print(f"Sampling interval: {self.interval}s")
        
        start_time = time.time()
        sample_count = 0
        
        # Wait for process to appear (useful for short-lived processes)
        proc = None
        wait_timeout = 10.0  # Wait up to 10 seconds for process to appear
        wait_start = time.time()
        
        while proc is None and (time.time() - wait_start) < wait_timeout:
            proc = self.find_process()
            if proc is None:
                time.sleep(0.1)
        
        if proc is None:
            print(f"ERROR: Could not find process matching '{self.process_identifier}'")
            print("Hint: Make sure the process is running or starts shortly after this profiler.")
            sys.exit(1)
        
        print(f"Found process: PID {proc.pid}")
        print(f"Profiling started. Press Ctrl+C to stop early.\n")
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Check sample limit
                if max_samples and sample_count >= max_samples:
                    break
                
                # Sample memory
                sample = self.sample_memory(proc)
                if sample:
                    self.samples.append(sample)
                    sample_count += 1
                    print(f"[{sample['timestamp']}] PID {sample['pid']}: "
                          f"RSS={sample['rss_mb']:.2f} MB, "
                          f"VMS={sample['vms_mb']:.2f} MB")
                else:
                    # Process has terminated
                    print(f"\nProcess {proc.pid} has terminated.")
                    break
                
                # Sleep before next sample
                time.sleep(self.interval)
                
                # Re-find process in case PID changed (for some runtimes)
                # This helps with short-lived processes that restart
                new_proc = self.find_process()
                if new_proc and new_proc.pid != proc.pid:
                    print(f"Process PID changed: {proc.pid} -> {new_proc.pid}")
                    proc = new_proc
                    
        except KeyboardInterrupt:
            print("\n\nProfiling interrupted by user.")
        
        print(f"\nCollected {len(self.samples)} samples.")
    
    def save_csv(self, output_path: str):
        """Save samples to CSV file."""
        if not self.samples:
            print("WARNING: No samples to save.")
            return
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'label', 'pid', 'rss_mb', 'vms_mb']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Memory profiler for on-device LLM runtimes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile Ollama process
  python memory_profiler.py --name "ollama" --label "ollama-run1" --output ollama_memory.csv
  
  # Profile llama.cpp with 50ms sampling
  python memory_profiler.py --name "llama-cli" --label "llamacpp-run1" --interval 0.05 --output llamacpp_memory.csv
  
  # Profile for maximum 30 seconds
  python memory_profiler.py --name "ollama" --label "test" --duration 30 --output test.csv
        """
    )
    
    parser.add_argument(
        '--name', '--process',
        dest='process_name',
        required=True,
        help='Process name or command substring to match (e.g., "ollama", "llama-cli")'
    )
    
    parser.add_argument(
        '--label',
        required=True,
        help='Label for this profiling run (e.g., "ollama-run1", "llamacpp-baseline")'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=0.1,
        help='Sampling interval in seconds (default: 0.1 = 100ms)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Maximum profiling duration in seconds (default: until process exits)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to collect (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    profiler = MemoryProfiler(
        process_identifier=args.process_name,
        label=args.label,
        interval=args.interval
    )
    
    profiler.profile(duration=args.duration, max_samples=args.max_samples)
    profiler.save_csv(args.output)


if __name__ == '__main__':
    main()
