#!/usr/bin/env python3
"""
Visualize KV-cache experiment results: combine runs from multiple result dirs
and plot peak RSS, prefill TPS, and gen TPS vs context length (by engine and KV type).

Usage:
  uv sync --extra analysis   # or: pip install pandas matplotlib
  python experiments/visualize_runs.py --runs results/20260217_160332 results/20260217_165615 --out results/figures
  python experiments/visualize_runs.py --runs results/20260217_160332 results/20260217_165615  # prints paths, no --out
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: uv sync --extra analysis  or  pip install pandas matplotlib", file=sys.stderr)
    sys.exit(1)


def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        csv_path = os.path.join(d, "runs.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df["run_dir"] = os.path.basename(d.rstrip("/"))
        frames.append(df)
    if not frames:
        raise SystemExit("No runs.csv found in any of the given directories.")
    combined = pd.concat(frames, ignore_index=True)
    # Coerce numeric columns (may be read as object if some cells are empty)
    for col in ["context_tokens", "baseline_idle_rss_mb", "peak_prefill_rss_mb", "peak_total_rss_mb", "prefill_tps", "gen_tps"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def plot_peak_rss(df: pd.DataFrame, out_dir: str | None) -> str | None:
    df_plot = df.dropna(subset=["peak_total_rss_mb", "context_tokens"])
    if df_plot.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for (engine, kv), grp in df_plot.groupby(["engine", "kv_cache_type"], sort=False):
        grp = grp.sort_values("context_tokens")
        label = f"{engine} {kv}"
        ax.plot(grp["context_tokens"], grp["peak_total_rss_mb"], "o-", label=label, markersize=6)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak total RSS (MB)")
    ax.set_title("Peak process RSS vs context length (by engine and KV-cache type)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", base=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "peak_rss_mb.png") if out_dir else None
    if path:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    plt.show()
    return None


def plot_prefill_tps(df: pd.DataFrame, out_dir: str | None) -> str | None:
    df_plot = df.dropna(subset=["prefill_tps", "context_tokens"])
    if df_plot.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for (engine, kv), grp in df_plot.groupby(["engine", "kv_cache_type"], sort=False):
        grp = grp.sort_values("context_tokens")
        label = f"{engine} {kv}"
        ax.plot(grp["context_tokens"], grp["prefill_tps"], "o-", label=label, markersize=6)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Prefill TPS (tokens/s)")
    ax.set_title("Prefill throughput vs context length (by engine and KV-cache type)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", base=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "prefill_tps.png") if out_dir else None
    if path:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    plt.show()
    return None


def plot_gen_tps(df: pd.DataFrame, out_dir: str | None) -> str | None:
    df_plot = df.dropna(subset=["gen_tps", "context_tokens"])
    if df_plot.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for (engine, kv), grp in df_plot.groupby(["engine", "kv_cache_type"], sort=False):
        grp = grp.sort_values("context_tokens")
        label = f"{engine} {kv}"
        ax.plot(grp["context_tokens"], grp["gen_tps"], "o-", label=label, markersize=6)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Generation TPS (tokens/s)")
    ax.set_title("Generation throughput vs context length (by engine and KV-cache type)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", base=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "gen_tps.png") if out_dir else None
    if path:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    plt.show()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize KV-cache experiment runs (RSS and TPS)")
    ap.add_argument("--runs", nargs="+", required=True, help="Result directories (each with runs.csv)")
    ap.add_argument("--out", default=None, help="Output directory for figures (default: show only)")
    args = ap.parse_args()

    df = load_runs(args.runs)
    print(f"Loaded {len(df)} rows from {args.runs}")
    print(df.groupby("engine").size().to_string(), end="\n\n")

    out_dir = args.out
    paths = []
    if p := plot_peak_rss(df, out_dir):
        paths.append(p)
    if p := plot_prefill_tps(df, out_dir):
        paths.append(p)
    if p := plot_gen_tps(df, out_dir):
        paths.append(p)

    if paths:
        print("Saved:")
        for p in paths:
            print(f"  {p}")
    elif out_dir:
        print("No data to plot (missing peak_total_rss_mb / prefill_tps / gen_tps).")


if __name__ == "__main__":
    main()
