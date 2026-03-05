#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as cfg  # your config.py


def find_latest_summary(results_dir: str) -> str | None:
    pattern = os.path.join(results_dir, "rodinia_*_summary.csv")
    files = glob.glob(pattern)
    files = [f for f in files if "breakdowns" not in os.path.basename(f).lower()]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "benchmark" not in df.columns or "avg_ms" not in df.columns:
        raise ValueError(f"{path}: expected columns benchmark, avg_ms")
    df["avg_ms"] = pd.to_numeric(df["avg_ms"], errors="coerce")
    return df[["benchmark", "avg_ms"]].copy()


def resolve_benchmark_order(dfs: dict) -> list[str]:
    if "CUDA" in dfs and len(dfs["CUDA"]) > 0:
        preferred = dfs["CUDA"]["benchmark"].tolist()
    else:
        preferred = next(iter(dfs.values()))["benchmark"].tolist()

    all_bench = sorted(set().union(*[set(df["benchmark"]) for df in dfs.values()]))

    seen, out = set(), []
    for b in preferred:
        if b in all_bench and b not in seen:
            out.append(b)
            seen.add(b)
    for b in all_bench:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="computation_bars.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--figsize", default="13,2.8")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--logy-min", type=float, default=1e-2)
    args = ap.parse_args()

    backends = [
        ("CUDA", "cuda/results"),
        ("HIP", "hipify_cuda_2/results"),
        ("SCALE", "scale/results"),
        ("SYCL", "sycl/results"),
    ]

    # Same backend colors as breakdown plot
    backend_colors = {
        "CUDA":"#76B900",   # green
        "HIP": "#9C2A44",
        "SCALE": "#2B1D4F",  # purple
        "SYCL": "#F35A1C",   # red
    }

    dfs = {}
    picked = []

    for name, rdir in backends:
        if not os.path.isdir(rdir):
            print(f"[ MISSING DIR ] {rdir}", file=sys.stderr)
            continue
        f = find_latest_summary(rdir)
        if not f:
            print(f"[ NO FILES    ] {rdir}/rodinia_*_summary.csv (excluding breakdowns)", file=sys.stderr)
            continue
        print(f"[ {name:<5} ] {f}")
        dfs[name] = load_summary(f)
        picked.append(name)

    if not dfs:
        print("[ ERROR ] No computation summary CSVs found.", file=sys.stderr)
        sys.exit(1)

    benchmarks = resolve_benchmark_order(dfs)

    table = pd.DataFrame({"benchmark": benchmarks}).set_index("benchmark")
    for name in picked:
        table[name] = dfs[name].set_index("benchmark").reindex(benchmarks)["avg_ms"]

    W, H = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(W, H), dpi=args.dpi)

    plt.rcParams.update({"font.size": cfg.fontsize, "axes.linewidth": cfg.edgewidth})

    n_apps = len(benchmarks)
    names = picked
    n_back = len(names)

    x = np.arange(n_apps)

    if n_back == 1:
        bar_w = 0.6
        offsets = np.array([0.0])
    else:
        group_width = 0.82
        bar_w = group_width / n_back
        offsets = (np.arange(n_back) - (n_back - 1) / 2.0) * bar_w

    # Keep patterns per backend (optional), but use backend colors primarily
    backend_hatches = cfg.patterns[:max(1, n_back)]

    for i, name in enumerate(names):
        y = table[name].to_numpy(dtype=float)
        mask = np.isfinite(y)
        ax.bar(
            x[mask] + offsets[i],
            y[mask],
            width=bar_w * 0.95,
            color=backend_colors.get(name, "#7f7f7f"),
            alpha=0.85,
            edgecolor=cfg.edgecolor,
            linewidth=cfg.edgewidth,
            hatch=backend_hatches[i],
            label=name,
            zorder=3,
        )

    ax.set_ylabel("Computation Time (ms)")
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", rotation_mode="anchor")

    if args.title:
        ax.set_title(args.title)

    if args.logy:
        if args.logy_min <= 0:
            raise SystemExit("--logy-min must be > 0.")
        ax.set_yscale("log")
        ax.set_ylim(args.logy_min, None)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # Legend closer to plot
    ax.legend(loc="upper left", bbox_to_anchor=(0.86, 0.98), frameon=False, borderaxespad=0.0)
    fig.subplots_adjust(bottom=0.25, right=0.87)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ DONE ] Wrote {args.out}")


if __name__ == "__main__":
    main()
