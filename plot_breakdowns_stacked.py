#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

import config as cfg  # your config.py

METRICS = [
    "Allocation time",
    "H2D transfer time",
    "Compute time",
    "D2H transfer time",
    "Free time",
]


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "benchmark" not in df.columns:
        raise ValueError(f"{path}: missing 'benchmark' column")

    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)

    return df[["benchmark"] + METRICS]


def resolve_benchmark_order(dfs: dict, backends: list) -> list:
    if "CUDA" in dfs and len(dfs["CUDA"]) > 0:
        preferred = dfs["CUDA"]["benchmark"].tolist()
    else:
        preferred = dfs[backends[0][0]]["benchmark"].tolist()

    all_bench = sorted(set().union(*[set(df["benchmark"]) for df in dfs.values()]))

    seen, benchmarks = set(), []
    for b in preferred:
        if b in all_bench and b not in seen:
            benchmarks.append(b)
            seen.add(b)
    for b in all_bench:
        if b not in seen:
            benchmarks.append(b)
            seen.add(b)
    return benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", default=None)
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--scale", default=None)
    ap.add_argument("--sycl", default=None)

    ap.add_argument("-o", "--out", default="breakdowns_stacked.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--figsize", default="13,3.2")
    ap.add_argument("--dpi", type=int, default=200)

    # log-scale y-axis
    ap.add_argument("--logy", action="store_true", help="Use log-scale y-axis.")
    ap.add_argument(
        "--logy-min",
        type=float,
        default=1e-2,
        help="Lower y-limit for log-scale plots (must be >0). Default: 1e-2",
    )

    # squash outlier segments
    ap.add_argument(
        "--squash-outliers",
        action="store_true",
        help="Squash extreme outlier *segments* by plotting them as a small constant height and adding a legend note.",
    )
    ap.add_argument("--squash-factor", type=float, default=8.0)
    ap.add_argument("--squash-abs-min", type=float, default=1000.0)
    ap.add_argument("--squash-to", type=float, default=1.0)

    # x-ticks
    ap.add_argument("--xtick-rotation", type=float, default=45.0)
    ap.add_argument("--xtick-ha", default="right", choices=["left", "center", "right"])

    args = ap.parse_args()

    if args.logy and args.squash_outliers:
        raise SystemExit("Choose one: --logy OR --squash-outliers (not both).")

    candidates = [
        ("CUDA", args.cuda),
        ("HIP", args.hip),
        ("SCALE", args.scale),
        ("SYCL", args.sycl),
    ]
    backends = [(name, path) for name, path in candidates if path]
    if not backends:
        raise SystemExit("No input CSVs provided. Pass at least one of --cuda/--hip/--scale/--sycl")

    dfs = {name: load_csv(path) for name, path in backends}
    benchmarks = resolve_benchmark_order(dfs, backends)

    for name in list(dfs.keys()):
        dfs[name] = (
            dfs[name]
            .set_index("benchmark")
            .reindex(benchmarks)
            .fillna(0.0)
            .reset_index()
        )

    plot_dfs = {k: v.copy() for k, v in dfs.items()}

    # squash outliers
    squashed = []
    if args.squash_outliers:
        for backend, _ in backends:
            df = plot_dfs[backend]
            for metric in METRICS:
                vals = df[metric].to_numpy(dtype=float)
                nz = vals[vals > 0]
                if nz.size < 2:
                    continue
                sorted_vals = np.sort(nz)
                second = sorted_vals[-2]
                mx = sorted_vals[-1]

                if mx < args.squash_abs_min:
                    continue
                if second <= 0:
                    continue
                if mx <= args.squash_factor * second:
                    continue

                idxs = np.where(vals == mx)[0]
                for i in idxs:
                    squashed.append(
                        dict(
                            backend=backend,
                            benchmark=df.loc[i, "benchmark"],
                            metric=metric,
                            original_ms=float(mx),
                            squashed_to_ms=float(args.squash_to),
                            second_ms=float(second),
                        )
                    )
                    df.at[i, metric] = float(args.squash_to)

    # Figure
    W, H = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(W, H), dpi=args.dpi)

    plt.rcParams.update(
        {
            "font.size": cfg.fontsize,
            "axes.linewidth": cfg.edgewidth,
        }
    )

    n_apps = len(benchmarks)
    n_back = len(backends)
    group_x = np.arange(n_apps)

    if n_back == 1:
        bar_w = 0.6
        offsets = np.array([0.0])
    else:
        group_width = 0.82
        bar_w = group_width / n_back
        offsets = (np.arange(n_back) - (n_back - 1) / 2.0) * bar_w

    # Backend colors (distinct architectures)
    backend_colors = {
        "CUDA":"#76B900",   # green
        "HIP": "#9C2A44",
        "SCALE": "#2B1D4F",  # purple
        "SYCL": "#F35A1C",   # red
    }
    backend_order = [b for b, _ in backends]
    b_colors = [backend_colors.get(b, "#7f7f7f") for b in backend_order]

    # Metric patterns (keep patterns for METRICS)
    metric_hatches = [
        cfg.patterns[2],  # "" (Allocation)
        cfg.patterns[0],  # ///// (H2D)
        cfg.patterns[3],  # |||||| (Compute)
        cfg.patterns[1],  # xxxx (D2H)
        cfg.patterns[4],  # ----- (Free)
    ]

    # Slightly different alpha per metric to help readability while keeping patterns
    metric_alphas = [0.35, 0.55, 0.75, 0.65, 0.45]

    edgecolor = cfg.edgecolor
    lw = cfg.edgewidth

    # Squash overlay styling + legend entry
    squash_face = "#000000"
    squash_alpha = 0.18
    squash_hatch = "..."
    squash_label = f"Outlier segment shown as {args.squash_to:g} ms"
    squash_handle_added = False

    # Draw grouped stacked bars
    for bi, (backend, _) in enumerate(backends):
        df_plot = plot_dfs[backend]
        df_raw = dfs[backend]

        x = group_x + offsets[bi]
        bottom = np.zeros(n_apps, dtype=float)

        base_color = b_colors[bi]

        for mi, metric in enumerate(METRICS):
            vals_plot = df_plot[metric].to_numpy(dtype=float)
            vals_raw = df_raw[metric].to_numpy(dtype=float)

            metric_label = metric if bi == 0 else None

            ax.bar(
                x,
                vals_plot,
                width=bar_w * 0.95,
                bottom=bottom,
                color=base_color,
                alpha=metric_alphas[mi],
                edgecolor=edgecolor,
                linewidth=lw,
                hatch=metric_hatches[mi],
                label=metric_label,
                zorder=3,
            )

            # Squash overlay for altered segments
            if args.squash_outliers:
                mask = (vals_raw != vals_plot) & (vals_plot > 0)
                if np.any(mask):
                    of_label = squash_label if not squash_handle_added else None
                    squash_handle_added = True
                    ax.bar(
                        x[mask],
                        vals_plot[mask],
                        width=bar_w * 0.95,
                        bottom=bottom[mask],
                        color=squash_face,
                        alpha=squash_alpha,
                        edgecolor=edgecolor,
                        linewidth=lw,
                        hatch=squash_hatch,
                        label=of_label,
                        zorder=5,
                    )

            bottom += vals_plot

    ax.set_ylabel("Execution Time (ms)")
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)

    ax.set_xticks(group_x)
    ax.set_xticklabels(
        benchmarks, rotation=args.xtick_rotation, ha=args.xtick_ha, rotation_mode="anchor"
    )

    if args.title:
        ax.set_title(args.title)

    # Log scale formatting (optional)
    if args.logy:
        if args.logy_min <= 0:
            raise SystemExit("--logy-min must be > 0 for log scale.")
        ax.set_yscale("log")
        ax.set_ylim(args.logy_min, None)

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Legends
    # 1) Breakdown legend (patterns)
    breakdown_handles = [
        Patch(facecolor="#ffffff", edgecolor=edgecolor, hatch=metric_hatches[i], label=METRICS[i])
        for i in range(len(METRICS))
    ]
    if args.squash_outliers:
        breakdown_handles.append(
            Patch(facecolor=squash_face, alpha=squash_alpha, edgecolor=edgecolor, hatch=squash_hatch, label=squash_label)
        )

    # 2) Backend legend (colors)
    backend_handles = [
        Patch(facecolor=backend_colors.get(b, "#7f7f7f"), edgecolor=edgecolor, label=b)
        for b in backend_order
    ]

    leg1 = fig.legend(
        handles=breakdown_handles,
        loc="upper left",
        bbox_to_anchor=(0.85, 0.98),
        frameon=False,
        borderaxespad=0.0,
        title="Breakdown",
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=backend_handles,
        loc="upper left",
        bbox_to_anchor=(0.85, 0.48),
        frameon=False,
        borderaxespad=0.0,
        title="Backend",
    )

    # Layout
    fig.subplots_adjust(bottom=0.22, right=0.85)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote: {args.out}")

    # Squash notes
    if args.squash_outliers:
        if not squashed:
            print("[INFO] --squash-outliers enabled, but no segments met the outlier condition.")
        else:
            print("[NOTE] Squashed extreme outlier segment(s) for readability:")
            for s in squashed:
                print(
                    f"  - {s['backend']} / {s['benchmark']} / {s['metric']}: "
                    f"{s['original_ms']:.2f} ms (2nd={s['second_ms']:.2f} ms) -> {s['squashed_to_ms']:.2f} ms "
                    f"(condition: max > {args.squash_factor:g}×second and max >= {args.squash_abs_min:g} ms)"
                )


if __name__ == "__main__":
    main()
