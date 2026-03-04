#!/usr/bin/env bash
set -euo pipefail

# Normal (linear y-axis)
#./run_plot_breakdowns.sh
# Log y-axis
#LOGY=1 ./run_plot_breakdowns.sh
# Log y-axis with smaller minimum (helps show tiny bars)
#LOGY=1 LOGY_MIN=1e-3 ./run_plot_breakdowns.sh
# Squash extreme outliers (non-log) so the rest is readable:
#SQUASH_OUTLIERS=1 ./run_plot_breakdowns.sh
# Tune squashing:
#SQUASH_OUTLIERS=1 SQUASH_FACTOR=6 SQUASH_ABS_MIN=800 SQUASH_TO=1 ./run_plot_breakdowns.sh

# Hardcoded results dirs (relative to where you run the script from)
CUDA_DIR="cuda/results"
HIP_DIR="hipify_cuda_2/results"
SCALE_DIR="scale/results"
SYCL_DIR="sycl/results"

OUT="${OUT:-breakdowns_stacked.png}"
TITLE="${TITLE:-}"

# Optional: log-scale y-axis
LOGY="${LOGY:-0}"
LOGY_MIN="${LOGY_MIN:-}"

# Optional: squash extreme outlier segments (linear axis)
SQUASH_OUTLIERS="${SQUASH_OUTLIERS:-0}"
SQUASH_FACTOR="${SQUASH_FACTOR:-}"
SQUASH_ABS_MIN="${SQUASH_ABS_MIN:-}"
SQUASH_TO="${SQUASH_TO:-}"

find_latest() {
  local dir="$1"
  local pattern="$2"

  if [[ ! -d "$dir" ]]; then
    echo "[ MISSING DIR ] $dir" >&2
    return 1
  fi

  local f
  f=$(ls -t "$dir"/$pattern 2>/dev/null | head -n1 || true)
  if [[ -z "${f}" ]]; then
    echo "[ NO FILES    ] $dir/$pattern" >&2
    return 1
  fi

  echo "$f"
  return 0
}

CUDA_FILE="$(find_latest "$CUDA_DIR" "*_breakdowns_summary.csv" || true)"
HIP_FILE="$(find_latest "$HIP_DIR" "*_breakdowns_summary.csv" || true)"
SCALE_FILE="$(find_latest "$SCALE_DIR" "*_breakdowns_summary.csv" || true)"
SYCL_FILE="$(find_latest "$SYCL_DIR" "*_breakdowns_summary.csv" || true)"

cmd=(python3 plot_breakdowns_stacked.py -o "$OUT")
if [[ -n "$TITLE" ]]; then cmd+=(--title "$TITLE"); fi

# logy toggle
if [[ "$LOGY" == "1" || "$LOGY" == "true" || "$LOGY" == "TRUE" ]]; then
  cmd+=(--logy)
  if [[ -n "${LOGY_MIN}" ]]; then
    cmd+=(--logy-min "${LOGY_MIN}")
  fi
fi

# squash toggle (alternative to logy)
if [[ "$SQUASH_OUTLIERS" == "1" || "$SQUASH_OUTLIERS" == "true" || "$SQUASH_OUTLIERS" == "TRUE" ]]; then
  cmd+=(--squash-outliers)
  if [[ -n "${SQUASH_FACTOR}" ]]; then cmd+=(--squash-factor "${SQUASH_FACTOR}"); fi
  if [[ -n "${SQUASH_ABS_MIN}" ]]; then cmd+=(--squash-abs-min "${SQUASH_ABS_MIN}"); fi
  if [[ -n "${SQUASH_TO}" ]]; then cmd+=(--squash-to "${SQUASH_TO}"); fi
fi

n=0
if [[ -n "${CUDA_FILE}" ]]; then
  echo "[ CUDA  ] ${CUDA_FILE}"
  cmd+=(--cuda "${CUDA_FILE}")
  n=$((n+1))
fi
if [[ -n "${HIP_FILE}" ]]; then
  echo "[ HIP   ] ${HIP_FILE}"
  cmd+=(--hip "${HIP_FILE}")
  n=$((n+1))
fi
if [[ -n "${SCALE_FILE}" ]]; then
  echo "[ SCALE ] ${SCALE_FILE}"
  cmd+=(--scale "${SCALE_FILE}")
  n=$((n+1))
fi
if [[ -n "${SYCL_FILE}" ]]; then
  echo "[ SYCL  ] ${SYCL_FILE}"
  cmd+=(--sycl "${SYCL_FILE}")
  n=$((n+1))
fi

if [[ "$n" -eq 0 ]]; then
  echo "[ ERROR ] No breakdown summary CSVs found in any of:" >&2
  echo "          $CUDA_DIR" >&2
  echo "          $HIP_DIR" >&2
  echo "          $SCALE_DIR" >&2
  echo "          $SYCL_DIR" >&2
  exit 1
fi

echo "[ RUN   ] ${cmd[*]}"
"${cmd[@]}"
echo "[ DONE  ] Wrote ${OUT}"
