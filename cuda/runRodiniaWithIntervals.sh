#!/bin/bash
# Drop-in replacement for runRodinia.sh
# Adds:
#  - RUNS (default 5): number of full-suite iterations
#  - SLEEP_SECS (default 0): sleep between iterations (seconds)
#  - GPU_ID (optional): sets CUDA_VISIBLE_DEVICES to pick a GPU
#  - Writes per-iteration CSV + summary CSV (min/max/avg over iterations)
#  - Examples of usage:
#    - 5 iterations with 10 minutes interval, cuda, GPU id 0:  LABEL=cuda GPU_ID=0 RUNS=5 SLEEP_SECS=600 ./runRodiniaWithIntervals.sh
#    - select GPU 1:                           GPU_ID=1 ./runRodiniaWithIntervals.sh
#    - label outputs (cuda/hip/scale):         LABEL=cuda GPU_ID=0 RUNS=5 SLEEP_SECS=600 ./runRodiniaWithIntervals.sh

set -u
ROOT_DIR="$(pwd)"
benchmarks=("backprop" "bfs" "b+tree" "cfd" "dwt2d" "gaussian" "heartwall" "hotspot" "hotspot3D" "huffman" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
total_benchmarks=${#benchmarks[@]}

# Config (override via env vars)
RUNS=${RUNS:-5}
SLEEP_SECS=${SLEEP_SECS:-0}
OUT_DIR=${OUT_DIR:-results}
OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
LABEL=${LABEL:-}     # optional: cuda/hip/scale/etc
GPU_ID=${GPU_ID:-}   # optional: 0,1,2,...

mkdir -p "${OUT_DIR}"

# Optional GPU selection (filters/renumbers devices)
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "[ GPU SELECTED       ] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${LABEL}" ]]; then
  base="${OUT_DIR}/rodinia_${LABEL}_${timestamp}"
else
  base="${OUT_DIR}/rodinia_${timestamp}"
fi

per_iter_csv="${base}_per_iteration.csv"
summary_csv="${base}_summary.csv"

echo "iteration,benchmark,avg_ms" > "${per_iter_csv}"

echo "[===================]"

extract_avg_ms() {
  if [[ -f "average.csv" ]]; then
    grep -m1 "Computation" "average.csv" | awk -F',' '{printf "%.2f", $2}'
  else
    echo ""
  fi
}

for ((iter=1; iter<=RUNS; iter++)); do
  echo "[ ITERATION         ] ${iter}/${RUNS}"

  for str in "${benchmarks[@]}"; do
    echo "[ RUN               ] $str"
    cd "${str}" || { echo "ERROR: cannot cd into ${str}" >&2; continue; }

    path="$(pwd)"

    # Keep benchmark output as-is; don't stop the suite on failures
    ./run.sh "${path}" "${str}.csv" || true

    value="$(extract_avg_ms)"

    if [[ -n "${value}" ]]; then
      echo "[   AVG GPU time    ] $value ms"
      echo "${iter},${str},${value}" >> "${per_iter_csv}"
    else
      echo "[   AVG GPU time    ] NA ms"
      echo "${iter},${str}," >> "${per_iter_csv}"
    fi

    cd - &>/dev/null || true
  done

  if [[ "${iter}" -lt "${RUNS}" && "${SLEEP_SECS}" -gt 0 ]]; then
    echo "[ SLEEP             ] ${SLEEP_SECS} seconds"
    sleep "${SLEEP_SECS}"
  fi
done

echo "[===================] $total_benchmarks apps ran."

# Summary CSV: min/max/avg across iterations for each benchmark
echo "benchmark,min_ms,max_ms,avg_ms,samples" > "${summary_csv}"

for str in "${benchmarks[@]}"; do
  awk -F',' -v b="${str}" '
    NR==1 { next }                # skip header
    $2==b && $3!="" {
      v=$3+0.0
      if (!seen) { min=v; max=v; seen=1 }
      if (v<min) min=v
      if (v>max) max=v
      sum+=v
      cnt+=1
    }
    END {
      if (cnt>0) {
        printf "%s,%.2f,%.2f,%.2f,%d\n", b, min, max, sum/cnt, cnt
      } else {
        printf "%s,,,,0\n", b
      }
    }
  ' "${per_iter_csv}" >> "${summary_csv}"
done

echo "[ PER-ITER CSV      ] ${per_iter_csv}"
echo "[ SUMMARY CSV       ] ${summary_csv}"
