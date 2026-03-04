#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Rodinia runner (HIP)

Usage:
  ./runRodiniaWithIntervals.sh [--help]

This script is env-var driven (no positional args).
By default it does NOT re-run benchmarks; it only parses existing per-app outputs (average.csv).

Environment variables:
  RERUN_APPS=0|1        0=reuse existing outputs (default), 1=run ./run.sh per app per iteration
  DO_BREAKDOWNS=0|1     0=skip breakdown aggregation (default), 1=aggregate breakdown CSVs
  RUNS=<N>              Number of iterations (default: 5)
  SLEEP_SECS=<secs>     Sleep between iterations (default: 0; only used when RERUN_APPS=1)
  GPU_ID=<id>           Sets HIP_VISIBLE_DEVICES=<id>
  OUT_DIR=<dir>         Output dir name (default: results)
  LABEL=<str>           Optional label added to output filenames (e.g., hip/scale)

Examples:
  # Default: parse existing average.csv only (no reruns)
  ./runRodiniaWithIntervals.sh

  # Also aggregate breakdowns (still no reruns)
  DO_BREAKDOWNS=1 ./runRodiniaWithIntervals.sh

  # Re-run all benchmarks 5 times, sleep 10 minutes between iterations, GPU 0, plus breakdowns
  LABEL=hip GPU_ID=0 RUNS=5 SLEEP_SECS=600 RERUN_APPS=1 DO_BREAKDOWNS=1 ./runRodiniaWithIntervals.sh

  # Select GPU 1
  GPU_ID=1 ./runRodiniaWithIntervals.sh
EOF
}

# Help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "ERROR: This script does not accept positional arguments." >&2
  echo "Run with --help." >&2
  exit 2
fi

ROOT_DIR="$(pwd)"
benchmarks=("backprop" "bfs" "b+tree" "cfd" "dwt2d" "gaussian" "heartwall" "hotspot" "hotspot3D" "huffman" "lavaMD" "nn" "nw" "pathfinder")
total_benchmarks=${#benchmarks[@]}

# Config (override via env vars)
RUNS=${RUNS:-5}
SLEEP_SECS=${SLEEP_SECS:-0}
OUT_DIR=${OUT_DIR:-results}
OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
LABEL=${LABEL:-}
GPU_ID=${GPU_ID:-}

RERUN_APPS=${RERUN_APPS:-0}         # 0=reuse (default), 1=rerun ./run.sh
DO_BREAKDOWNS=${DO_BREAKDOWNS:-0}   # 0=skip breakdown aggregation (default), 1=aggregate breakdowns

mkdir -p "${OUT_DIR}"

if [[ -n "${GPU_ID}" ]]; then
  export HIP_VISIBLE_DEVICES="${GPU_ID}"
  echo "[ GPU SELECTED       ] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${LABEL}" ]]; then
  base="${OUT_DIR}/rodinia_${LABEL}_${timestamp}"
else
  base="${OUT_DIR}/rodinia_${timestamp}"
fi

per_iter_csv="${base}_per_iteration.csv"
summary_csv="${base}_summary.csv"

break_iter_csv="${base}_breakdowns_per_iteration.csv"
break_summary_csv="${base}_breakdowns_summary.csv"

echo "iteration,benchmark,avg_ms" > "${per_iter_csv}"
if [[ "${DO_BREAKDOWNS}" == "1" ]]; then
  echo "iteration,benchmark,metric,avg_ms" > "${break_iter_csv}"
fi

echo "[===================]"
echo "[ MODE              ] RERUN_APPS=${RERUN_APPS} (0=reuse outputs, 1=run apps)"
echo "[ BREAKDOWNS        ] DO_BREAKDOWNS=${DO_BREAKDOWNS} (0=skip, 1=aggregate)"

extract_avg_ms() {
  if [[ -f "average.csv" ]]; then
    grep -m1 "Computation" "average.csv" | awk -F',' '{printf "%.2f", $2}'
  else
    echo ""
  fi
}

extract_breakdowns() {
  if [[ -f "average_breakdowns.csv" ]]; then
    tail -n +2 "average_breakdowns.csv" | awk -F',' '
      NF>=2 {
        gsub(/^[ \t]+|[ \t]+$/, "", $1);
        gsub(/^[ \t]+|[ \t]+$/, "", $2);
        if ($1 != "") print $1 "," $2;
      }'
  fi
}

for ((iter=1; iter<=RUNS; iter++)); do
  echo "[ ITERATION         ] ${iter}/${RUNS}"

  for b in "${benchmarks[@]}"; do
    echo "[ RUN               ] $b"
    cd "${b}" || { echo "ERROR: cannot cd into ${b}" >&2; continue; }

    path="$(pwd)"

    if [[ "${RERUN_APPS}" == "1" ]]; then
      ./run.sh "${path}" "${b}.csv" || true
    fi

    value="$(extract_avg_ms)"
    if [[ -n "${value}" ]]; then
      echo "[   AVG GPU time    ] $value ms"
      echo "${iter},${b},${value}" >> "${per_iter_csv}"
    else
      echo "[   AVG GPU time    ] NA ms"
      echo "${iter},${b}," >> "${per_iter_csv}"
    fi

    if [[ "${DO_BREAKDOWNS}" == "1" ]]; then
      if [[ -x "${ROOT_DIR}/find_avg_per_app_break.py" ]]; then
        "${ROOT_DIR}/find_avg_per_app_break.py" "${b}" || true
        while IFS= read -r line; do
          metric="${line%,*}"
          mval="${line#*,}"
          echo "${iter},${b},${metric},${mval}" >> "${break_iter_csv}"
        done < <(extract_breakdowns)
      else
        echo "[ BREAKDOWNS        ] missing ${ROOT_DIR}/find_avg_per_app_break.py (skipping)"
      fi
    fi

    cd - &>/dev/null || true
  done

  if [[ "${iter}" -lt "${RUNS}" && "${SLEEP_SECS}" -gt 0 && "${RERUN_APPS}" == "1" ]]; then
    echo "[ SLEEP             ] ${SLEEP_SECS} seconds"
    sleep "${SLEEP_SECS}"
  fi
done

echo "[===================] $total_benchmarks apps processed."

echo "benchmark,min_ms,max_ms,avg_ms,samples" > "${summary_csv}"
for b in "${benchmarks[@]}"; do
  awk -F',' -v bench="${b}" '
    NR==1 { next }
    $2==bench && $3!="" {
      v=$3+0.0
      if (!seen) { min=v; max=v; seen=1 }
      if (v<min) min=v
      if (v>max) max=v
      sum+=v
      cnt+=1
    }
    END {
      if (cnt>0) printf "%s,%.2f,%.2f,%.2f,%d\n", bench, min, max, sum/cnt, cnt;
      else       printf "%s,,,,0\n", bench;
    }
  ' "${per_iter_csv}" >> "${summary_csv}"
done

if [[ "${DO_BREAKDOWNS}" == "1" ]]; then
  echo "benchmark,Allocation time,H2D transfer time,Compute time,D2H transfer time,Free time,samples" > "${break_summary_csv}"

  for b in "${benchmarks[@]}"; do
    awk -F',' -v bench="${b}" '
      BEGIN { a=0;h=0;c=0;d=0;f=0; ac=0;hc=0;cc=0;dc=0;fc=0; }
      NR==1 { next }
      $2==bench && $4!="" {
        m=$3; v=$4+0.0
        if (m=="Allocation time")       { a+=v; ac++ }
        else if (m=="H2D transfer time"){ h+=v; hc++ }
        else if (m=="Compute time")     { c+=v; cc++ }
        else if (m=="D2H transfer time"){ d+=v; dc++ }
        else if (m=="Free time")        { f+=v; fc++ }
      }
      END {
        s=ac; if (hc>s) s=hc; if (cc>s) s=cc; if (dc>s) s=dc; if (fc>s) s=fc;

        printf "%s,", bench
        if (ac>0) printf "%.2f,", a/ac; else printf ","
        if (hc>0) printf "%.2f,", h/hc; else printf ","
        if (cc>0) printf "%.2f,", c/cc; else printf ","
        if (dc>0) printf "%.2f,", d/dc; else printf ","
        if (fc>0) printf "%.2f,", f/fc; else printf ","
        printf "%d\n", s
      }
    ' "${break_iter_csv}" >> "${break_summary_csv}"
  done
fi

echo "[ PER-ITER CSV      ] ${per_iter_csv}"
echo "[ SUMMARY CSV       ] ${summary_csv}"
if [[ "${DO_BREAKDOWNS}" == "1" ]]; then
  echo "[ BREAK ITER CSV    ] ${break_iter_csv}"
  echo "[ BREAK SUMMARY CSV ] ${break_summary_csv}"
fi
