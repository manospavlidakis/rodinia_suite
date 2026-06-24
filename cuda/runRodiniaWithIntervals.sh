#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Rodinia runner (CUDA)

Usage:
  ./runRodiniaWithIntervals.sh [--help]

This script is env-var driven (no positional args).
By default it does NOT re-run benchmarks; it parses existing per-app statistics CSVs
(average.csv, min.csv, max.csv, median.csv, iqr.csv, std_dev.csv).

Environment variables:
  RERUN_APPS=0|1        0=reuse existing outputs (default), 1=run ./run.sh per app per iteration
  DO_BREAKDOWNS=0|1     0=skip breakdown aggregation (default), 1=aggregate breakdown CSVs
  RUNS=<N>              Number of iterations (default: 5)
  SLEEP_SECS=<secs>     Sleep between iterations (default: 10; only used when RERUN_APPS=1)
  OUT_DIR=<dir>         Output dir name (default: results)
  LABEL=<str>           Optional label added to output filenames (e.g., cuda/scale)

GPU selection:
  NVIDIA_GPU_ID=<id>    Sets CUDA_VISIBLE_DEVICES=<id>   (preferred for CUDA runner)
  GPU_ID=<id>           Back-compat alias for CUDA_VISIBLE_DEVICES=<id>
  AMD_GPU_ID=<id>       Ignored by CUDA runner (use the HIP runner instead)

Examples:
  # Default: parse existing average.csv only (no reruns)
  ./runRodiniaWithIntervals.sh

  # Also aggregate breakdowns (still no reruns)
  DO_BREAKDOWNS=1 ./runRodiniaWithIntervals.sh

  # Re-run all benchmarks 5 times, sleep 10 minutes between iterations, GPU 0, plus breakdowns
  LABEL=cuda NVIDIA_GPU_ID=0 RUNS=5 SLEEP_SECS=600 RERUN_APPS=1 DO_BREAKDOWNS=1 ./runRodiniaWithIntervals.sh

  # Back-compat GPU selector
  GPU_ID=1 ./runRodiniaWithIntervals.sh

  # What we report:
  - min = minimum of the 5 per-round mins
  - max = maximum of the 5 per-round maxes
  - avg = average of the 5 per-round averages
  - median = average of the 5 per-round medians
  - iqr = average of the 5 per-round IQRs
  - std_dev = average of the 5 per-round std_devs
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
SLEEP_SECS=${SLEEP_SECS:-10}
OUT_DIR=${OUT_DIR:-results}
OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
LABEL=${LABEL:-}

# Back-compat GPU selector
GPU_ID=${GPU_ID:-}

# New per-vendor selectors
NVIDIA_GPU_ID=${NVIDIA_GPU_ID:-}
AMD_GPU_ID=${AMD_GPU_ID:-}

RERUN_APPS=${RERUN_APPS:-0}         # 0=reuse (default), 1=rerun ./run.sh
DO_BREAKDOWNS=${DO_BREAKDOWNS:-0}   # 0=skip breakdown aggregation (default), 1=aggregate breakdowns

# Correctness check (VERIFY=1), per app, in priority order:
#   1) GOLDEN compare — the app writes result.txt; nat_result.txt is the committed
#      native reference. Compare them (default exact `diff -q`); a mismatch means
#      haishare produced a different result than native.
#   2) fallback (no nat_result.txt) — scan the app's own self-verify output for FAIL.
#   plus the run.sh exit code in either case.
# Surfaces a silently-wrong haishare result instead of hiding it behind the timings.
#   VERIFY_DIFF_CMD  comparator for result.txt vs nat_result.txt (default 'diff -q';
#                    set to a tolerance script if an FP benchmark diffs spuriously)
#   VERIFY_FAIL_RE   failure regex for the self-verify-output fallback
VERIFY=${VERIFY:-1}
VERIFY_DIFF_CMD="${VERIFY_DIFF_CMD:-diff -q}"
VERIFY_FAIL_RE=${VERIFY_FAIL_RE:-'FAILED|Test failed|verification failed|do(es)? not match|mismatch|incorrect|wrong result|illegal memory|Segmentation fault|core dumped|terminate called|CUDA[ _]error|cudaError|out of memory'}
VERIFY_FAILS=0

mkdir -p "${OUT_DIR}"

# Vendor-aware GPU selection
# Precedence: NVIDIA_GPU_ID > GPU_ID
if [[ -n "${AMD_GPU_ID}" ]]; then
  echo "[ WARN              ] AMD_GPU_ID is set (${AMD_GPU_ID}) but this is the CUDA runner; ignoring." >&2
fi

if [[ -n "${NVIDIA_GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_GPU_ID}"
  echo "[ GPU SELECTED       ] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (from NVIDIA_GPU_ID)"
elif [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "[ GPU SELECTED       ] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (from GPU_ID)"
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

echo "iteration,benchmark,avg_ms,min_ms,max_ms,median_ms,iqr_ms,std_dev_ms" > "${per_iter_csv}"
verify_csv="${base}_verify.csv"
[[ "${VERIFY}" == "1" ]] && echo "iteration,benchmark,run_rc,verify" > "${verify_csv}"
if [[ "${DO_BREAKDOWNS}" == "1" ]]; then
  echo "iteration,benchmark,metric,avg_ms" > "${break_iter_csv}"
fi

echo "[===================]"
echo "[ MODE              ] RERUN_APPS=${RERUN_APPS} (0=reuse outputs, 1=run apps)"
echo "[ BREAKDOWNS        ] DO_BREAKDOWNS=${DO_BREAKDOWNS} (0=skip, 1=aggregate)"

extract_metric_ms() {
  local file="$1"
  local metric="$2"

  if [[ -f "${file}" ]]; then
    awk -F',' -v metric="${metric}" '
      NR > 1 {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if ($1 == metric && $2 != "") {
          printf "%.2f", $2
          exit
        }
      }
    ' "${file}"
  else
    echo ""
  fi
}

extract_avg_ms()    { extract_metric_ms "average.csv"  "Computation"; }
extract_min_ms()    { extract_metric_ms "min.csv"      "Computation"; }
extract_max_ms()    { extract_metric_ms "max.csv"      "Computation"; }
extract_median_ms() { extract_metric_ms "median.csv"   "Computation"; }
extract_iqr_ms()    { extract_metric_ms "iqr.csv"      "Computation"; }
extract_stddev_ms() { extract_metric_ms "std_dev.csv"  "Computation"; }

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

    rc_run=0
    if [[ "${RERUN_APPS}" == "1" ]]; then
      ./run.sh "${path}" "${b}.csv"; rc_run=$?
    fi

    # Correctness check (still cd'd in the app's dir): prefer the GOLDEN compare
    # (result.txt vs nat_result.txt); fall back to scanning the app's own self-verify
    # output (<size>_<iter>_<b>.csv). run.sh exit code (rc_run) gates both.
    if [[ "${VERIFY}" == "1" && "${RERUN_APPS}" == "1" ]]; then
      if [[ "${rc_run}" -ne 0 ]]; then
        verify="FAIL"
      elif [[ -f nat_result.txt ]]; then
        if [[ ! -f result.txt ]]; then
          verify="NORESULT"
        elif ${VERIFY_DIFF_CMD} result.txt nat_result.txt >/dev/null 2>&1; then
          verify="OK"
        else
          verify="FAIL"
        fi
      else
        shopt -s nullglob; verify_logs=( *_"${b}.csv" ); shopt -u nullglob
        if [[ "${#verify_logs[@]}" -eq 0 ]]; then
          verify="NOLOG"
        elif grep -qiE "${VERIFY_FAIL_RE}" "${verify_logs[@]}" 2>/dev/null; then
          verify="FAIL"
        else
          verify="OK"
        fi
      fi
      echo "[   VERIFY          ] ${verify} (run_rc=${rc_run})"
      echo "${iter},${b},${rc_run},${verify}" >> "${verify_csv}"
      [[ "${verify}" == "FAIL" ]] && VERIFY_FAILS=$((VERIFY_FAILS + 1))
    fi

    avg_val="$(extract_avg_ms)"
    min_val="$(extract_min_ms)"
    max_val="$(extract_max_ms)"
    median_val="$(extract_median_ms)"
    iqr_val="$(extract_iqr_ms)"
    stddev_val="$(extract_stddev_ms)"

    echo "[   AVG GPU time    ] ${avg_val:-NA} ms"
    echo "[   MIN GPU time    ] ${min_val:-NA} ms"
    echo "[   MAX GPU time    ] ${max_val:-NA} ms"
    echo "[   MEDIAN GPU time ] ${median_val:-NA} ms"
    echo "[   IQR GPU time    ] ${iqr_val:-NA} ms"
    echo "[   STDDEV GPU time ] ${stddev_val:-NA} ms"

    echo "${iter},${b},${avg_val},${min_val},${max_val},${median_val},${iqr_val},${stddev_val}" >> "${per_iter_csv}"  # CHANGED: store all stats per iteration
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

if [[ "${VERIFY}" == "1" && "${RERUN_APPS}" == "1" ]]; then
  echo "[ VERIFY            ] ${VERIFY_FAILS} app-iteration(s) FAILED correctness (see ${verify_csv})"
  if [[ "${VERIFY_FAILS}" -gt 0 ]]; then
    awk -F, 'NR>1 && $4!="OK"{printf "    %-6s bench=%s iter=%s run_rc=%s\n",$4,$2,$1,$3}' "${verify_csv}"
  fi
fi

echo "benchmark,min_ms,max_ms,avg_ms,median_ms,iqr_ms,std_dev_ms,samples" > "${summary_csv}"  # CHANGED: expanded summary CSV schema
for b in "${benchmarks[@]}"; do
  awk -F',' -v bench="${b}" '
    NR==1 { next }
    $2==bench {
      if ($3 != "") {
        avg_v = $3 + 0.0
        avg_sum += avg_v
        avg_cnt += 1
      }
      if ($4 != "") {
        min_v = $4 + 0.0
        if (!min_seen || min_v < min_all) {
          min_all = min_v
          min_seen = 1
        }
      }
      if ($5 != "") {
        max_v = $5 + 0.0
        if (!max_seen || max_v > max_all) {
          max_all = max_v
          max_seen = 1
        }
      }
      if ($6 != "") {
        median_v = $6 + 0.0
        median_sum += median_v
        median_cnt += 1
      }
      if ($7 != "") {
        iqr_v = $7 + 0.0
        iqr_sum += iqr_v
        iqr_cnt += 1
      }
      if ($8 != "") {
        std_v = $8 + 0.0
        std_sum += std_v
        std_cnt += 1
      }
    }
    END {
      printf "%s,", bench

      if (min_seen)     printf "%.2f,", min_all; else printf ","
      if (max_seen)     printf "%.2f,", max_all; else printf ","
      if (avg_cnt>0)    printf "%.2f,", avg_sum/avg_cnt; else printf ","
      if (median_cnt>0) printf "%.2f,", median_sum/median_cnt; else printf ","
      if (iqr_cnt>0)    printf "%.2f,", iqr_sum/iqr_cnt; else printf ","
      if (std_cnt>0)    printf "%.2f,", std_sum/std_cnt; else printf ","

      samples = avg_cnt
      printf "%d\n", samples
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
