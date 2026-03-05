#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the Rodinia suite end-to-end for CUDA + HIP backends (from rodinia_suite/).

What it does:
  1) Build CUDA + HIP without breakdowns
  2) Run CUDA + HIP (produces results/ inside each backend dir)
  3) Plot computation bars (uses results inside cuda/ and hipify_cuda_2/)
  4) Rebuild CUDA + HIP with breakdowns enabled
  5) Re-run CUDA + HIP with breakdowns enabled
  6) Plot breakdown stacked bars (uses results inside cuda/ and hipify_cuda_2/)

Environment variables:
  NVIDIA_GPU_ID=<id>   NVIDIA index for CUDA runs (mapped to GPU_ID for cuda/runRodiniaWithIntervals.sh)
  AMD_GPU_ID=<id>      AMD index for HIP runs (passed to hipify_cuda_2/runRodiniaWithIntervals.sh as AMD_GPU_ID)

  RUNS=<N>             Number of iterations per backend run (default: 5)
  SLEEP_SECS=<secs>    Sleep between iterations (default: 0) (only relevant when rerunning apps)
  OUT_DIR=<dir>        Results directory name inside each backend (default: results)

Notes:
  - This script always sets RERUN_APPS=1 (it actually runs benchmarks).
  - Breakdown runs set DO_BREAKDOWNS=1.
  - Build scripts are called with --no-breakdowns / --breakdowns.

Examples:
  NVIDIA_GPU_ID=0 AMD_GPU_ID=1 ./run_all.sh
  NVIDIA_GPU_ID=0 AMD_GPU_ID=1 RUNS=3 SLEEP_SECS=60 ./run_all.sh

How to find GPU IDs (recommended):
  NVIDIA: nvidia-smi -L
  AMD:    rocm-smi --showid

Optional device listing:
  ./run_all.sh --list-gpus
EOF
}

list_gpus() {
  echo "=== NVIDIA (CUDA) ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "nvidia-smi not found."
  fi
  echo
  echo "=== AMD (ROCm/HIP) ==="
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid
  else
    echo "rocm-smi/rocminfo not found."
  fi
}

# Argument handling
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --list-gpus)
    list_gpus
    exit 0
    ;;
  "")
    ;;
  *)
    echo "ERROR: unknown argument: $1" >&2
    echo "Run with --help." >&2
    exit 2
    ;;
esac

ROOT_DIR="$(pwd)"
CUDA_DIR="${ROOT_DIR}/cuda"
HIP_DIR="${ROOT_DIR}/hipify_cuda_2"

RUNS="${RUNS:-5}"
SLEEP_SECS="${SLEEP_SECS:-0}"
OUT_DIR="${OUT_DIR:-results}"

NVIDIA_GPU_ID="${NVIDIA_GPU_ID:-}"
AMD_GPU_ID="${AMD_GPU_ID:-}"

if [[ ! -d "${CUDA_DIR}" ]]; then
  echo "[ ERROR ] Missing directory: ${CUDA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${HIP_DIR}" ]]; then
  echo "[ ERROR ] Missing directory: ${HIP_DIR}" >&2
  exit 1
fi

if [[ -z "${NVIDIA_GPU_ID}" ]]; then
  echo "[ WARN  ] NVIDIA_GPU_ID is not set. CUDA runs will use default device visibility." >&2
fi
if [[ -z "${AMD_GPU_ID}" ]]; then
  echo "[ WARN  ] AMD_GPU_ID is not set. HIP runs will use default device visibility." >&2
fi

run_in_dir() {
  local dir="$1"; shift
  ( cd "$dir" && "$@" )
}

echo "[===================]"
echo "[ CUDA DIR          ] ${CUDA_DIR}"
echo "[ HIP DIR           ] ${HIP_DIR}"
echo "[ RUNS              ] ${RUNS}"
echo "[ SLEEP_SECS        ] ${SLEEP_SECS}"
echo "[ OUT_DIR           ] ${OUT_DIR}"
echo "[ NVIDIA_GPU_ID     ] ${NVIDIA_GPU_ID:-<unset>}"
echo "[ AMD_GPU_ID        ] ${AMD_GPU_ID:-<unset>}"
echo "[===================]"

################################################################################
# 1) Build without breakdowns
################################################################################
echo "[ STEP 1 ] Build CUDA (no breakdowns)"
#run_in_dir "${CUDA_DIR}" ./buildall.sh --no-breakdowns

echo "[ STEP 1 ] Build HIP (no breakdowns)"
#run_in_dir "${HIP_DIR}" ./buildall.sh --no-breakdowns

################################################################################
# 2) Run without breakdowns
################################################################################
echo "[ STEP 2 ] Run CUDA (no breakdowns)"
#run_in_dir "${CUDA_DIR}" env \
#  RERUN_APPS=1 DO_BREAKDOWNS=0 LABEL=cuda OUT_DIR="${OUT_DIR}" \
#  RUNS="${RUNS}" SLEEP_SECS="${SLEEP_SECS}" \
#  GPU_ID="${NVIDIA_GPU_ID}" \
#  ./runRodiniaWithIntervals.sh

echo "[ STEP 2 ] Run HIP (no breakdowns)"
run_in_dir "${HIP_DIR}" env \
  RERUN_APPS=1 DO_BREAKDOWNS=0 LABEL=hip OUT_DIR="${OUT_DIR}" \
  RUNS="${RUNS}" SLEEP_SECS="${SLEEP_SECS}" \
  AMD_GPU_ID="${AMD_GPU_ID}" \
  ./runRodiniaWithIntervals.sh

################################################################################
# 3) Plot computation
################################################################################
echo "[ STEP 3 ] Plot computation (reads ${CUDA_DIR}/${OUT_DIR} and ${HIP_DIR}/${OUT_DIR})"
cd "${ROOT_DIR}"
./run_plot_computation.sh

################################################################################
# 4) Build with breakdowns
################################################################################
echo "[ STEP 4 ] Build CUDA (with breakdowns)"
run_in_dir "${CUDA_DIR}" ./buildall.sh --breakdowns

echo "[ STEP 4 ] Build HIP (with breakdowns)"
run_in_dir "${HIP_DIR}" ./buildall.sh --breakdowns

################################################################################
# 5) Re-run with breakdowns enabled
################################################################################
echo "[ STEP 5 ] Run CUDA (with breakdowns)"
run_in_dir "${CUDA_DIR}" env \
  RERUN_APPS=1 DO_BREAKDOWNS=1 LABEL=cuda OUT_DIR="${OUT_DIR}" \
  RUNS="${RUNS}" SLEEP_SECS="${SLEEP_SECS}" \
  GPU_ID="${NVIDIA_GPU_ID}" \
  ./runRodiniaWithIntervals.sh

echo "[ STEP 5 ] Run HIP (with breakdowns)"
run_in_dir "${HIP_DIR}" env \
  RERUN_APPS=1 DO_BREAKDOWNS=1 LABEL=hip OUT_DIR="${OUT_DIR}" \
  RUNS="${RUNS}" SLEEP_SECS="${SLEEP_SECS}" \
  AMD_GPU_ID="${AMD_GPU_ID}" \
  ./runRodiniaWithIntervals.sh

################################################################################
# 6) Plot breakdowns
################################################################################
echo "[ STEP 6 ] Plot breakdowns (reads ${CUDA_DIR}/${OUT_DIR} and ${HIP_DIR}/${OUT_DIR})"
cd "${ROOT_DIR}"
./run_plot_breakdowns.sh

echo "[ DONE ]"
