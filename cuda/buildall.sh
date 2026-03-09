#!/usr/bin/env bash
usage() {
  cat <<'EOF'
Rodinia CUDA build script

Usage:
  buildall.sh [options]
  buildall.sh <CUDA_DIR> <SM_VERSION> <SPECTRAL_FLAG>   (legacy)

Options:
  --cuda <path>        Path to CUDA toolkit root (e.g., /usr/local/cuda, /opt/cuda-12.4)
  --sm <NN>            SM version (e.g., 86, 89, 90). Default: 86
  --breakdowns         Enable BREAKDOWNS (-DBREAKDOWNS)
  --no-breakdowns      Disable BREAKDOWNS (default)
  --spectral           Enable "spectral" mode (equivalent to SPECTRAL=true)
  --no-spectral        Disable spectral mode (default)
  --print-config       Print resolved configuration and exit
  -h, --help           Show this help

Environment variables (alternative to args):
  CUDA_DIR             CUDA toolkit root
  SM_VERSION           SM version
  SPECTRAL             true/false
  BREAKDOWNS           true/false

Examples:
  # autodetect CUDA, default SM=86, no breakdowns
  ./buildall.sh

  # explicit CUDA + SM
  ./buildall.sh --cuda /usr/local/cuda-13.0 --sm 89

  # enable breakdowns
  ./buildall.sh --breakdowns

  # spectral + breakdowns
  ./buildall.sh --spectral --breakdowns --cuda /usr/local/cuda --sm 86

  # legacy positional
  ./buildall.sh /usr/local/cuda-12.9 86 true
EOF
}

# Count total Makefiles only in cuda path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Count total Makefiles only under cuda/
TOTAL=$(find "${SCRIPT_DIR}" -name 'Makefile' | wc -l)
COUNT=0

# Function to print progress bar
print_progress() {
    local progress=$1
    local total=$2
    local percent=$((progress * 100 / total))
    local bar_size=40
    local filled=$((percent * bar_size / 100))
    local empty=$((bar_size - filled))

    printf "\r["
    printf "%0.s#" $(seq 1 $filled)
    printf "%0.s-" $(seq 1 $empty)
    printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

detect_cuda_dir() {
  if command -v nvcc >/dev/null 2>&1; then
    readlink -f "$(command -v nvcc)" | sed -E 's#/bin/nvcc$##'
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then echo /usr/local/cuda; return 0; fi
  for d in /opt/cuda-* /usr/local/cuda-*; do
    [[ -x "$d/bin/nvcc" ]] && { echo "$d"; return 0; }
  done
  return 1
}

# For native CUDA: export PATH=/usr/local/cuda/bin:$PATH
# Defaults from env (if set)
CUDA_DIR="${CUDA_DIR:-}"
SM_VERSION="${SM_VERSION:-89}"
SPECTRAL="${SPECTRAL:-false}"
BREAKDOWNS="${BREAKDOWNS:-false}"
PRINT_CONFIG="false"

# Parse args (supports both options and legacy positional)
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --cuda) CUDA_DIR="${2:-}"; shift 2 ;;
    --sm) SM_VERSION="${2:-}"; shift 2 ;;
    --spectral) SPECTRAL="true"; shift ;;
    --no-spectral) SPECTRAL="false"; shift ;;
    --breakdowns) BREAKDOWNS="true"; shift ;;
    --no-breakdowns) BREAKDOWNS="false"; shift ;;
    --print-config) PRINT_CONFIG="true"; shift ;;
    --) shift; positional+=("$@"); break ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Run with --help to see options." >&2
      exit 2
      ;;
    *) positional+=("$1"); shift ;;
  esac
done

# Legacy positional: <CUDA_DIR> <SM_VERSION> <SPECTRAL_FLAG>
if [[ ${#positional[@]} -ge 1 && -z "${CUDA_DIR}" ]]; then CUDA_DIR="${positional[0]}"; fi
if [[ ${#positional[@]} -ge 2 && "${SM_VERSION}" == "86" ]]; then SM_VERSION="${positional[1]}"; fi
if [[ ${#positional[@]} -ge 3 ]]; then SPECTRAL="${positional[2]}"; fi

# Resolve CUDA dir
if [[ -z "${CUDA_DIR}" ]]; then
  CUDA_DIR="$(detect_cuda_dir)" || { echo "ERROR: CUDA not found. Use --cuda or set CUDA_DIR." >&2; exit 1; }
fi

# Resolve CUDA lib dir robustly
if [[ -d "${CUDA_DIR}/lib64" ]]; then
  CUDA_LIB_DIR="${CUDA_DIR}/lib64"
elif [[ -d "${CUDA_DIR}/targets/x86_64-linux/lib" ]]; then
  CUDA_LIB_DIR="${CUDA_DIR}/targets/x86_64-linux/lib"
else
  echo "ERROR: CUDA lib dir not found under ${CUDA_DIR}" >&2
  exit 1
fi
echo "========== DEBUG COMPILER =========="
echo "PATH=$PATH"
echo "NVCC=$(which nvcc)"
echo "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-<unset>}"
echo "CUDA_DIR=$CUDA_DIR"
echo "CUDA_LIB_DIR=$CUDA_LIB_DIR"
nvcc --version
echo "===================================="
if [[ "${PRINT_CONFIG}" == "true" ]]; then
  echo "CUDA_DIR=${CUDA_DIR}"
  echo "CUDA_LIB_DIR=${CUDA_LIB_DIR}"
  echo "SM_VERSION=${SM_VERSION}"
  echo "SPECTRAL=${SPECTRAL}"
  echo "BREAKDOWNS=${BREAKDOWNS}"
  exit 0
fi

if [[ "${SPECTRAL}" == "true" ]]; then
  echo "Run with Spectral"
else
  echo "Run without Spectral"
fi

if [[ "${BREAKDOWNS}" == "true" ]]; then
  echo "Breakdowns: ENABLED"
else
  echo "Breakdowns: DISABLED"
fi

echo "CUDA dir: ${CUDA_DIR}"
echo "SM version: ${SM_VERSION}"

GENCODE_FLAGS="-gencode arch=compute_${SM_VERSION},code=sm_${SM_VERSION} -gencode arch=compute_${SM_VERSION},code=compute_${SM_VERSION}"
CXXFLAGS=" -m64 -O3"

CXXFLAGS+=" -DOUTPUT"

# Enable/disable breakdowns via flag
if [[ "${BREAKDOWNS}" == "true" ]]; then
  CXXFLAGS+=" -DBREAKDOWNS"
fi

for mf in $(find "${SCRIPT_DIR}" -name 'Makefile'); do
    COUNT=$((COUNT + 1))
    dir=$(dirname "$mf")
    echo "$dir"
    cd "$dir" || exit

    dir_name=$(basename "$dir")

    if [[ "$dir_name" == "kmeans" || "$dir_name" == "hybridsort" ]]; then
        echo "Kmeans and hybridsort are not supported by SCALE due to texture issues."
    else
        make clean >/dev/null 2>&1
        make -s -j CUDA_DIR="$CUDA_DIR" GENCODE_FLAGS="$GENCODE_FLAGS" CXXFLAGS="$CXXFLAGS" CUDA_LIB_DIR="$CUDA_LIB_DIR"
    fi

    cd - > /dev/null
    print_progress "$COUNT" "$TOTAL"
done

echo -e "\nCompilation complete."
