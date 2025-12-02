#!/usr/bin/env bash
usage() {
  cat <<'EOF'
Rodinia HIP build script

Usage:
  buildall_hip.sh [options]
  buildall_hip.sh <HIP_DIR> <GFX_ARCH>   (legacy)

Options:
  --hip <path>         Path to HIP/ROCm root (e.g., /opt/rocm)
  --gfx <arch>         GPU arch (e.g., gfx90a, gfx1030, gfx1100). If omitted, auto-detect.
  --print-config       Print resolved configuration and exit
  -h, --help           Show this help

Environment variables (alternative to args):
  HIP_DIR              ROCm root
  GFX_ARCH             gfx arch

Examples:
  ./buildall_hip.sh
  ./buildall_hip.sh --hip /opt/rocm --gfx gfx1100
  ./buildall_hip.sh /opt/rocm gfx1030
EOF
}

# Count total Makefiles
TOTAL=$(find -name 'Makefile' | wc -l | tr -d ' ')
COUNT=0

print_progress() {
  local progress=$1
  local total=$2
  local percent=$((progress * 100 / total))
  local bar_size=40
  local filled=$((percent * bar_size / 100))
  local empty=$((bar_size - filled))

  printf "\r["
  printf "%0.s#" $(seq 1 $filled 2>/dev/null)
  printf "%0.s-" $(seq 1 $empty 2>/dev/null)
  printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

detect_hip_dir() {
  if command -v hipcc >/dev/null 2>&1; then
    readlink -f "$(command -v hipcc)" | sed -E 's#/bin/hipcc$##'
    return 0
  fi
  [[ -d /opt/rocm ]] && { echo /opt/rocm; return 0; }
  for d in /opt/rocm-* /usr/local/rocm-*; do
    [[ -x "$d/bin/hipcc" ]] && { echo "$d"; return 0; }
  done
  return 1
}

detect_gfx_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1
    return 0
  fi
  if command -v hipcc >/dev/null 2>&1; then
    hipcc --amdgpu-targets 2>/dev/null | tr ',' '\n' | grep -oE 'gfx[0-9a-f]+' | head -n 1
    return 0
  fi
  return 1
}

# Defaults from env (if set)
HIP_DIR="${HIP_DIR:-}"
GFX_ARCH="${GFX_ARCH:-}"
PRINT_CONFIG="false"

# Parse args (supports both options and legacy positional)
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --hip) HIP_DIR="${2:-}"; shift 2 ;;
    --gfx) GFX_ARCH="${2:-}"; shift 2 ;;
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

# Legacy positional: <HIP_DIR> <GFX_ARCH>
if [[ ${#positional[@]} -ge 1 && -z "${HIP_DIR}" ]]; then HIP_DIR="${positional[0]}"; fi
if [[ ${#positional[@]} -ge 2 && -z "${GFX_ARCH}" ]]; then GFX_ARCH="${positional[1]}"; fi

# Resolve HIP/ROCm dir
if [[ -z "${HIP_DIR}" ]]; then
  HIP_DIR="$(detect_hip_dir)" || { echo "ERROR: ROCm/HIP not found. Use --hip or set HIP_DIR." >&2; exit 1; }
fi

# Resolve GFX arch
if [[ -z "${GFX_ARCH}" ]]; then
  GFX_ARCH="$(detect_gfx_arch)" || {
    echo "ERROR: Could not auto-detect GFX arch. Use --gfx (e.g., gfx1100) or set GFX_ARCH." >&2
    exit 1
  }
fi

# Resolve HIP lib dir (optional; may be unused by Makefiles)
if [[ -d "${HIP_DIR}/lib" ]]; then
  HIP_LIB_DIR="${HIP_DIR}/lib"
elif [[ -d "${HIP_DIR}/lib64" ]]; then
  HIP_LIB_DIR="${HIP_DIR}/lib64"
elif [[ -d "${HIP_DIR}/lib/x86_64-linux-gnu" ]]; then
  HIP_LIB_DIR="${HIP_DIR}/lib/x86_64-linux-gnu"
else
  HIP_LIB_DIR=""
  echo "WARN: HIP lib dir not found under ${HIP_DIR} (continuing)" >&2
fi

if [[ "${PRINT_CONFIG}" == "true" ]]; then
  echo "HIP_DIR=${HIP_DIR}"
  echo "HIP_LIB_DIR=${HIP_LIB_DIR}"
  echo "GFX_ARCH=${GFX_ARCH}"
  exit 0
fi

echo "ROCm/HIP dir: ${HIP_DIR}"
echo "GFX arch: ${GFX_ARCH}"

AMDGPU_TARGETS="${GFX_ARCH}"
CXXFLAGS=" -m64 -O3"
CXXFLAGS+=" -DOUTPUT"
CXXFLAGS+=" -DDISABLE_HIP_CHECK"
CXXFLAGS+=" -DBREAKDOWNS"

for mf in $(find -name 'Makefile'); do
  COUNT=$((COUNT + 1))
  dir=$(dirname "$mf")
  echo "$dir"
  cd "$dir" || exit

  dir_name=$(basename "$dir")
  if [[ "$dir_name" == "kmeans" || "$dir_name" == "hybridsort" ]]; then
    echo "Kmeans and hybridsort are not supported by SCALE due to texture issues."
  else
    make clean >/dev/null 2>&1
    make -s -j \
      HIP_DIR="$HIP_DIR" ROCM_PATH="$HIP_DIR" \
      AMDGPU_TARGETS="$AMDGPU_TARGETS" HCC_AMDGPU_TARGET="$AMDGPU_TARGETS" \
      HIP_LIB_DIR="$HIP_LIB_DIR" \
      CXXFLAGS="$CXXFLAGS"
  fi

  cd - >/dev/null
  print_progress "$COUNT" "$TOTAL"
done

echo -e "\nCompilation complete."
