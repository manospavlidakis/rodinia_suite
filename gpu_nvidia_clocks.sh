#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
gpu_nvidia_clocks.sh — simple helper for NVIDIA persistence + clock locking

Modes (pick exactly one):
  --get                Print recommended clocks:
                          - detect max supported memory clock
                          - take max graphics clock at that memory clock
                          - step down graphics clock until throttling is NOT active
                        (does NOT lock clocks)

  --set --mem <MHz> --gfx <MHz>
                        Enable persistence mode and lock clocks:
                          sudo nvidia-smi -pm 1
                          sudo nvidia-smi -i <gpu> -lmc <mem>,<mem>
                          sudo nvidia-smi -i <gpu> -lgc <gfx>,<gfx>

  --reset              Reset clocks and disable persistence mode:
                          sudo nvidia-smi -i <gpu> -rgc
                          sudo nvidia-smi -i <gpu> -rmc
                          sudo nvidia-smi -pm 0

Common options:
  -g, --gpu <id>       GPU index as in `nvidia-smi -L` (default: 0)
  --step <MHz>         Step-down decrement for graphics clock in --get (default: 15)
  --tries <N>          Max step-down attempts in --get (default: 40)
  --sleep <secs>       Sleep after setting candidate gfx in --get (default: 2)
  --list-gpus          Print `nvidia-smi -L` and exit
  -h, --help           Show help

Examples:
  ./gpu_nvidia_clocks.sh --list-gpus
  ./gpu_nvidia_clocks.sh -g 0 --get
  ./gpu_nvidia_clocks.sh -g 0 --set --mem 10501 --gfx 3090
  ./gpu_nvidia_clocks.sh -g 0 --reset

Notes:
  - Requires sudo privileges for clock controls.
  - "Not throttling" is detected by checking for any "Active: Yes" under
    "Clocks Throttle Reasons" in `nvidia-smi -q`.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ ERROR ] Missing: $1" >&2; exit 1; }
}

list_gpus() { nvidia-smi -L || true; }

# Extract: "<max_mem> <max_gfx_at_max_mem>"
get_max_supported_clocks() {
  local gpu="${1:?gpu}"
  nvidia-smi -i "$gpu" -q -d SUPPORTED_CLOCKS 2>/dev/null | \
  awk '
    /Memory[[:space:]]*:[[:space:]]*[0-9]+[[:space:]]*MHz/ {
      mem=$3+0
      if (mem > max_mem) { max_mem = mem; max_gfx_at_max_mem = 0 }
      current_mem = mem
      next
    }
    /Graphics[[:space:]]*:[[:space:]]*[0-9]+[[:space:]]*MHz/ {
      gfx=$3+0
      if (current_mem == max_mem && gfx > max_gfx_at_max_mem) max_gfx_at_max_mem = gfx
      next
    }
    END {
      if (max_mem > 0 && max_gfx_at_max_mem > 0) {
        printf "%d %d\n", max_mem, max_gfx_at_max_mem
      } else {
        exit 1
      }
    }'
}

# Returns 0 if throttling is ACTIVE (any reason), 1 if not (or if section missing)
is_throttling_active() {
  local gpu="${1:?gpu}"
  local out
  out="$(nvidia-smi -i "$gpu" -q -d CLOCK,POWER,TEMPERATURE 2>/dev/null || true)"

  # If the section isn't present, assume "not throttling" (return 1).
  echo "$out" | grep -q "Clocks Throttle Reasons" || return 1

  # If ANY "Active: Yes" exists, throttling is active.
  echo "$out" | grep -Eq "Active[[:space:]]*:[[:space:]]*Yes"
}

# --get: step down gfx until not throttling, but DO NOT lock final clocks permanently
# We still need to apply temporary -lgc to test candidates.
find_recommended_clocks() {
  local gpu="${1:?gpu}"
  local step="${2:?step}"
  local tries="${3:?tries}"
  local sleep_s="${4:?sleep}"

  read -r mem_max gfx_max < <(get_max_supported_clocks "$gpu")
  echo "[ INFO ] GPU${gpu}: max supported: mem=${mem_max} MHz, gfx=${gfx_max} MHz"

  # Ensure persistence for consistent behavior during probing.
  sudo nvidia-smi -pm 1 >/dev/null

  # Lock memory during probing so gfx pairing is meaningful.
  sudo nvidia-smi -i "$gpu" -lmc "${mem_max},${mem_max}" >/dev/null

  local gfx="$gfx_max"
  for ((t=0; t<tries; t++)); do
    echo "[ PROBE ] GPU${gpu}: trying gfx=${gfx} MHz @ mem=${mem_max} MHz"
    sudo nvidia-smi -i "$gpu" -lgc "${gfx},${gfx}" >/dev/null
    sleep "$sleep_s"

    if is_throttling_active "$gpu"; then
      echo "[ PROBE ] throttling detected -> step down (${step} MHz)"
      gfx=$((gfx - step))
      continue
    fi

    echo "[ OK   ] recommended clocks: mem=${mem_max} MHz, gfx=${gfx} MHz"
    echo
    echo "MEM_MHZ=${mem_max}"
    echo "GFX_MHZ=${gfx}"
    return 0
  done

  echo "[ ERROR ] could not find non-throttling gfx within ${tries} steps." >&2
  return 1
}

do_set() {
  local gpu="${1:?gpu}"
  local mem="${2:?mem}"
  local gfx="${3:?gfx}"

  sudo nvidia-smi -pm 1 >/dev/null
  sudo nvidia-smi -i "$gpu" -lmc "${mem},${mem}"
  sudo nvidia-smi -i "$gpu" -lgc "${gfx},${gfx}"
  echo "[ DONE ] GPU${gpu}: locked mem=${mem} MHz, gfx=${gfx} MHz (persistence ON)"
}

do_reset() {
  local gpu="${1:?gpu}"
  sudo nvidia-smi -i "$gpu" -rgc || true
  sudo nvidia-smi -i "$gpu" -rmc || true
  sudo nvidia-smi -pm 0 || true
  echo "[ DONE ] GPU${gpu}: reset clocks (persistence OFF)"
}

################################################################################
# Args
################################################################################
MODE=""
GPU_ID=0
STEP_MHZ=15
TRIES=40
SLEEP_S=2
MEM_MHZ=""
GFX_MHZ=""
DO_LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list-gpus) DO_LIST=1; shift ;;
    -g|--gpu) GPU_ID="${2:?}"; shift 2 ;;

    --get) MODE="get"; shift ;;
    --set) MODE="set"; shift ;;
    --reset) MODE="reset"; shift ;;

    --mem) MEM_MHZ="${2:?}"; shift 2 ;;
    --gfx) GFX_MHZ="${2:?}"; shift 2 ;;

    --step) STEP_MHZ="${2:?}"; shift 2 ;;
    --tries) TRIES="${2:?}"; shift 2 ;;
    --sleep) SLEEP_S="${2:?}"; shift 2 ;;

    *)
      echo "[ ERROR ] Unknown option: $1" >&2
      echo "Run with --help." >&2
      exit 2
      ;;
  esac
done

need_cmd nvidia-smi

if [[ "$DO_LIST" == "1" ]]; then
  list_gpus
  exit 0
fi

if [[ -z "$MODE" ]]; then
  echo "[ ERROR ] You must pass exactly one of: --get, --set, --reset" >&2
  echo "Run with --help." >&2
  exit 2
fi

################################################################################
# Dispatch
################################################################################
case "$MODE" in
  get)
    find_recommended_clocks "$GPU_ID" "$STEP_MHZ" "$TRIES" "$SLEEP_S"
    ;;
  set)
    if [[ -z "${MEM_MHZ}" || -z "${GFX_MHZ}" ]]; then
      echo "[ ERROR ] --set requires --mem <MHz> and --gfx <MHz>" >&2
      exit 2
    fi
    do_set "$GPU_ID" "$MEM_MHZ" "$GFX_MHZ"
    ;;
  reset)
    do_reset "$GPU_ID"
    ;;
  *)
    echo "[ ERROR ] internal: unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
