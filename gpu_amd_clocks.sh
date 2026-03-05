#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
gpu_amd_clocks.sh — helper for AMD/ROCm clock inspection + locking (rocm-smi/sysfs based)

Modes (pick exactly one):
  --get                Print recommended clocks:
                          - detect max supported memory clock (mclk) and gfx clock (sclk) when available
                          - step down gfx clock until droop/throttling is NOT observed
                        (does NOT permanently lock clocks; it may temporarily set clocks to probe)

  --set --mem <MHz> --gfx <MHz>
                        Lock clocks (best effort):
                          - sets mclk to <MHz> (or nearest supported state)
                          - sets sclk to <MHz> (or nearest supported state)

  --reset              Reset clocks to default/auto (best effort)

Common options:
  -g, --gpu <id>       GPU index as seen by rocm-smi (default: 0)
  --step <MHz>         Step-down decrement for gfx clock in --get (default: 25)
  --tries <N>          Max step-down attempts in --get (default: 60)
  --sleep <secs>       Sleep after setting candidate gfx in --get (default: 2)
  --droop <MHz>        Allowed droop vs target before considered throttling (default: 40)
  --samples <N>        Samples per candidate (default: 5)
  --sample-sleep <s>   Sleep between samples (default: 1)
  --list-gpus          Print GPU list (rocminfo Agent/Marketing Name + rocm-smi if present) and exit
  -h, --help           Show help

Examples:
  ./gpu_amd_clocks.sh --list-gpus
  ./gpu_amd_clocks.sh -g 0 --get
  ./gpu_amd_clocks.sh -g 0 --set --mem 1600 --gfx 1700
  ./gpu_amd_clocks.sh -g 0 --reset

Notes:
  - Requires sudo privileges for clock controls (rocm-smi / sysfs writes).
  - Not all ROCm stacks expose supported DPM states. If unavailable, --get will
    use current clocks as starting points and step down from there.
EOF
}

need_any_cmd() {
  local ok=1
  for c in "$@"; do
    if command -v "$c" >/dev/null 2>&1; then ok=0; fi
  done
  if [[ $ok -ne 0 ]]; then
    echo "[ ERROR ] Need one of: $*" >&2
    exit 1
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

list_gpus() {
  echo "=== rocminfo Agent/Marketing Name (recommended mapping) ==="
  if has_cmd rocminfo; then
    rocminfo 2>/dev/null | grep -E "Agent|Marketing Name" -n || true
  else
    echo "rocminfo not found."
  fi
  echo
  echo "=== rocm-smi (if available) ==="
  if has_cmd rocm-smi; then
    rocm-smi || true
  else
    echo "rocm-smi not found."
  fi
}

# Try to map "GPU index" -> sysfs card path:
# Prefer /sys/class/drm/card${N}/device when it's an amdgpu device.
sysfs_card_path_for_gpu() {
  local gpu="${1:?gpu}"
  local p="/sys/class/drm/card${gpu}/device"
  [[ -d "$p" ]] || return 1
  # quick sanity: amdgpu devices typically have pp_dpm_sclk
  [[ -f "${p}/pp_dpm_sclk" ]] || return 1
  echo "$p"
}

# Read supported DPM states from sysfs; format like:
# "0: 500Mhz *"
# Return max MHz found.
sysfs_max_mhz_from_dpm_file() {
  local file="${1:?file}"
  awk '
    {
      for (i=1; i<=NF; i++) {
        if ($i ~ /Mhz/ || $i ~ /MHz/) {
          gsub(/[^0-9]/,"",$i)
          v=$i+0
          if (v>max) max=v
        }
      }
    }
    END { if (max>0) print max; else exit 1 }' "$file"
}

# Read current sclk/mclk from sysfs (best effort).
sysfs_cur_mhz_from_freq_file() {
  local file="${1:?file}"
  # lines may have "*" on current state; prefer that
  local cur
  cur="$(grep -E '\*' "$file" 2>/dev/null | head -n1 | grep -Eo '[0-9]+' | head -n1 || true)"
  if [[ -z "$cur" ]]; then
    cur="$(head -n1 "$file" 2>/dev/null | grep -Eo '[0-9]+' | head -n1 || true)"
  fi
  [[ -n "$cur" ]] && echo "$cur" || return 1
}

# rocm-smi helpers (vary by version; keep loose parsing)
rocm_smi_cur_sclk() {
  local gpu="${1:?gpu}"
  rocm-smi -g "$gpu" --showclocks 2>/dev/null | grep -Ei 'sclk|gfx' | head -n1 | grep -Eo '[0-9]+' | tail -n1
}
rocm_smi_cur_mclk() {
  local gpu="${1:?gpu}"
  rocm-smi -g "$gpu" --showclocks 2>/dev/null | grep -Ei 'mclk|mem' | head -n1 | grep -Eo '[0-9]+' | tail -n1
}

# Best-effort get max supported clocks: prefer sysfs DPM tables; fallback to current.
get_max_supported_clocks() {
  local gpu="${1:?gpu}"
  if [[ -n "$(sysfs_card_path_for_gpu "$gpu" 2>/dev/null || true)" ]]; then
    local dev
    dev="$(sysfs_card_path_for_gpu "$gpu")"
    local max_m=0 max_s=0
    if [[ -f "${dev}/pp_dpm_mclk" ]]; then
      max_m="$(sysfs_max_mhz_from_dpm_file "${dev}/pp_dpm_mclk" || echo 0)"
    fi
    if [[ -f "${dev}/pp_dpm_sclk" ]]; then
      max_s="$(sysfs_max_mhz_from_dpm_file "${dev}/pp_dpm_sclk" || echo 0)"
    fi
    if [[ "$max_s" -gt 0 ]]; then
      echo "$max_m $max_s"
      return 0
    fi
  fi

  # fallback: rocm-smi current clocks
  if has_cmd rocm-smi; then
    local m s
    m="$(rocm_smi_cur_mclk "$gpu" || echo 0)"
    s="$(rocm_smi_cur_sclk "$gpu" || echo 0)"
    [[ "${s:-0}" -gt 0 ]] || return 1
    echo "${m:-0} ${s}"
    return 0
  fi

  return 1
}

get_cur_gfx_mhz() {
  local gpu="${1:?gpu}"
  # sysfs preferred
  if [[ -n "$(sysfs_card_path_for_gpu "$gpu" 2>/dev/null || true)" ]]; then
    local dev
    dev="$(sysfs_card_path_for_gpu "$gpu")"
    sysfs_cur_mhz_from_freq_file "${dev}/pp_dpm_sclk"
    return 0
  fi
  # fallback: rocm-smi
  if has_cmd rocm-smi; then
    rocm_smi_cur_sclk "$gpu"
    return 0
  fi
  return 1
}

# Set clocks: prefer rocm-smi, fallback to sysfs state selection (if possible).
set_clocks() {
  local gpu="${1:?gpu}"
  local mem="${2:?mem}"
  local gfx="${3:?gfx}"

  if has_cmd rocm-smi; then
    # Many rocm-smi versions accept --setsclk/--setmclk with either "level" or "MHz".
    # We try MHz first; if it fails, user can use a level manually later.
    sudo rocm-smi -g "$gpu" --setmclk "$mem" >/dev/null 2>&1 || true
    sudo rocm-smi -g "$gpu" --setsclk "$gfx" >/dev/null 2>&1 || true

    echo "[ DONE ] GPU${gpu}: requested mem=${mem} MHz, gfx=${gfx} MHz via rocm-smi (best-effort)"
    return 0
  fi

  # sysfs fallback (only if DPM tables exist)
  local dev
  dev="$(sysfs_card_path_for_gpu "$gpu")"
  # Try to set to closest state by picking highest state <= requested MHz
  if [[ -f "${dev}/pp_dpm_sclk" ]]; then
    local s_state
    s_state="$(awk -v tgt="$gfx" '
      {
        # state: "N: XXXMhz"
        if ($1 ~ /^[0-9]+:/) {
          n=$1; gsub(":","",n)
          mhz=$2; gsub(/[^0-9]/,"",mhz)
          if (mhz<=tgt && mhz>best_mhz) { best_mhz=mhz; best=n }
        }
      }
      END { if (best!="") print best; else print "" }' "${dev}/pp_dpm_sclk")"
    if [[ -n "$s_state" ]]; then
      echo "$s_state" | sudo tee "${dev}/pp_dpm_sclk" >/dev/null
    fi
  fi
  if [[ -f "${dev}/pp_dpm_mclk" ]]; then
    local m_state
    m_state="$(awk -v tgt="$mem" '
      {
        if ($1 ~ /^[0-9]+:/) {
          n=$1; gsub(":","",n)
          mhz=$2; gsub(/[^0-9]/,"",mhz)
          if (mhz<=tgt && mhz>best_mhz) { best_mhz=mhz; best=n }
        }
      }
      END { if (best!="") print best; else print "" }' "${dev}/pp_dpm_mclk")"
    if [[ -n "$m_state" ]]; then
      echo "$m_state" | sudo tee "${dev}/pp_dpm_mclk" >/dev/null
    fi
  fi

  echo "[ DONE ] GPU${gpu}: requested mem=${mem} MHz, gfx=${gfx} MHz via sysfs (state selection)"
}

reset_clocks() {
  local gpu="${1:?gpu}"

  if has_cmd rocm-smi; then
    sudo rocm-smi -g "$gpu" --resetclocks >/dev/null 2>&1 || true
    sudo rocm-smi -g "$gpu" --resetperf >/dev/null 2>&1 || true
    echo "[ DONE ] GPU${gpu}: reset via rocm-smi (best-effort)"
    return 0
  fi

  local dev
  dev="$(sysfs_card_path_for_gpu "$gpu")"
  # Reset methods vary. Try "0" or "auto" where supported.
  [[ -f "${dev}/pp_dpm_sclk" ]] && echo "0" | sudo tee "${dev}/pp_dpm_sclk" >/dev/null || true
  [[ -f "${dev}/pp_dpm_mclk" ]] && echo "0" | sudo tee "${dev}/pp_dpm_mclk" >/dev/null || true
  echo "[ DONE ] GPU${gpu}: reset via sysfs (best-effort)"
}

is_drooping() {
  local target="${1:?target}"
  local cur="${2:?cur}"
  local droop="${3:?droop}"
  (( cur < target - droop ))
}

find_recommended_clocks() {
  local gpu="${1:?gpu}"
  local step="${2:?step}"
  local tries="${3:?tries}"
  local sleep_s="${4:?sleep}"
  local droop="${5:?droop}"
  local samples="${6:?samples}"
  local sample_sleep="${7:?sample_sleep}"

  local mem_max gfx_max
  read -r mem_max gfx_max < <(get_max_supported_clocks "$gpu")
  echo "[ INFO ] GPU${gpu}: max seen: mem=${mem_max} MHz, gfx=${gfx_max} MHz"

  local gfx="$gfx_max"
  for ((t=0; t<tries; t++)); do
    echo "[ PROBE ] GPU${gpu}: trying gfx=${gfx} MHz"
    # try to apply candidate gfx (and keep mem at max if known)
    set_clocks "$gpu" "${mem_max:-0}" "$gfx"
    sleep "$sleep_s"

    local ok=1
    local vals=()
    for ((i=0; i<samples; i++)); do
      local cur
      cur="$(get_cur_gfx_mhz "$gpu" || echo "")"
      vals+=("${cur:-NA}")
      if [[ -z "$cur" ]]; then
        ok=0
      else
        if is_drooping "$gfx" "$cur" "$droop"; then
          ok=0
        fi
      fi
      sleep "$sample_sleep"
    done

    echo "[ PROBE ] samples: ${vals[*]}"
    if (( ok == 1 )); then
      echo "[ OK   ] recommended clocks: mem=${mem_max} MHz, gfx=${gfx} MHz"
      echo
      echo "MEM_MHZ=${mem_max}"
      echo "GFX_MHZ=${gfx}"
      return 0
    fi

    echo "[ PROBE ] droop detected -> step down (${step} MHz)"
    gfx=$((gfx - step))
    (( gfx > 0 )) || break
  done

  echo "[ ERROR ] could not find non-drooping gfx within ${tries} steps." >&2
  return 1
}

################################################################################
# Args
################################################################################
MODE=""
GPU_ID=0
STEP_MHZ=25
TRIES=60
SLEEP_S=2
DROOP_MHZ=40
SAMPLES=5
SAMPLE_SLEEP=1
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

    --droop) DROOP_MHZ="${2:?}"; shift 2 ;;
    --samples) SAMPLES="${2:?}"; shift 2 ;;
    --sample-sleep) SAMPLE_SLEEP="${2:?}"; shift 2 ;;

    *)
      echo "[ ERROR ] Unknown option: $1" >&2
      echo "Run with --help." >&2
      exit 2
      ;;
  esac
done

need_any_cmd rocminfo rocm-smi

if [[ "$DO_LIST" == "1" ]]; then
  list_gpus
  exit 0
fi

if [[ -z "$MODE" ]]; then
  echo "[ ERROR ] You must pass exactly one of: --get, --set, --reset" >&2
  echo "Run with --help." >&2
  exit 2
fi

case "$MODE" in
  get)
    find_recommended_clocks "$GPU_ID" "$STEP_MHZ" "$TRIES" "$SLEEP_S" "$DROOP_MHZ" "$SAMPLES" "$SAMPLE_SLEEP"
    ;;
  set)
    if [[ -z "${MEM_MHZ}" || -z "${GFX_MHZ}" ]]; then
      echo "[ ERROR ] --set requires --mem <MHz> and --gfx <MHz>" >&2
      exit 2
    fi
    set_clocks "$GPU_ID" "$MEM_MHZ" "$GFX_MHZ"
    ;;
  reset)
    reset_clocks "$GPU_ID"
    ;;
esac
