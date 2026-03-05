#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
gpu_virtualization.sh — detect system/GPU virtualization and write a report

Usage:
  ./gpu_virtualization.sh [--out <report_file>] [--help]

Outputs:
  - Report file (default: gpu_virtualization_report.txt)
  - PCI GPU device list: <report_dir>/pci_gpu_devices.txt

Examples:
  ./gpu_virtualization.sh
  ./gpu_virtualization.sh --out logs/virt_report.txt
EOF
}

OUT_FILE="gpu_virtualization_report.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --out) OUT_FILE="${2:?}"; shift 2 ;;
    *)
      echo "[ ERROR ] Unknown option: $1" >&2
      echo "Run with --help." >&2
      exit 2
      ;;
  esac
done

# Prepare output paths
OUT_DIR="$(dirname "$OUT_FILE")"
mkdir -p "$OUT_DIR"
PCI_GPU_FILE="${OUT_DIR}/pci_gpu_devices.txt"

# Mirror all output (stdout+stderr) to the report file
exec > >(tee "$OUT_FILE") 2>&1

echo "====================================="
echo " GPU Virtualization Detection Script "
echo "====================================="
echo
echo "[ REPORT FILE        ] $OUT_FILE"
echo "[ PCI GPU DEVICES    ] $PCI_GPU_FILE"
echo

############################################
# System virtualization
############################################
echo "=== System virtualization ==="

virt="unknown"

if command -v systemd-detect-virt >/dev/null 2>&1; then
    virt="$(systemd-detect-virt 2>/dev/null || echo "unknown")"
fi

if [[ "$virt" == "none" ]]; then
    echo "System virtualization: NONE (bare metal)"
elif [[ "$virt" == "unknown" || -z "$virt" ]]; then
    echo "System virtualization: unable to determine"
else
    echo "System virtualization: $virt"
fi

echo

############################################
# NVIDIA GPUs
############################################
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== NVIDIA GPUs ==="
    nvidia-smi -L 2>/dev/null || echo "Unable to query NVIDIA GPUs"
    echo

    echo "--- MIG mode ---"
    nvidia-smi --query-gpu=index,name,mig.mode.current \
        --format=csv,noheader 2>/dev/null || \
        echo "MIG not supported or disabled"

    echo

    echo "--- vGPU check ---"
    nvidia-smi -q 2>/dev/null | grep -i "Virtualization Mode" || \
        echo "No NVIDIA vGPU detected"

    echo
fi

############################################
# AMD GPUs
############################################
echo "=== AMD GPUs ==="

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E "Agent|Marketing Name" || \
        echo "Unable to query AMD GPUs"
else
    echo "rocminfo not available"
fi

echo

############################################
# AMD SR-IOV
############################################
echo "--- AMD SR-IOV detection ---"

sriov_found=0

for d in /sys/bus/pci/devices/*; do
    if [[ -f "$d/sriov_numvfs" ]]; then
        vfs="$(cat "$d/sriov_numvfs" 2>/dev/null || echo 0)"
        if [[ "$vfs" != "0" ]]; then
            echo "SR-IOV enabled on $d ($vfs virtual functions)"
            sriov_found=1
        fi
    fi
done

if [[ "$sriov_found" == "0" ]]; then
    echo "No AMD SR-IOV virtualization detected"
fi

echo

############################################
# PCI GPU devices
############################################
echo "=== PCI GPU devices ==="

# Prefer only GPU-ish classes (VGA/3D/Display). Fall back to vendor grep if needed.
gpu_devices="$(lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' || true)"
if [[ -z "$gpu_devices" ]]; then
  gpu_devices="$(lspci -nn 2>/dev/null | grep -Ei 'nvidia|amd|radeon' || true)"
fi

if [[ -n "$gpu_devices" ]]; then
    echo "$gpu_devices"
    printf "%s\n" "$gpu_devices" > "$PCI_GPU_FILE"
else
    echo "No GPU PCI devices found"
    : > "$PCI_GPU_FILE"
fi

echo

############################################
# Virtual function detection
############################################
echo "--- PCI Virtual Function check ---"

if lspci -vv 2>/dev/null | grep -i "Virtual Function" >/dev/null; then
    echo "PCI Virtual Functions detected"
else
    echo "No PCI Virtual Functions detected"
fi

echo
echo "====================================="
echo " Final Result"
echo "====================================="

if [[ "$virt" == "none" && "$sriov_found" == "0" ]]; then
    echo "No system or GPU virtualization detected."
    echo "Environment appears to be bare metal with direct GPU access."
else
    echo "Virtualization features detected above."
fi
