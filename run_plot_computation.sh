#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   ./run_plot_computation.sh
#   OUT=comp.png TITLE="Rodinia computation" ./run_plot_computation.sh
#   LOGY=1 ./run_plot_computation.sh
#   LOGY=1 LOGY_MIN=1e-3 OUT=comp_log.png ./run_plot_computation.sh

OUT="${OUT:-computation_bars.png}"
TITLE="${TITLE:-}"

LOGY="${LOGY:-0}"
LOGY_MIN="${LOGY_MIN:-}"

cmd=(python3 plot_computation_bars.py -o "$OUT")
if [[ -n "$TITLE" ]]; then cmd+=(--title "$TITLE"); fi

if [[ "$LOGY" == "1" || "$LOGY" == "true" || "$LOGY" == "TRUE" ]]; then
  cmd+=(--logy)
  if [[ -n "$LOGY_MIN" ]]; then
    cmd+=(--logy-min "$LOGY_MIN")
  fi
fi

echo "[ RUN ] ${cmd[*]}"
"${cmd[@]}"
echo "[ DONE ] Wrote ${OUT}"
