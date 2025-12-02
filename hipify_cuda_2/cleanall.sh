#!/usr/bin/env bash
set -euo pipefail

# Clean outputs in every directory that has a Makefile:
# 1) make clean
# 2) remove *.csv (and a few common output files)
# 3) remove leftover executables (files with +x) in that directory, excluding scripts

EXTRA_GLOBS=(
  "*.csv"
  "result.txt"
  "*.out"
  "*.log"
)

# Safety: don't delete these even if executable
EXE_KEEP_REGEX='(\.sh$|\.bash$|\.py$|\.pl$|\.rb$|\.jar$|\.so$|\.a$|\.o$|\.cu$|\.cpp$|\.cc$|\.c$|\.h$|\.hpp$|\.cuh$)'

for mf in $(find -name 'Makefile'); do
  dir="$(dirname "$mf")"
  echo "==> Cleaning $dir"
  pushd "$dir" >/dev/null

  # 1) Makefile clean (best-effort)
  make clean >/dev/null 2>&1 || true

  # 2) Remove csv and other common outputs
  for g in "${EXTRA_GLOBS[@]}"; do
    rm -f $g 2>/dev/null || true
  done

  popd >/dev/null
done

echo "Deep clean complete."
