#!/bin/bash
benchmarks=("backprop" "bfs" "b+tree" "cfd" "dwt2d" "gaussian" "heartwall" "hotspot" "hotspot3D" "huffman" "lavaMD" "nn" "nw" "pathfinder")
for b in "${benchmarks[@]}"; do
  echo "[ RUN               ] $b"
  cd "$b" || continue
  ./../find_avg_per_app_break.py "$b" || true
  cd - &>/dev/null
done
