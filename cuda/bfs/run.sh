#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph1k.txt &> ${path}/1k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph2k.txt &> ${path}/2k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph4k.txt &> ${path}/4k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph8k.txt &> ${path}/8k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph16k.txt &> ${path}/16k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph32k.txt &> ${path}/32k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph64k.txt &> ${path}/64k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph128k.txt &> ${path}/128k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph256k.txt &> ${path}/256k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph512k.txt &> ${path}/512k_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph1M.txt &> ${path}/1M_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph2M.txt &> ${path}/2M_${file}
time taskset -c 4-7 ./bfs /mnt/vol0/data/bfs/inputGen/graph4M.txt &> ${path}/4M_${file}
