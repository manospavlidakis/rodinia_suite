#!/bin/bash
path=$1
file=$2
selection=$3
#echo "filename: "${file}
#echo "path: "${path}
#echo "1k"
if [ ${selection} -eq 1 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph1k.txt &> ${path}/1k_${file}
elif [ ${selection} -eq 2 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph2k.txt &> ${path}/2k_${file}
elif [ ${selection} -eq 3 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph4k.txt &> ${path}/4k_${file}
elif [ ${selection} -eq 4 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph8k.txt &> ${path}/8k_${file}
elif [ ${selection} -eq 5 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph16k.txt &> ${path}/16k_${file}
elif [ ${selection} -eq 6 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph32k.txt &> ${path}/32k_${file}
elif [ ${selection} -eq 7 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph64k.txt &> ${path}/64k_${file}
elif [ ${selection} -eq 8 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph128k.txt &> ${path}/128k_${file}
elif [ ${selection} -eq 9 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph256k.txt &> ${path}/256k_${file}
elif [ ${selection} -eq 10 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph512k.txt &> ${path}/512k_${file}
elif [ ${selection} -eq 11 ]
then
  time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph1M.txt &> ${path}/1M_${file}
elif [ ${selection} -eq 12 ]
then
  echo "It may chrash!!! 2M nodes"
  #time taskset -c 4-7 ./bfs /spare/data/bfs/inputGen/graph2M.txt > ${path}/2M_${file}
else
  echo "Not valid choice!"
fi
  sleep 5
