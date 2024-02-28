#!/bin/bash
path=$1
file=$2
selection=$3

#echo "filename: "${file}
#echo "path: "${path}
if [ ${selection} -eq 1 ]    
then
  time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix256.txt &> ${path}/256_${file}
elif [ ${selection} -eq 2 ]    
then
  time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix512.txt &> ${path}/512_${file}
elif [ ${selection} -eq 3 ]    
then
  time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix1024.txt &> ${path}/1024_${file}
elif [ ${selection} -eq 4 ]    
then
  time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix2048.txt &> ${path}/2048_${file}
elif [ ${selection} -eq 5 ]    
then
  echo "4096 matric does not work"
  #time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix4096.txt &> ${path}/4096_${file}
elif [ ${selection} -eq 6 ]    
then
  echo "8192 matrix does not work"
  #time taskset -c 4-7 ./gaussian -f /spare/data/gaussian/matrix8192.txt &> ${path}/8192_${file}
else 
  echo "Not valid choice!"
fi
