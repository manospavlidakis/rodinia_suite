#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
#time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 10 -np 10 &> ${path}/128_10_10_${file}
#time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 10 -np 100 &> ${path}/128_10_100_${file}
time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 10 -np 1000 &> ${path}/128_10_1000_${file}
time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 100 -np 10 &> ${path}/128_100_10_${file}
time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 100 -np 100 &> ${path}/128_100_100_${file}
time taskset -c 20-23 ./particlefilter_float -x 128 -y 128 -z 100 -np 1000 &> ${path}/128_100_1000_${file}

time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 10 -np 10 &> ${path}/256_10_10_${file}
time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 10 -np 100 &> ${path}/256_10_100_${file}
time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 10 -np 1000 &> ${path}/256_10_1000_${file}
time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 100 -np 10 &> ${path}/256_100_10_${file}
#time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 100 -np 100 &> ${path}/256_100_100_${file}
#time taskset -c 20-23 ./particlefilter_float -x 256 -y 256 -z 100 -np 1000 &> ${path}/256_100_1000_${file}
