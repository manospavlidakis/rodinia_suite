#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
for ((iter=1; iter<=5;iter++))
do
./${benchmark_name} -x 128 -y 128 -z 100 -np 1000 &> 128_100_1000_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
# ./particlefilter_float -x 256 -y 256 -z 10 -np 10 &> 256_10_10_${file}
# ./particlefilter_float -x 256 -y 256 -z 10 -np 100 &> 256_10_100_${file}
# ./particlefilter_float -x 256 -y 256 -z 10 -np 1000 &> 256_10_1000_${file}
# ./particlefilter_float -x 256 -y 256 -z 100 -np 10 &> 256_100_10_${file}
# ./particlefilter_float -x 256 -y 256 -z 100 -np 100 &> 256_100_100_${file}
# ./particlefilter_float -x 256 -y 256 -z 100 -np 1000 &> 256_100_1000_${file}
