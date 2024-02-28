#!/bin/bash
path=$1
file=$2
selection=$3
if [ ${selection} -eq 1 ]
then
  echo ${selectio}
  time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 10 -np 10 &> ${path}/128_10_10_${file}
elif [ ${selection} -eq 2 ]
then
  echo ${selectio}
  time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 10 -np 100 &> ${path}/128_10_100_${file}
elif [ ${selection} -eq 3 ]
then
  echo ${selectio}
  time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 10 -np 1000 &> ${path}/128_10_1000_${file}
elif [ ${selection} -eq 4 ]
then
  echo ${selection} "Runs only for 3 iter"
  #time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 100 -np 10 &> ${path}/128_100_10_${file}
elif [ ${selection} -eq 5 ]
then
  echo ${selection} "Runs only for 6 iter"
  #time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 100 -np 100 &> ${path}/128_100_100_${file}
elif [ ${selection} -eq 6 ]
then
  time taskset -c 4-7 ./particlefilter_float -x 128 -y 128 -z 100 -np 1000 &> ${path}/128_100_1000_${file}
elif [ ${selection} -eq 7 ]
then
  time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 10 -np 10 &> ${path}/256_10_10_${file}
elif [ ${selection} -eq 8 ]
then
  time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 10 -np 100 &> ${path}/256_10_100_${file}
elif [ ${selection} -eq 9 ]
then
  time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 10 -np 1000 &> ${path}/256_10_1000_${file}
elif [ ${selection} -eq 10 ]
then
  time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 100 -np 10 &> ${path}/256_100_10_${file}
elif [ ${selection} -eq 11 ]
then
  echo "-z 100 -np 100 does not work for many iterations"
  #time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 100 -np 100 &> ${path}/256_100_100_${file}
elif [ ${selection} -eq 12 ]
then
  echo "-z 100 -np 1000 does not work for many iterations"
  #time taskset -c 4-7 ./particlefilter_float -x 256 -y 256 -z 100 -np 1000 &> ${path}/256_100_1000_${file}
else
  echo "not valid selection!"
fi
sleep 2
