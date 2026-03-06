#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"

PROFILING=${PROFILING:-0}

if [[ "$PROFILING" == "1" ]]; then
    report="nsys_${benchmark_name}"

    nsys profile --trace=cuda --cuda-event-trace=false --sample=none --output ${report} ./${benchmark_name} list2048k.txt -r 5 -lat 30 -lng 90 -q

    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv ${report}.nsys-rep > ${benchmark_name}_nsight.csv
else
    for ((iter=1; iter<=30; iter++))
    do
        ./${benchmark_name} list2048k.txt -r 5 -lat 30 -lng 90 -q &> 2048_${iter}_${file}
    done

    ../find_avg_per_app.py ${benchmark_name}
fi
