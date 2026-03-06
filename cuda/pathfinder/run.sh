#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"

PROFILING=${PROFILING:-0}

if [[ "$PROFILING" == "1" ]]; then
    report="nsys_${benchmark_name}"

    nsys profile --trace=cuda --cuda-event-trace=false --sample=none --output ${report} ./${benchmark_name} 4096 4096 2

    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv ${report}.nsys-rep > ${benchmark_name}_nsight.csv
else
    for ((iter=1; iter<=30; iter++))
    do
        ./${benchmark_name} 4096 4096 2 &> 4096_${iter}_${file}
    done

    ../find_avg_per_app.py ${benchmark_name}
fi
