#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"

PROFILING=${PROFILING:-0}

if [[ "$PROFILING" == "1" ]]; then
    report="nsys_${benchmark_name}"
    rm -f ${report}.sqlite 2> /dev/null
    nsys profile --trace=cuda --cuda-event-trace=false --sample=none --output ${report} ./${benchmark_name} file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt

    nsys stats --format csv --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum ${report}.nsys-rep > ${benchmark_name}_nsight.csv
else
    for ((iter=1; iter<=30; iter++))
    do
        ./${benchmark_name} file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt &>6000_${iter}_${file}
    done

    ../find_avg_per_app.py ${benchmark_name}
fi
