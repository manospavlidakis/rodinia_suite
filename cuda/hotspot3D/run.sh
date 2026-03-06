#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"

PROFILING=${PROFILING:-0}

if [[ "$PROFILING" == "1" ]]; then
    report="nsys_${benchmark_name}"
    rm -f ${report}.sqlite 2> /dev/null

    nsys profile --trace=cuda --cuda-event-trace=false --sample=none --output ${report} ./${benchmark_name} 512 4 1000 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 result.txt

    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv ${report}.nsys-rep > ${benchmark_name}_nsight.csv
else
    for ((iter=1; iter<=30; iter++))
    do
        ./${benchmark_name} 512 4 1000 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 result.txt &> 512x4_1000_${iter}_${file}
    done

    ../find_avg_per_app.py ${benchmark_name}
fi
