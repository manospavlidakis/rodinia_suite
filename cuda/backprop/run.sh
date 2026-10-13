#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"

PROFILING=${PROFILING:-0}

if [[ "$PROFILING" == "1" ]]; then
    report="nsys_${benchmark_name}"
    rm -f ${report}.sqlite 2> /dev/null
    nsys profile --trace=cuda --cuda-event-trace=false --sample=none --output ${report} ./${benchmark_name} 4194304

    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv ${report}.nsys-rep > ${benchmark_name}_nsight.csv
else
    rm -f *_"${file}" average.csv min.csv max.csv median.csv iqr.csv std_dev.csv  # stale logs/stats
    fails=0
    for ((iter=1; iter<=30; iter++))
    do
        rm -f result.txt; ./${benchmark_name} 4194304 &>4194304_${iter}_${file} || fails=$((fails+1))
    done

    ../find_avg_per_app.py ${benchmark_name} || fails=$((fails+1))
fi
exit $(( ${fails:-0} > 0 ))
