#!/bin/bash

# Check if a directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_with_cu_files>"
    exit 1
fi

# Assign the first argument as the input directory
INPUT_DIR="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: '$INPUT_DIR' is not a directory."
    exit 1
fi

# Process all .cu files in the given directory
for file in "$INPUT_DIR"/*.cu; do
    if [ -f "$file" ]; then  # Check if file exists
        echo "Converting $file to HIP..."
        hipify-clang "$file" --cuda-path=/usr/local/cuda -o "${file%.cu}_hip.cpp"
    else
        echo "No .cu files found in $INPUT_DIR"
    fi
done

echo "Conversion completed!"

