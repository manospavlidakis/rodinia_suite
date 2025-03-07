#!/usr/bin/env python3
import pandas as pd
import glob
import os
import sys
import numpy as np  # Import numpy for standard deviation calculation

# Check if a benchmark argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <benchmark>")
    sys.exit(1)

# Get the benchmark from the command-line argument
benchmark = sys.argv[1]

# Directory to save the result files
output_dir = "."
# os.makedirs(output_dir, exist_ok=True)

# File pattern to match any directory and filenames ending with _{benchmark}.csv
file_pattern = f"**/*_{benchmark}.csv"
csv_files = glob.glob(file_pattern, recursive=True)

# Check if there are any files for this benchmark
if not csv_files:
    print(f"No files found for benchmark '{benchmark}'. Exiting...")
    sys.exit(1)

# Initialize a dictionary to collect values for each metric
data = {"Init time": [], "Computation": [], "Elapsed time": [], "Warmup time": []}

# Process each CSV file for the specified benchmark
for file in csv_files:
    with open(file, 'r') as f:
        for line in f:
            if ':' in line:
                metric, value = line.split(':')
                metric = metric.strip()
                try:
                    value = float(value.replace('ms', '').strip())  # Convert to float
                    if metric in data:
                        data[metric].append(value)
                except ValueError:
                    print(f"Warning: Skipping invalid value in file {file}: {line.strip()}")

# Compute statistics and save each to a separate CSV file
stats_dict = {
    "average.csv": {metric: np.mean(values) for metric, values in data.items() if values},
    "min.csv": {metric: np.min(values) for metric, values in data.items() if values},
    "max.csv": {metric: np.max(values) for metric, values in data.items() if values},
    "std_dev.csv": {metric: np.std(values, ddof=1) for metric, values in data.items() if values},  # Sample std dev
}

# Save each statistic to its respective CSV file
for filename, stats in stats_dict.items():
    output_file = os.path.join(output_dir, filename)
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", filename.replace('.csv', '').capitalize()])
    stats_df.to_csv(output_file, index=False)
    print(f"Saved {filename} in {output_dir}")

print("All statistics have been computed and saved separately.")
