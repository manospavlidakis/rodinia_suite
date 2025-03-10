#!/usr/bin/env python3
import pandas as pd
import glob
import os
import sys
import numpy as np  # Import numpy for calculations

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

# Metrics of interest
metrics_of_interest = ["Allocation time", "Transfer time", "Compute time", "Transfer Back time"]

# Initialize a dictionary to collect values for each metric
data = {metric: [] for metric in metrics_of_interest}

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

# Compute only the average for selected metrics
average_stats = {metric: np.mean(values) for metric, values in data.items() if values}

# Save the average results to "average_breakdowns.csv"
output_file = os.path.join(output_dir, "average_breakdowns.csv")
stats_df = pd.DataFrame(list(average_stats.items()), columns=["Metric", "Average"])
stats_df.to_csv(output_file, index=False)

print(f"Saved average values to {output_file}")

