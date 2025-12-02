#!/usr/bin/env python3
import pandas as pd
import glob
import os

# List of benchmarks to process
benchmarks = ["bfs", "gaussian", "hotspot", "hotspot3D", "lavaMD",
              "nn", "nw", "particlefilter", "pathfinder"]

# Directory to save the average files
output_dir = "averages"
os.makedirs(output_dir, exist_ok=True)

# Iterate over each benchmark
for benchmark in benchmarks:
    # File pattern to match any directory and filenames ending with _{benchmark}.csv
    file_pattern = f"**/*_{benchmark}.csv"
    csv_files = glob.glob(file_pattern, recursive=True)
    
    # Check if there are any files for this benchmark
    if not csv_files:
        print(f"No files found for benchmark '{benchmark}'. Skipping...")
        continue
    
    # Initialize a dictionary to collect values for averaging
    data = {"Init time": [], "Computation": [], "Elapsed time": [], "Warmup time": []}
    
    # Process each CSV file for the current benchmark
    for file in csv_files:
        # Open and read the file line by line
        with open(file, 'r') as f:
            for line in f:
                # Split the line into metric and value parts
                if ':' in line:
                    metric, value = line.split(':')
                    metric = metric.strip()
                    # Convert value to float after stripping "ms" and spaces
                    value = float(value.replace('ms', '').strip())
                    # Add the value to the appropriate list in the data dictionary
                    if metric in data:
                        data[metric].append(value)
    
    # Calculate the average of each metric
    averages = {metric: (sum(values) / len(values)) for metric, values in data.items() if values}
    
    # Convert averages dictionary to a DataFrame for easy saving
    averages_df = pd.DataFrame(list(averages.items()), columns=["Metric", "Average Value"])
    
    # Save the averages to a new CSV file for this benchmark
    output_file = os.path.join(output_dir, f"average_{benchmark}.csv")
    averages_df.to_csv(output_file, index=False)
    
    print(f"Average values for '{benchmark}' have been written to '{output_file}'")

