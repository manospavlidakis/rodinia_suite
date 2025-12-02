#!/usr/bin/env python3

import os
import glob
import csv

# Define the output summary file
output_file = "computation_summary.csv"

# Define the patterns to search for
file_patterns = ["**/max.csv", "**/min.csv", "**/std_dev.csv"]

# Open the summary file in append mode
with open(output_file, "a") as summary:
    # Write header if the file is empty
    if os.stat(output_file).st_size == 0:
        summary.write("Filename,Computation Time (ms)\n")

    # Search for each pattern
    for pattern in file_patterns:
        for file in glob.glob(pattern, recursive=True):
            with open(file, "r") as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader, None)  # Read the first line (header)

                for row in csv_reader:
                    if row and row[0].strip() == "Computation":
                        computation_value = row[1].strip()  # Extract the value
                        summary.write(f"{file},{computation_value}\n")
                        break  # Stop reading once we find "Computation"

print(f"Computation times appended to {output_file}")
