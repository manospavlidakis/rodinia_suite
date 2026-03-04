#!/usr/bin/env python3
import glob
import os
import re
import sys
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python find_avg_per_app_break.py <benchmark>")
    sys.exit(1)

benchmark = sys.argv[1]
output_dir = "."

file_pattern = f"**/*_{benchmark}.csv"
csv_files = glob.glob(file_pattern, recursive=True)

DEFAULT_METRICS = [
    "Allocation time",
    "H2D transfer time",
    "Compute time",
    "D2H transfer time",
    "Free time",
]

out_path = os.path.join(output_dir, "average_breakdowns.csv")

# If no breakdowns exist, emit a file with the expected metrics (blank values) and exit cleanly.
if not csv_files:
    df = pd.DataFrame([{"Metric": m, "Average_ms": ""} for m in DEFAULT_METRICS])
    df.to_csv(out_path, index=False)
    print(f"No breakdown files for '{benchmark}'. Wrote empty template to {out_path}")
    sys.exit(0)

data = {m: [] for m in DEFAULT_METRICS}

# matches: "<metric>: <number> ms" where metric may include extra text (e.g., "(...)")
line_re = re.compile(r"^\s*(?P<metric>[^:]+)\s*:\s*(?P<value>[-+0-9.eE]+)\s*ms\s*$")

def canonical_metric(raw: str) -> str | None:
    raw = raw.strip()

    # Canonical (exact or with parenthetical suffix)
    for canon in DEFAULT_METRICS:
        if raw == canon:
            return canon
        if raw.startswith(canon + " ") or raw.startswith(canon + "("):
            return canon

    # Heartwall special labels
    # "H2D transfer (memcpy+memcpy2sym)" -> "H2D transfer time"
    if raw.startswith("H2D transfer") and not raw.startswith("H2D transfer time"):
        return "H2D transfer time"

    # If ever you get "D2H transfer (...)" etc, you can add similar mappings.
    return None

for file in csv_files:
    with open(file, "r", errors="ignore") as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            raw_metric = m.group("metric").strip()
            metric = canonical_metric(raw_metric)
            if metric is None:
                continue
            try:
                data[metric].append(float(m.group("value")))
            except ValueError:
                print(f"Warning: Skipping invalid value in {file}: {line.strip()}")

rows = []
for metric in DEFAULT_METRICS:
    vals = data[metric]
    avg = round(float(np.mean(vals)), 2) if vals else ""
    rows.append({"Metric": metric, "Average_ms": avg})

pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Saved average values to {out_path}")
