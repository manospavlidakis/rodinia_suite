# Rodinia Benchmark suite
This repository provides scripts to **build, run, collect results, and plot performance metrics** for Rodinia benchmarks across multiple GPU frameworks (CUDA, HIP, SCALE, SYCL). There are three main differences from the original rodinia repo (https://rodinia.cs.virginia.edu/):

    1. We introduce consistent timers to compare computation time (that includes all gpu related calls).

    2. We add breakdowns to meassure individual parts of computation time (i.e., Allocation, Transfer, Kernel, and Free time).

    3. We removed deprecated CUDA calls and support latest cuda version (>13.0).
---

# Repository Structure

```
common/              # Common functions
data/                # Datasets used for evaluation
cuda/                # CUDA implementation
hipify_cuda_2/       # HIP implementation using hipify
sycl/                # SYCL implementation (will be added soon)
plot_*.py            # Plotting scripts
run_plot_*.sh        # Plot automation scripts
run_all.sh           # Builds and run everything
hip                  # Deprecated (hand porting of cuda)
hipify_cuda          # Old hipified version

```

---

# Build

Navigate to either the CUDA or HIP directory and build all benchmarks:

```bash
cd cuda
./buildall.sh
````

or

```bash
cd hipify_cuda_2
./buildall.sh

```

### Options

Default behavior:

* Automatically detects CUDA or ROCm installation
* Uses default architecture (`SM=86` for CUDA)

Manual configuration:

CUDA:

```bash
./buildall.sh --cuda <cuda_path> --sm <sm_version>
```

ROCm:

```bash
./buildall.sh --hip <rocm_path> --gfx <gpu_arch>
```

---

# Run Benchmarks

Run all Rodinia benchmarks once:

```bash
./runRodinia.sh
```

---

# Run Multiple Iterations

Use the automated runner:

```bash
./runRodiniaWithIntervals.sh
```

### Options:

Run benchmarks again:

```bash
RERUN_APPS=1 ./runRodiniaWithIntervals.sh
```

Specify number of runs:

```bash
RUNS=5 ./runRodiniaWithIntervals.sh
```

Add delay between runs:

```bash
RUNS=5 SLEEP_SECS=600 ./runRodiniaWithIntervals.sh
```

Select GPU device:

```bash
GPU_ID=0 ./runRodiniaWithIntervals.sh
```

---

# Execution Breakdowns

To enable detailed breakdown metrics (allocation, transfers, compute):

1. Enable instrumentation and rebuild via `buildall.sh`:

```bash
CXXFLAGS+=" -DBREAKDOWNS"
```

2. Run:

```bash
DO_BREAKDOWNS=1 ./runRodiniaWithIntervals.sh
```

---

# Plot Results

## Computation Time

Generate a grouped bar chart comparing CUDA / HIP / SCALE / SYCL:

```bash
./run_plot_computation.sh
```

Optional logarithmic scale:

```bash
LOGY=1 ./run_plot_computation.sh
```

---

## Execution Breakdowns

Generate stacked breakdown plots:

```bash
./run_plot_breakdowns.sh
```

Options:

Logarithmic scale:

```bash
LOGY=1 ./run_plot_breakdowns.sh
```

Compress extreme outliers (e.g., very large transfers):

```bash
SQUASH_OUTLIERS=1 ./run_plot_breakdowns.sh
```

---

# Result Validation

Reference outputs are generated using **NVIDIA GPUs**.

Results from other architectures (HIP, SCALE, SYCL) are validated by comparing them against the CUDA baseline outputs.

```

