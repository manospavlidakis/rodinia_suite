# Rodinia benchmark suite
## Modifications versus native version include
-- Timers that count the same blocks for all versions

-- Opencl version adjusted to work on intel aria 10 FPGA
# Build 
## Go to cuda/hip dir and use buildall.sh 
# Run
## Go to cuda/hip and use runRodinia.sh
# Timers
## Go to cuda/hip and use find_avg.py
## find_avg.py creates an average dir that contains the avg of 5 runs
