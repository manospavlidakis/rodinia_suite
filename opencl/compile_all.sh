#!/bin/bash
#cd bfs/
#aoc bfs.cl -o bfs.aocx -board=de5a_net_ddr4 -v
#cd -
cd gaussian
aoc gaussianElim_kernels.cl -o gaussianElim_kernels.aocx -board=de5a_net_ddr4 -v
cd -
#cd hotspot
#aoc hotspot.cl -o hotspot.aocx -board=de5a_net_ddr4 -v
#cd -
cd hotspot3D
aoc hotspot3D_kernel_v0.cl -o hotspot3D_kernel_v0.aocx -board=de5a_net_ddr4 -v
cd -
#cd lavaMD/
#aoc lavaMD_kernel_v0.cl -o lavaMD_kernel_v0.aocx -board=de5a_net_ddr4 -v
#cd -
cd nn
aoc nn_kernel_v0.cl -o nn_kernel_v0.aocx -board=de5a_net_ddr4 -v
cd -
cd nw
aoc nw_kernel_v0.cl -o nw_kernel_v0.aocx -board=de5a_net_ddr4 -v
cd -
cd pathfinder/
aoc pathfinder_kernel_v0.cl -o pathfinder_kernel_v0.aocx -board=de5a_net_ddr4 -v
cd -
cd particlefilter
aoc particle_single.cl -o particle_single.aocx -board=de5a_net_ddr4 -v
