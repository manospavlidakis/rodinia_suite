#!/usr/bin/env python3
#######################################################
#       Find the avg of Iterations+Concurrent instances #
# Step1: Find the average of Nxconcurrent instances     #
# Step2: Find the average of Nxiterations               #
# In case of cifar add the two times and then do 1,2    #
#######################################################
# EG : ./find_avg.py -d 03_06_2022_jenna/ -b kernel_without_inout -c 4
import optparse
import os
from statistics import mean
work_bfs = ["1k", "2k", "4k", "8k", "16k", "32k", "64k",
            "128k", "512k", "1M", "2M", "4M"]

work_gaussian = ["256", "512", "1024", "2048"]
work_hotspot = ["512_10M", "512_100M", "1024_1M", "1024_10M"]
work_hotspot3D = ["512x4_10", "512x4_100",
                  "512x4_1000", "512x8_10", "512x8_100", "512x8_1000"]
work_lavaMD = ["20", "30", "40"]
work_nn = ["256k", "512k", "1024k", "2048k"]
work_nw = ["1024", "2048", "4096", "8192"]
work_particlefilter = ["128_10_1000", "128_100_1000", "256_10_10", "256_10_100",
                       "256_10_1000", "256_100_10"]
work_pathfinder = ["1024", "2048", "4096"]
timers = ["elapsed", "init", "compute", "warmup"]
# timers = ["elapsed"]


def parser_func(benc, directory, concurrent):
    print("Call function")
    print("directory: "+directory+" , concurrent: " +
          concurrent+" benchmark: "+benc)

    base_dir = os.getcwd()
    os.chdir(directory+"/"+benc)
    arrayname = "work_"+benc
    # Iterate all workloads
    for work in globals()[arrayname]:
        files_elapsed = []
        files_warmup = []
        files_compute = []
        files_init = []
        # Iterate through all collected timers
        for time in timers:
            for iter in range(10):
                print("work: "+work+" time: "+time+" iter: "+str(iter))
                file = time+'_merged_'+work+'_'+benc + \
                    '_conc-'+concurrent+'_'+str(iter)+'.out'
                timer_array = "files_"+time
                #print("Name: "+timer_array)
                # print("File: "+file)
                locals()[timer_array].append(file)
                # print(file)
                # files_elapsed.append(file)
        # print(files_elapsed)
        avg = []

        for time in timers:
            # print(time)
            timer_array = "files_"+time
            for i in locals()[timer_array]:
                #print("timer: "+i)
                # Open the first file
                with open(i, 'r') as f:
                    data = f.read().split()
                    floats = []
                    # Calculate the AVG
                    for elem in data:
                        # print(elem)
                        try:
                            floats.append(float(elem))
                        except ValueError:
                            pass
                    #print("Float array: "+str(floats))
                    avg.append(mean(floats))
                    floats.clear()
            #print("Avg array: "+str(avg))
            avg_without_max = []
            avg_without_max.clear()
            max_v = 0
            max_v = max(avg)
            # print(avg)
            #print("Exclude max: "+str(max_v))
            for i in avg:
                if i != max_v:
                    avg_without_max.append(i)
            #print("Array to Calculate avg: "+str(avg_without_max))
            # print(mean(avg))
            avg.clear()
            avg2 = 0
            avg2 = mean(avg_without_max)
            with open("AVG_"+time+"_"+work+"_concurrency_"+concurrent+'_'+benc+'.final', "a") as file_object:
                file_object.write(str(avg2)+"\n")
                print("==> AVG_" + time+"_" + work + "concurrency " +
                      concurrent+" benchmark "+benc+": "+str(avg2))
                print("\n")
    os.chdir(base_dir)


# Arguments
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-d", "--directory", dest="dir",
                  help="Provide directory (eg 02_24_2022_jenna/10epochs/mnist/)")
parser.add_option("-c", "--concurrent", dest="concurrent",
                  help="Provide concurrent instances (eg 2)")
(options, args) = parser.parse_args()

# Get the arguments
directory = options.dir
concurrent = options.concurrent

benchmarks = ["bfs", "gaussian", "hotspot", "lavaMD",
              "nn", "nw", "particlefilter", "pathfinder"]
print("Directory of files: "+directory)
print("Concurrency: "+concurrent)

for benc in benchmarks:
    print("Benchmark: "+benc)
    parser_func(benc, directory, concurrent)
