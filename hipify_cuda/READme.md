# Convert CUDA to HIP using hipify-clang
- I used the original cuda implementation along with hipify to generate this code
- To regenerate this code use the hipify_script.sh <app_dir>
- BFS requires fixes
  1. Create a header file and add MAX_THREAD_PER_BLOCK and node struct. 
- LavaMD requires fixes
  1. For simplicity merge kkernel_gpu_cuda.cu and kernel_gpu_cuda_wrapper.cu
  2. Do not forget to hipify files in kernel and util/device dirs

