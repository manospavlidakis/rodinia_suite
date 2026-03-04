#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include "./../main.h"
#include "./kernel_gpu_cuda_wrapper.h"
#include "../../../common/helper_hip.h"
//#define BREAKDOWNS
#ifdef BREAKDOWNS
std::chrono::high_resolution_clock::time_point s_b0;
std::chrono::high_resolution_clock::time_point e_b0;
std::chrono::high_resolution_clock::time_point s_b1;
std::chrono::high_resolution_clock::time_point e_b1;
std::chrono::high_resolution_clock::time_point s_b2;
std::chrono::high_resolution_clock::time_point e_b2;
std::chrono::high_resolution_clock::time_point s_b3;
std::chrono::high_resolution_clock::time_point e_b3;
std::chrono::high_resolution_clock::time_point s_b4;
std::chrono::high_resolution_clock::time_point e_b4;
#endif
__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                box_str* d_box_gpu,     FOUR_VECTOR* d_rv_gpu,
                fp* d_qv_gpu, FOUR_VECTOR* d_fv_gpu) {

        int bx = blockIdx.x;
        int tx = threadIdx.x;
        int wtx = tx;

        if(bx<d_dim_gpu.number_boxes){
                // parameters
                fp a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

                // home box
                int first_i;
                FOUR_VECTOR* rA;
                FOUR_VECTOR* fA;
                __shared__ FOUR_VECTOR rA_shared[100];

                // nei box
                int pointer;
                int k = 0;
                int first_j;
                FOUR_VECTOR* rB;
                fp* qB;
                int j = 0;
                __shared__ FOUR_VECTOR rB_shared[100];
                __shared__ double qB_shared[100];
		// common
                fp r2;
                fp u2;
                fp vij;
                fp fs;
                fp fxij;
                fp fyij;
                fp fzij;
                THREE_VECTOR d;

                // home box - box parameters
                first_i = d_box_gpu[bx].offset;

                // home box - distance, force, charge and type parameters
                rA = &d_rv_gpu[first_i];
                fA = &d_fv_gpu[first_i];

                // home box - shared memory
                while(wtx<NUMBER_PAR_PER_BOX){
                        rA_shared[wtx] = rA[wtx];
                        wtx = wtx + NUMBER_THREADS;
                }
                wtx = tx;

                // synchronize threads  - not needed, but just to be safe
                __syncthreads();

                // loop over neiing boxes of home box
                for (k=0; k<(1+d_box_gpu[bx].nn); k++){
                        if(k==0){
				       pointer = bx;
                        }
                        else{
                                pointer = d_box_gpu[bx].nei[k-1].number;
                        }

                        // nei box - box parameters
                        first_j = d_box_gpu[pointer].offset;

                        // nei box - distance, (force), charge and (type) parameters
                        rB = &d_rv_gpu[first_j];
                        qB = &d_qv_gpu[first_j];

                        // nei box - shared memory
                        while(wtx<NUMBER_PAR_PER_BOX){
                                rB_shared[wtx] = rB[wtx];
                                qB_shared[wtx] = qB[wtx];
                                wtx = wtx + NUMBER_THREADS;
                        }
                        wtx = tx;

                        __syncthreads();
			// for (int i=0; i<nTotal_i; i++){
                        while(wtx<NUMBER_PAR_PER_BOX){

                                // loop for the number of particles in the current nei box
                                for (j=0; j<NUMBER_PAR_PER_BOX; j++){

                                        r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v - DOT((fp)rA_shared[wtx],(fp)rB_shared[j]);
                                        u2 = a2*r2;
                                        vij= exp(-u2);
                                        fs = 2*vij;

                                        d.x = (fp)rA_shared[wtx].x  - (fp)rB_shared[j].x;
                                        fxij=fs*d.x;
                                        d.y = (fp)rA_shared[wtx].y  - (fp)rB_shared[j].y;
                                        fyij=fs*d.y;
                                        d.z = (fp)rA_shared[wtx].z  - (fp)rB_shared[j].z;
                                        fzij=fs*d.z;

                                        fA[wtx].v +=  (double)((fp)qB_shared[j]*vij);
                                        fA[wtx].x +=  (double)((fp)qB_shared[j]*fxij);
                                        fA[wtx].y +=  (double)((fp)qB_shared[j]*fyij);
					fA[wtx].z +=  (double)((fp)qB_shared[j]*fzij);

                                }

                                // increment work thread index
                                wtx = wtx + NUMBER_THREADS;
                        }

                        // reset work index
                        wtx = tx;

                        __syncthreads();
                }

                }
	}

void kernel_gpu_cuda_wrapper(par_str par_cpu, dim_str dim_cpu, box_str *box_cpu,
                             FOUR_VECTOR *rv_cpu, fp *qv_cpu,
                             FOUR_VECTOR *fv_cpu) {
  box_str *d_box_gpu;
  FOUR_VECTOR *d_rv_gpu;
  fp *d_qv_gpu;
  FOUR_VECTOR *d_fv_gpu;

  dim3 threads;
  dim3 blocks;

  blocks.x = dim_cpu.number_boxes;
  blocks.y = 1;
  threads.x = NUMBER_THREADS;
  threads.y = 1;
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif
  HIP_CHECK(hipMalloc((void **)&d_box_gpu, dim_cpu.box_mem));
  HIP_CHECK(hipMalloc((void **)&d_rv_gpu, dim_cpu.space_mem));
  HIP_CHECK(hipMalloc((void **)&d_qv_gpu, dim_cpu.space_mem2));
  HIP_CHECK(hipMalloc((void **)&d_fv_gpu, dim_cpu.space_mem));
#ifdef BREAKDOWNS
  HIP_CHECK(hipDeviceSynchronize());
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, hipMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif
  // launch kernel - all boxes
  kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, d_box_gpu, d_rv_gpu,
                                       d_qv_gpu, d_fv_gpu);
#ifdef BREAKDOWNS
  HIP_CHECK(hipDeviceSynchronize());
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, hipMemcpyDeviceToHost));
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
  s_b4 = std::chrono::high_resolution_clock::now();
#endif
  HIP_CHECK(hipFree(d_rv_gpu));
  HIP_CHECK(hipFree(d_qv_gpu));
  HIP_CHECK(hipFree(d_fv_gpu));
  HIP_CHECK(hipFree(d_box_gpu));
#ifdef BREAKDOWNS
  e_b4 = std::chrono::high_resolution_clock::now();
#endif

  #ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown Computation #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation = e_b0 - s_b0;
  std::cerr << "Allocation time: " << allocation.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer = e_b2 - s_b2;
  std::cerr << "H2D transfer time: " << transfer.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute = e_b1 - s_b1;
  std::cerr << "Compute time: " << compute.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer2 = e_b3 - s_b3;
  std::cerr << "D2H transfer time: " << transfer2.count() << " ms"
            << std::endl;
  std::chrono::duration<double, std::milli> freetime = e_b4 - s_b4;
  std::cerr << "Free time: " << freetime.count() << " ms" << std::endl;
  std::cerr << " #################################" << std::endl;
#endif
}
