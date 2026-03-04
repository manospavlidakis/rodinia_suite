
#include <hip/hip_runtime.h>
#include <iostream>
__global__ void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw,
        float cn, float cs,
        float ct, float cb,
        float cc)
{
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}
#ifdef BREAKDOWNS
extern "C" {
double g_hotspot3d_alloc_ms = 0.0;
double g_hotspot3d_h2d_ms = 0.0;
double g_hotspot3d_compute_ms = 0.0;
double g_hotspot3d_d2h_ms = 0.0;
double g_hotspot3d_free_ms = 0.0;
}
#endif
void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap,
        float Rx, float Ry, float Rz,
        float dt, int numiter)
{
    float ce, cw, cn, cs, ct, cb, cc;
#ifdef BREAKDOWNS
    auto s_alloc = std::chrono::high_resolution_clock::now();
    auto e_alloc = s_alloc;
    auto s_h2d = s_alloc;
    auto e_h2d = s_alloc;
    auto s_compute = s_alloc;
    auto e_compute = s_alloc;
    auto s_d2h = s_alloc;
    auto e_d2h = s_alloc;
    auto s_free = s_alloc;
    auto e_free = s_alloc;
#endif
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * nx * ny * nz;
    float *tIn_d, *tOut_d, *p_d;
#ifdef BREAKDOWNS
    s_alloc = std::chrono::high_resolution_clock::now();
#endif
    HIP_CHECK(hipMalloc((void**)&p_d,s));
    HIP_CHECK(hipMalloc((void**)&tIn_d,s));
    HIP_CHECK(hipMalloc((void **)&tOut_d, s));
#ifdef BREAKDOWNS
    e_alloc = std::chrono::high_resolution_clock::now();
    s_h2d = std::chrono::high_resolution_clock::now();
#endif
    HIP_CHECK(hipMemcpy(tIn_d, tIn, s, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(p_d, p, s, hipMemcpyHostToDevice));
#ifdef BREAKDOWNS
    e_h2d = std::chrono::high_resolution_clock::now();
#endif
    HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(hotspotOpt1), hipFuncCachePreferL1));
#ifdef BREAKDOWNS
    s_compute = std::chrono::high_resolution_clock::now();
#endif
    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (p_d, tIn_d, tOut_d, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
        float *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
#ifdef DEBUG
        float *in1 = 0, *in2 = 0, *out = 0;
        size_t sz = sizeof(float) * nx * ny * nz;
        in1 = (float *)malloc(sz);
        in2 = (float *)malloc(sz);
        out = (float *)malloc(sz);
        HIP_CHECK(hipMemcpy(in1, tIn_d, sz, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(out, tOut_d, sz, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(in2, p_d, sz, hipMemcpyDeviceToHost));
        for (int i=0; i<nx*ny*nz; i++){
            std::cerr<<"i: "<<i<<" in2 = "<<in2[i]<<" in1 = "<<in1[i]<<", out = "<<out[i]<<std::endl;
        }
#endif

    }
#ifdef DEBUG
      hipError_t err = hipSuccess;
      err = hipGetLastError();
      if (err != hipSuccess){
          std::cerr<<"Error: "<<hipGetErrorString(err)<<" err: "<<err<<std::endl;
          abort();
      }
#endif
    #ifdef BREAKDOWNS
    HIP_CHECK(hipDeviceSynchronize());
    e_compute = std::chrono::high_resolution_clock::now();
    s_d2h = std::chrono::high_resolution_clock::now();
#endif
    HIP_CHECK(hipMemcpy(tOut, tOut_d, s, hipMemcpyDeviceToHost));
#ifdef BREAKDOWNS
    e_d2h = std::chrono::high_resolution_clock::now();
    s_free = std::chrono::high_resolution_clock::now();
#endif
    HIP_CHECK(hipFree(p_d));
    HIP_CHECK(hipFree(tIn_d));
    HIP_CHECK(hipFree(tOut_d));
#ifdef BREAKDOWNS
    e_free = std::chrono::high_resolution_clock::now();
    g_hotspot3d_alloc_ms = std::chrono::duration<double, std::milli>(e_alloc - s_alloc).count();
    g_hotspot3d_h2d_ms = std::chrono::duration<double, std::milli>(e_h2d - s_h2d).count();
    g_hotspot3d_compute_ms = std::chrono::duration<double, std::milli>(e_compute - s_compute).count();
    g_hotspot3d_d2h_ms = std::chrono::duration<double, std::milli>(e_d2h - s_d2h).count();
    g_hotspot3d_free_ms = std::chrono::duration<double, std::milli>(e_free - s_free).count();
#endif
    return;
}
