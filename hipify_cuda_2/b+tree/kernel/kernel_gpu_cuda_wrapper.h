#ifdef __cplusplus
extern "C" {
#endif

void
kernel_gpu_cuda_wrapper(record *records, long records_mem, knode *knodes, long knodes_elem,
						long knodes_mem, int order,	long maxheight,	int count, long *currKnode,
						long *offset, int *keys, record *ans);

#ifdef BREAKDOWNS
// Exposed breakdown times (ms) set by kernel_gpu_cuda_wrapper_2
extern double g_btree1_alloc_ms;
extern double g_btree1_h2d_ms;
extern double g_btree1_compute_ms;
extern double g_btree1_d2h_ms;
extern double g_btree1_free_ms;
#endif

#ifdef __cplusplus
}
#endif
