#ifdef __cplusplus
extern "C" {
#endif

void
kernel_gpu_cuda_wrapper_2(	knode *knodes, long knodes_elem, long knodes_mem,
							int order, long maxheight, int count, long *currKnode,
							long *offset, long *lastKnode, long *offset_2,
							int *start,	int *end, int *recstart, int *reclength);

#ifdef BREAKDOWNS
// Exposed breakdown times (ms) set by kernel_gpu_cuda_wrapper_2
extern double g_btree2_alloc_ms;
extern double g_btree2_h2d_ms;
extern double g_btree2_compute_ms;
extern double g_btree2_d2h_ms;
extern double g_btree2_free_ms;
#endif

#ifdef __cplusplus
}
#endif
