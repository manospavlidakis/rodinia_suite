#ifndef _IMAGENET_H_
#define _IMAGENET_H_
#ifdef __cplusplus
extern "C" {
#endif

void bpnn_train_cuda(BPNN *net, float *eo, float *eh);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj,
                       float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                       float **who, float *hidden, float *err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                         float **w, float **oldw);
void load(BPNN *net);

int setup(int argc, char **argv);
#ifdef __cplusplus
}
#endif

#endif
