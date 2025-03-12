#ifndef _IMAGENET_H_
#define _IMAGENET_H_

void load(BPNN *net);
void bpnn_train_cuda(BPNN *net, float *eo, float *eh);
#endif
