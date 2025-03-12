#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h> // (in library path known to compiler)		needed by printf

void setdevice(void);

void checkCUDAError(const char *msg);
#ifdef __cplusplus
}
#endif
