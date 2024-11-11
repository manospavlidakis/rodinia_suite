#ifndef UTIL_H_
#define UTIL_H_

#define ALIGNMENT 64
#include <stdio.h>
#include <stdlib.h>

inline static void *alignedMalloc(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, ALIGNMENT, size)) {
    fprintf(stderr, "Aligned Malloc failed due to insufficient memory.\n");
    exit(-1);
  }
  return ptr;
}

#endif /* UTIL_H_ */
