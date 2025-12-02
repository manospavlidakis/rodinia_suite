#ifndef UTIL_H_
#define UTIL_H_

#define ALIGNMENT 64
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

inline static void *alignedMalloc(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, ALIGNMENT, size)) {
    fprintf(stderr, "Aligned Malloc failed due to insufficient memory.\n");
    exit(-1);
  }
  return ptr;
}
inline static void fatal(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

/* For printing to FILE* (stderr, fp, etc.) */
#define FPRINTF_CHECK(stream, fmt, ...)                                         \
  do {                                                                          \
    if (fprintf((stream), (fmt), ##__VA_ARGS__) < 0) {                          \
      fprintf(stderr, "ERROR: fprintf failed (%s:%d)\n", __FILE__, __LINE__);   \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)


#define FGETS_CHECK(dst, size, fp) do {                                         \
  if (fgets((dst), (size), (fp)) == NULL) {                                     \
    if (feof((fp))) {                                                           \
      fprintf(stderr, "ERROR: unexpected EOF (%s:%d)\n", __FILE__, __LINE__);   \
    } else {                                                                    \
      fprintf(stderr, "ERROR: fgets failed (%s:%d): %s\n",                      \
              __FILE__, __LINE__, strerror(errno));                             \
    }                                                                           \
    exit(1);                                                                    \
  }                                                                             \
} while (0)

/* fscanf that checks expected conversions */
#define FSCANF_CHECKN(fp, expected, fmt, ...) do {                              \
  int _n = fscanf((fp), (fmt), ##__VA_ARGS__);                                  \
  if (_n != (expected)) {                                                       \
    fprintf(stderr, "ERROR: fscanf failed (%s:%d): expected %d, got %d\n",      \
            __FILE__, __LINE__, (expected), _n);                                \
    exit(1);                                                                    \
  }                                                                             \
} while (0)

/* Convenience: at least succeeded (>=0) */
#define FSCANF_CHECK(fp, fmt, ...) do {                                         \
  int _n = fscanf((fp), (fmt), ##__VA_ARGS__);                                  \
  if (_n < 0) {                                                                  \
    fprintf(stderr, "ERROR: fscanf failed (%s:%d)\n", __FILE__, __LINE__);      \
    exit(1);                                                                    \
  }                                                                             \
} while (0)

/* scanf that checks expected conversions */
#define SCANF_CHECKN(expected, fmt, ...) do {                                   \
  int _n = scanf((fmt), ##__VA_ARGS__);                                         \
  if (_n != (expected)) {                                                       \
    fprintf(stderr, "ERROR: scanf failed (%s:%d): expected %d, got %d\n",       \
            __FILE__, __LINE__, (expected), _n);                                \
    exit(1);                                                                    \
  }                                                                             \
} while (0)

#define SNPRINTF_CHECK(dst, fmt, ...) do {                                \
  int _n = snprintf((dst), sizeof(dst), (fmt), __VA_ARGS__);              \
  if (_n < 0 || _n >= (int)sizeof(dst)) {                                 \
    fprintf(stderr, "ERROR: snprintf overflow (%s:%d)\n", __FILE__, __LINE__); \
    exit(1);                                                             \
  }                                                                       \
} while (0)

/* Explicit capacity version (always safe) */
#define SNPRINTF_CHECKA(dst, dst_cap, fmt, ...) do {                          \
  int _n = snprintf((dst), (dst_cap), (fmt), ##__VA_ARGS__);                  \
  if (_n < 0 || _n >= (int)(dst_cap)) {                                       \
    fprintf(stderr, "ERROR: snprintf overflow (%s:%d)\n", __FILE__, __LINE__);\
    exit(1);                                                                 \
  }                                                                          \
} while (0)

/* Convenience: the common case = read exactly 1 item */
#define FSCANF_CHECK1(fp, fmt, a1) FSCANF_CHECKN((fp), 1, (fmt), (a1))
#define SCANF_CHECK1(fmt, a1)      SCANF_CHECKN(1, (fmt), (a1))

/* fread must read exactly "count" items of size "size" */
#define FREAD_CHECK(ptr, size, count, fp)                                      \
  do {                                                                         \
    size_t _r = fread((ptr), (size), (count), (fp));                            \
    if (_r != (size_t)(count)) {                                               \
      if (feof((fp)))                                                          \
        fprintf(stderr, "ERROR: unexpected EOF (%s:%d)\n", __FILE__, __LINE__);\
      else {                                                                   \
        perror("fread");                                                       \
        fprintf(stderr, "ERROR: fread failed (%s:%d)\n", __FILE__, __LINE__);  \
      }                                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#endif /* UTIL_H_ */
