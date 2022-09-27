from cffi import FFI

ffi = FFI()
ffi.set_source('_khash_ffi', '#include "khash_int2int.h"')

ffi.cdef('''\
typedef int... khint64_t;

static inline void *khash_int2int_init(void);
static void khash_int2int_destroy(void *);
static inline khint64_t khash_int2int_get(void *, khint64_t, khint64_t);
static inline int khash_int2int_set(void *, khint64_t, khint64_t);
''')

if __name__ == '__main__':
    ffi.compile(verbose=True)
