// ggml-pool.cpp - Fixed-size memory pool for L3 cache-resident inference
//
// Design: A single contiguous memory region (e.g., 48MB) allocated via mmap
// with MAP_POPULATE (immediate physical page allocation), mlock (prevent swap),
// and MADV_HUGEPAGE (reduce TLB misses). All ggml backend allocations are
// serviced from this pool using a bump allocator.
//
// Memory layout:
//   [0 ... weights ... | ... KV cache ... | compute_mark ... compute buffers ... capacity]
//   ^--- fixed after load ---^              ^--- reset between inferences ---^
//

#include "ggml-pool.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#define POOL_ALIGNMENT 64  // cache line size

static struct {
    void   *base;
    size_t  capacity;
    size_t  used;
    size_t  compute_mark;  // offset where compute zone begins
    int     initialized;
} g_pool = { NULL, 0, 0, 0, 0 };

int ggml_pool_init(size_t pool_size_bytes) {
    if (g_pool.initialized) {
        fprintf(stderr, "[ggml-pool] already initialized\n");
        return 0;
    }

#ifdef __linux__
    // Allocate with MAP_POPULATE to fault all pages immediately
    // MAP_PRIVATE | MAP_ANONYMOUS = no file backing
    void * ptr = mmap(NULL, pool_size_bytes,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                      -1, 0);
    if (ptr == MAP_FAILED) {
        perror("[ggml-pool] mmap failed");
        return -1;
    }

    // Lock pages in RAM - prevent swapping
    if (mlock(ptr, pool_size_bytes) != 0) {
        perror("[ggml-pool] mlock failed (need CAP_IPC_LOCK or sufficient RLIMIT_MEMLOCK)");
        // Continue anyway - mlock is best-effort
    }

    // Request transparent huge pages for reduced TLB pressure
    madvise(ptr, pool_size_bytes, MADV_HUGEPAGE);

    // Zero the entire region (also ensures pages are in page tables)
    memset(ptr, 0, pool_size_bytes);

#else
    // Fallback for non-Linux: aligned malloc
    void * ptr = NULL;
    #ifdef _WIN32
    ptr = _aligned_malloc(pool_size_bytes, POOL_ALIGNMENT);
    #else
    if (posix_memalign(&ptr, POOL_ALIGNMENT, pool_size_bytes) != 0) {
        ptr = NULL;
    }
    #endif
    if (!ptr) {
        fprintf(stderr, "[ggml-pool] allocation failed\n");
        return -1;
    }
    memset(ptr, 0, pool_size_bytes);
#endif

    g_pool.base = ptr;
    g_pool.capacity = pool_size_bytes;
    g_pool.used = 0;
    g_pool.compute_mark = 0;
    g_pool.initialized = 1;

    fprintf(stderr, "[ggml-pool] initialized: %zu MB at %p\n",
            pool_size_bytes / (1024 * 1024), ptr);
    return 0;
}

void ggml_pool_destroy(void) {
    if (!g_pool.initialized) return;

#ifdef __linux__
    munlock(g_pool.base, g_pool.capacity);
    munmap(g_pool.base, g_pool.capacity);
#else
    #ifdef _WIN32
    _aligned_free(g_pool.base);
    #else
    free(g_pool.base);
    #endif
#endif

    g_pool.base = NULL;
    g_pool.capacity = 0;
    g_pool.used = 0;
    g_pool.compute_mark = 0;
    g_pool.initialized = 0;
}

int ggml_pool_is_active(void) {
    return g_pool.initialized;
}

void * ggml_pool_alloc(size_t size) {
    if (!g_pool.initialized) return NULL;

    // Align to cache line
    size_t aligned_size = (size + POOL_ALIGNMENT - 1) & ~(size_t)(POOL_ALIGNMENT - 1);

    if (g_pool.used + aligned_size > g_pool.capacity) {
        fprintf(stderr, "[ggml-pool] EXHAUSTED: requested %zu bytes, "
                "used %zu / %zu (%.2f MB free)\n",
                size, g_pool.used, g_pool.capacity,
                (double)(g_pool.capacity - g_pool.used) / (1024.0 * 1024.0));
        return NULL;
    }

    void * ptr = (char *)g_pool.base + g_pool.used;
    g_pool.used += aligned_size;
    return ptr;
}

void ggml_pool_mark_compute_zone(void) {
    g_pool.compute_mark = g_pool.used;
    fprintf(stderr, "[ggml-pool] compute zone marked at offset %zu (%.2f MB used for weights+KV)\n",
            g_pool.compute_mark, (double)g_pool.compute_mark / (1024.0 * 1024.0));
}

void ggml_pool_reset_compute_zone(void) {
    if (g_pool.compute_mark > 0) {
        g_pool.used = g_pool.compute_mark;
    }
}

size_t ggml_pool_get_used(void) {
    return g_pool.used;
}

size_t ggml_pool_get_capacity(void) {
    return g_pool.capacity;
}

void * ggml_pool_get_base(void) {
    return g_pool.base;
}

void ggml_pool_warmup_l3(void) {
    if (!g_pool.initialized) return;

    size_t size = g_pool.used;
    if (size == 0) size = g_pool.capacity;

    fprintf(stderr, "[ggml-pool] warming up %.2f MB into L3 cache...\n",
            (double)size / (1024.0 * 1024.0));

    volatile char * p = (volatile char *)g_pool.base;

    // Pass 1: sequential read - brings data into cache hierarchy
    for (size_t i = 0; i < size; i += 64) {
        (void)p[i];
    }

    // Pass 2: read-modify-write to ensure M/E state in cache
    // (prevents later write-miss on first modification)
    for (size_t i = 0; i < size; i += 64) {
        p[i] = p[i];
    }

    // Pass 3: reverse traversal to exercise all cache sets
    for (size_t i = size > 64 ? size - 64 : 0; i > 0; i -= 64) {
        (void)p[i];
    }

    fprintf(stderr, "[ggml-pool] warmup complete\n");
}

int ggml_pool_owns(const void * ptr) {
    if (!g_pool.initialized) return 0;
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t base = (uintptr_t)g_pool.base;
    return (addr >= base && addr < base + g_pool.capacity);
}
