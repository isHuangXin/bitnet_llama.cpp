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
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>

// mbind constants - avoid requiring libnuma-dev at link time
#ifndef MPOL_BIND
#define MPOL_BIND       2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE    (1 << 1)
#endif
#ifndef MPOL_MF_STRICT
#define MPOL_MF_STRICT  (1 << 0)
#endif

static long pool_mbind(void *addr, unsigned long len, int mode,
                       const unsigned long *nodemask, unsigned long maxnode,
                       unsigned flags) {
    return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}
#endif

#define POOL_ALIGNMENT 64  // cache line size

static struct {
    void   *base;
    size_t  capacity;
    size_t  used;
    size_t  compute_mark;  // offset where compute zone begins
    size_t  weight_mark;   // offset where weights end (for mprotect)
    int     initialized;
} g_pool = { NULL, 0, 0, 0, 0, 0 };

int ggml_pool_init(size_t pool_size_bytes) {
    return ggml_pool_init_numa(pool_size_bytes, -1);
}

int ggml_pool_init_numa(size_t pool_size_bytes, int numa_node) {
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

    // NUMA binding: force physical pages onto the specified NUMA node
    if (numa_node >= 0) {
        unsigned long nodemask = 1UL << numa_node;
        if (pool_mbind(ptr, pool_size_bytes, MPOL_BIND, &nodemask,
                  sizeof(nodemask) * 8, MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
            perror("[ggml-pool] mbind failed (NUMA binding)");
            // Continue anyway - NUMA binding is best-effort
        } else {
            fprintf(stderr, "[ggml-pool] bound to NUMA node %d\n", numa_node);
        }
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

void ggml_pool_protect_weights(void) {
    if (!g_pool.initialized || g_pool.weight_mark == 0) return;

#ifdef __linux__
    // Round down to page boundary for mprotect
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    size_t protect_size = g_pool.weight_mark & ~(page_size - 1);
    if (protect_size > 0) {
        if (mprotect(g_pool.base, protect_size, PROT_READ) != 0) {
            perror("[ggml-pool] mprotect(PROT_READ) failed");
        } else {
            fprintf(stderr, "[ggml-pool] weight region protected as read-only (%.2f MB)\n",
                    (double)protect_size / (1024.0 * 1024.0));
        }
    }
#endif
}

void ggml_pool_mark_weights(void) {
    g_pool.weight_mark = g_pool.used;
    fprintf(stderr, "[ggml-pool] weight mark at offset %zu (%.2f MB)\n",
            g_pool.weight_mark, (double)g_pool.weight_mark / (1024.0 * 1024.0));
}

size_t ggml_pool_get_weight_mark(void) {
    return g_pool.weight_mark;
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

#ifdef __linux__
struct warmup_thread_arg {
    volatile char * base;
    size_t          offset;
    size_t          length;
};

static void * warmup_thread_fn(void * arg) {
    struct warmup_thread_arg * a = (struct warmup_thread_arg *)arg;
    volatile char * p = a->base + a->offset;
    size_t n = a->length;

    // Pass 1: sequential read
    for (size_t i = 0; i < n; i += 64) {
        (void)p[i];
    }
    // Pass 2: read-modify-write for M/E state
    for (size_t i = 0; i < n; i += 64) {
        p[i] = p[i];
    }
    // Pass 3: reverse traversal
    for (size_t i = n > 64 ? n - 64 : 0; i > 0; i -= 64) {
        (void)p[i];
    }
    return NULL;
}
#endif

void ggml_pool_warmup_l3_parallel(int n_threads) {
    if (!g_pool.initialized) return;

    size_t size = g_pool.used;
    if (size == 0) size = g_pool.capacity;

    if (n_threads <= 1) {
        ggml_pool_warmup_l3();
        return;
    }

#ifdef __linux__
    fprintf(stderr, "[ggml-pool] warming up %.2f MB into L3 cache with %d threads...\n",
            (double)size / (1024.0 * 1024.0), n_threads);

    pthread_t * threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    struct warmup_thread_arg * args = (struct warmup_thread_arg *)malloc(n_threads * sizeof(struct warmup_thread_arg));

    size_t chunk = (size + n_threads - 1) / n_threads;
    // Align chunk to cache line
    chunk = (chunk + 63) & ~(size_t)63;

    for (int i = 0; i < n_threads; i++) {
        args[i].base   = (volatile char *)g_pool.base;
        args[i].offset = i * chunk;
        args[i].length = (i == n_threads - 1) ? (size - args[i].offset) : chunk;
        if (args[i].offset >= size) {
            args[i].length = 0;
        }
        pthread_create(&threads[i], NULL, warmup_thread_fn, &args[i]);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);

    fprintf(stderr, "[ggml-pool] warmup complete\n");
#else
    ggml_pool_warmup_l3();
#endif
}
