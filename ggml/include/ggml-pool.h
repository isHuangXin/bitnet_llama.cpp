#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fixed-size memory pool for L3 cache-resident inference
// All allocations come from a single contiguous mmap'd region

// Initialize the memory pool with a fixed size (e.g., 48MB)
// Uses MAP_POPULATE + mlock + MADV_HUGEPAGE for guaranteed physical residency
// Returns 0 on success, -1 on failure
int ggml_pool_init(size_t pool_size_bytes);

// Destroy the memory pool
void ggml_pool_destroy(void);

// Check if pool is initialized and active
int ggml_pool_is_active(void);

// Allocate from the pool (bump allocator, 64-byte aligned)
// Returns NULL if pool exhausted
void * ggml_pool_alloc(size_t size);

// Mark a region as "compute zone" - can be reset between inferences
// Call after model weights + KV cache are allocated to mark the compute boundary
void ggml_pool_mark_compute_zone(void);

// Reset compute zone (free all compute buffers, keep weights + KV cache)
void ggml_pool_reset_compute_zone(void);

// Get pool usage statistics
size_t ggml_pool_get_used(void);
size_t ggml_pool_get_capacity(void);
void * ggml_pool_get_base(void);

// Warmup: sequential read over all used memory to fill L3 cache
// Call after model loading and before benchmark timing
void ggml_pool_warmup_l3(void);

// Check if a pointer belongs to the pool (used to skip free)
int ggml_pool_owns(const void * ptr);

#ifdef __cplusplus
}
#endif
