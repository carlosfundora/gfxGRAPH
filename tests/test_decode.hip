/**
 * @file test_decode.hip
 * @brief Capture-safe decode pool (hgb_decode_pool_*) — proves the fix for the
 *        decode-attn-under-hipGraph garble.
 *
 * A captured decode graph reads the pool's PERSISTENT device metadata buffers (seq_lens,
 * block_tables). We capture ONE graph per bucket, then replay the SAME bucket graph at two different
 * sequence lengths by refreshing seq_lens in place — and assert the output is correct at BOTH lengths
 * (length-dependent), with a negative control that the two lengths genuinely differ. If the captured
 * graph baked the length, the second replay would garble (track the first length).
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>

// One thread per sequence. Walks the block table for `len` logical positions (len read from the
// DEVICE seq_lens buffer — not a launch arg) and sums the gathered content. With content==1.0 the
// result is exactly the (clamped) sequence length, so a stale-length replay is detectable.
__global__ void decode_sum_kernel(float* out, const float* content, const int* d_seq_lens,
                                  const int* d_block_tables, int max_blocks_per_seq, int page_size,
                                  int max_seq_len) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    int len = d_seq_lens[seq];
    if (len > max_seq_len) len = max_seq_len;   // bucket clamp (mirrors real partition sizing)
    float acc = 0.f;
    for (int s = 0; s < len; s++) {
        int blk  = s / page_size;
        int slot = s % page_size;
        int phys = d_block_tables[seq * max_blocks_per_seq + blk];
        acc += content[phys * page_size + slot];
    }
    out[seq] = acc;
}

struct DecodeCtx {
    float* out;
    float* content;
    int    max_num_seqs;
    int    max_blocks_per_seq;
    int    page_size;
};

static hipError_t capture_decode(int bucket_max, const int* d_seq_lens, const int* d_block_tables,
                                 hipGraph_t* graph, void* ctxv) {
    DecodeCtx* c = (DecodeCtx*)ctxv;
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    // grid = max_num_seqs threads (constant — capture-safe); reads the persistent metadata by pointer.
    decode_sum_kernel<<<1, c->max_num_seqs, 0, stream>>>(
        c->out, c->content, d_seq_lens, d_block_tables,
        c->max_blocks_per_seq, c->page_size, bucket_max);
    hipStreamEndCapture(stream, graph);
    hipStreamDestroy(stream);
    return hipSuccess;
}

int main() {
    printf("=== test_decode (capture-safe decode pool) ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int MAX_NUM_SEQS = 4;
    const int MAX_BLOCKS   = 8;
    const int PAGE_SIZE    = 16;
    const int NUM_PAGES    = MAX_NUM_SEQS * MAX_BLOCKS;   // distinct physical page per (seq, blk)

    // content == 1.0 everywhere -> out[seq] == (clamped) seq_len[seq].
    float* d_content;
    hipMalloc(&d_content, (size_t)NUM_PAGES * PAGE_SIZE * sizeof(float));
    {
        int n = NUM_PAGES * PAGE_SIZE;
        float* h = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h[i] = 1.0f;
        hipMemcpy(d_content, h, (size_t)n * sizeof(float), hipMemcpyHostToDevice);
        free(h);
    }
    float* d_out;
    hipMalloc(&d_out, MAX_NUM_SEQS * sizeof(float));

    DecodeCtx ctx{ d_out, d_content, MAX_NUM_SEQS, MAX_BLOCKS, PAGE_SIZE };

    int buckets[] = {32, 64, 128};
    hgb_decode_pool_t pool;
    err = hgb_decode_pool_create(capture_decode, &ctx, buckets, 3, MAX_NUM_SEQS, MAX_BLOCKS, &pool);
    assert(err == hipSuccess);
    printf("  Created decode pool: %d buckets, max_num_seqs=%d\n", pool.num_buckets, pool.max_num_seqs);

    // Identity-ish block table: logical (seq, blk) -> distinct physical page. num_seqs == MAX_NUM_SEQS.
    int bt[MAX_NUM_SEQS * MAX_BLOCKS];
    for (int s = 0; s < MAX_NUM_SEQS; s++)
        for (int b = 0; b < MAX_BLOCKS; b++)
            bt[s * MAX_BLOCKS + b] = s * MAX_BLOCKS + b;   // < NUM_PAGES

    hipStream_t stream;
    hipStreamCreate(&stream);
    float h_out[MAX_NUM_SEQS];
    int actual = 0;

    // Replay A and B BOTH map to bucket 32 -> they reuse the SAME captured graph. Different lengths,
    // refreshed in place. (max length per call <= 32 so both land on bucket 32.)
    int lensA[MAX_NUM_SEQS] = {10, 20, 5, 0};
    int lensB[MAX_NUM_SEQS] = {30, 25, 31, 28};

    err = hgb_decode_pool_replay(&pool, 20, lensA, MAX_NUM_SEQS, bt, stream, &actual);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);
    hipMemcpy(h_out, d_out, MAX_NUM_SEQS * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Replay A (bucket %d): out = [%.0f %.0f %.0f %.0f], expect [10 20 5 0]\n",
           actual, h_out[0], h_out[1], h_out[2], h_out[3]);
    assert(actual == 32);
    for (int i = 0; i < MAX_NUM_SEQS; i++) assert(h_out[i] == (float)lensA[i]);

    err = hgb_decode_pool_replay(&pool, 31, lensB, MAX_NUM_SEQS, bt, stream, &actual);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);
    hipMemcpy(h_out, d_out, MAX_NUM_SEQS * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Replay B (bucket %d, SAME graph): out = [%.0f %.0f %.0f %.0f], expect [30 25 31 28]\n",
           actual, h_out[0], h_out[1], h_out[2], h_out[3]);
    assert(actual == 32);  // same bucket/graph as A
    for (int i = 0; i < MAX_NUM_SEQS; i++) assert(h_out[i] == (float)lensB[i]);

    // Negative control: A and B answers genuinely differ, so a stale (un-refreshed) graph WOULD garble.
    bool differ = false;
    for (int i = 0; i < MAX_NUM_SEQS; i++) if (lensA[i] != lensB[i]) differ = true;
    assert(differ);
    printf("  Negative control: lengths differ -> a baked-length graph would have garbled ✓\n");

    // Cross-bucket sanity: a longer length routes to bucket 64 and is also correct.
    int lensC[MAX_NUM_SEQS] = {40, 60, 33, 50};
    err = hgb_decode_pool_replay(&pool, 60, lensC, MAX_NUM_SEQS, bt, stream, &actual);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);
    hipMemcpy(h_out, d_out, MAX_NUM_SEQS * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Replay C (bucket %d): out = [%.0f %.0f %.0f %.0f], expect [40 60 33 50]\n",
           actual, h_out[0], h_out[1], h_out[2], h_out[3]);
    assert(actual == 64);
    for (int i = 0; i < MAX_NUM_SEQS; i++) assert(h_out[i] == (float)lensC[i]);

    hgb_decode_pool_destroy(&pool);
    hipFree(d_content);
    hipFree(d_out);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
