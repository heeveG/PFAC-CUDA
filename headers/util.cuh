//
// Created by heeve on 07.05.22.
//

#ifndef CMAKE_AND_CUDA_UTIL_CUHH
#define CMAKE_AND_CUDA_UTIL_CUHH

#include <chrono>
#include <atomic>
#include <fstream>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <sstream>

#include "trie.hpp"

#define M 13

template<class T>
static constexpr T minimum(T a, T b) { return a < b ? a : b; }

#define check(ans) { assert_((ans), __FILE__, __LINE__); }

inline void assert_(cudaError_t code, const char *file, int line) {
    if (code == cudaSuccess) return;
    std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
    abort();
}

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_us(const D &d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

__global__ void matchWordsStreams(const unsigned char *str, int *matched, trieOptimized *root, unsigned int streamSize,
                                  unsigned int size);

__global__
void matchWordsSharedMem(unsigned char *str, unsigned int *matched, trieOptimized *root, unsigned int size, unsigned int sharedMemPerBlock);

    __global__
void matchWordsSharedMem2(unsigned char *str, unsigned int *matched, trieOptimized *root, unsigned int size, unsigned int sharedMemPerBlock);

__host__
int host_make_trie(trieOptimized *root, const unsigned char *begin, const unsigned char *end,
                   std::unordered_map<std::string, int> &patternIdMap);

__host__ __device__
void device_make_trie(trie &root, simt::std::atomic<trie *> &bump, const char *begin, const char *end, unsigned index,
                      unsigned domain);

__global__
void gpu_make_trie(trie *t, simt::std::atomic<trie *> *bump, const char *begin, const char *end);

__host__ __device__
int index_of(char c);

bool validateResult(const char *csvPath, std::unordered_map<std::string, int> &patternIdMap, const unsigned int *matches);

#include "util.tcc"

#endif //CMAKE_AND_CUDA_UTIL_CUHH
