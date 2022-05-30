//
// Created by heeve on 07.05.22.
//

#ifndef CMAKE_AND_CUDA_TRIE_H
#define CMAKE_AND_CUDA_TRIE_H

#include <simt/cstddef>
#include <simt/cstdint>
#include <simt/atomic>

#ifdef TURING
struct trie {
    struct {
        simt::atomic<trie*, simt::thread_scope_device> ptr = ATOMIC_VAR_INIT(nullptr);
        simt::std::atomic_flag flag = ATOMIC_FLAG_INIT;
    } next[26];
    simt::std::atomic<short> count = ATOMIC_VAR_INIT(0);
};
#else
struct trie {
    struct {
        trie *ptr = nullptr;
    } next[26];
    int id = -1;
};

struct trieOptimized {
    int next[26]{};
    int id = -1;
};
#endif

struct trieOptimized {
    int next[26]{};
    int id = -1;
};

#endif //CMAKE_AND_CUDA_TRIE_H
