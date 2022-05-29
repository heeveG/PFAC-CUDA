//
// Created by heeve on 09.05.22.
//

#include <vector>
#include "../headers/util.cuh"

__host__ __device__
int index_of(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    return -1;
};

__global__ void matchWords(const char *str, int *matched, trie *root, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const trie *node;

    for (int iter = i; iter < size; iter += stride) {
        node = root;
        while (i < size) {
            int index = index_of(str[i]);

            if (index == -1)
                break;

            node = node->next[index].ptr;
            if (node == nullptr)
                break;

            if (node->id != -1)
                atomicAdd(&matched[node->id], 1);

            ++i;
        }

        i = iter + stride;
    }
}

__host__
int host_make_trie(trie &root, trie *&bump, const char *begin, const char *end,
                   std::unordered_map<std::string, int> &patternIdMap) {
    int patternId = 0;
    int counter = 0;
    std::string word;
    auto n = &root;
    for (auto pc = begin; pc != end; ++pc) {
        auto const index = index_of(*pc);
        if (index == -1) {
            if (n != &root) {
                if (n->id == -1) { // $todo account for data race in multithreaded
                    n->id = patternId++;
                    patternIdMap.insert(std::make_pair(word, n->id));
                } else
                    counter++;
                n = &root;
            }
            word = "";
            continue;
        }
        word += *pc;
        if (n->next[index].ptr == nullptr)
            n->next[index].ptr = bump++;
        n = n->next[index].ptr;
    }

    std::cout << "same patterns: " << counter << std::endl;
    return patternId;
}

// $TODO - add Turing support
//__host__ __device__
//void device_make_trie(trie &root,
//                   simt::std::atomic<trie *> &bump,
//                   const char *begin, const char *end,
//                   unsigned index,
//                   unsigned domain) {
//
//    auto const size = end - begin;
//    auto const stride = (size / domain + 1);
//
//    auto off = minimum(size, stride * index);
//    auto const last = minimum(size, off + stride);
//
//    for (char c = begin[off]; off < size && off != last && c != 0 && index_of(c) != -1; ++off, c = begin[off]);
//    for (char c = begin[off]; off < size && off != last && c != 0 && index_of(c) == -1; ++off, c = begin[off]);
//
//    trie *n = &root;
//    for (char c = begin[off];; ++off, c = begin[off]) {
//        auto const index = off >= size ? -1 : index_of(c);
//        if (index == -1) {
//            if (n != &root) {
//                n->count.fetch_add(1, simt::std::memory_order_relaxed);
//                n = &root;
//            }
//            //end of last word?
//            if (off >= size || off > last)
//                break;
//            else
//                continue;
//        }
//        if (n->next[index].ptr.load(simt::memory_order_acquire) == nullptr) {
//            if (n->next[index].flag.test_and_set(simt::std::memory_order_relaxed))
//                n->next[index].ptr.wait(nullptr, simt::std::memory_order_acquire);
//            else {
//                auto next = bump.fetch_add(1, simt::std::memory_order_relaxed);
//                n->next[index].ptr.store(next, simt::std::memory_order_release);
//                n->next[index].ptr.notify_all();
//            }
//        }
//        n = n->next[index].ptr.load(simt::std::memory_order_relaxed);
//    }
//}
//
//__global__
//void gpu_make_trie(trie *t, simt::std::atomic<trie *> *bump, const char *begin, const char *end) {
//
//    auto const index = blockDim.x * blockIdx.x + threadIdx.x;
//    auto const domain = gridDim.x * blockDim.x;
//    make_trie(*t, *bump, begin, end, index, domain);
//
//}

bool validateResult(const char *csvPath, std::unordered_map<std::string, int> &patternIdMap, int *matches) {
    std::unordered_map<std::string, int> validMatches;

    // read valid results
    std::ifstream fin(csvPath);
    if (!fin.good()) {
        std::cerr << "Error opening '" << "'. Bailing out." << std::endl;
        exit(-1);
    }

    std::string pattern;
    int count;

    while (fin >> pattern >> count)
        validMatches[pattern] = count;

    fin.close();

    // validate
    for (const auto & match : validMatches)
        if (matches[patternIdMap.at(match.first)] != match.second)
            return false;
    return true;
}
