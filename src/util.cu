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
}

__host__
int host_make_trie(trieOptimized *root, const char *begin, const char *end,
                    std::unordered_map<std::string, int> &patternIdMap) {
    int patternId = 0;
    int bump = 0;
    std::string word;

    auto n = root;
    for (auto pc = begin; pc != end; ++pc) {
        auto const index = index_of(*pc);
        if (index == -1) {
            if (n != root) {
                if (n->id == -1) { // $todo account for data race in multithreaded
                    n->id = patternId++;
                    patternIdMap.insert(std::make_pair(word, n->id));
                }
                n = root;
            }
            word = "";
            continue;
        }
        word += tolower(*pc);
        if (n->next[index] == 0)
            n->next[index] = ++bump;
        n = root + n->next[index];
    }

    return ++bump;
}

__global__ void matchWords(const char *str, int *matched, trieOptimized *root, int streamSize, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const trieOptimized *node;

    for (int iter = i; iter < streamSize; iter += stride) {
        node = root;
        while (i < size) {
            int index = index_of(str[i]);

            if (index == -1)
                break;

            int nextIndex = node->next[index];
            if (nextIndex == 0)
                break;

            node = root + node->next[index];

            if (node->id != -1) {
                atomicAdd(&matched[node->id], 1);
            }
            ++i;
        }

        i = iter + stride;
    }
}

bool validateResult(const char *csvPath, std::unordered_map<std::string, int> &patternIdMap, const int *matches) {
    std::unordered_map<std::string, int> validMatches;

    // read valid results
    std::ifstream fin(csvPath);
    if (!fin.good()) {
        std::cerr << "Error opening '" << "'. Bailing out." << std::endl;
        exit(-1);
    }

    std::string line, pattern, count;
    std::stringstream ss;
    char *endP;
    while (getline(fin, line)) {
        ss << line;
        getline(ss, pattern, ',');
        getline(ss, count, ',');
        validMatches.insert(std::make_pair(pattern, strtol(count.data(), &endP, 10)));
        ss.clear();
    }

    fin.close();

    // validate
    for (const auto &match : validMatches)
        if (patternIdMap.find(match.first) == patternIdMap.end() ||
            matches[patternIdMap.at(match.first)] != match.second) {
            std::cout << match.first << " " << match.second << " " << matches[patternIdMap.at(match.first)]
                      << std::endl;
//            return false;
        }
    return patternIdMap.size() == validMatches.size();
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