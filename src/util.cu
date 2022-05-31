//
// Created by heeve on 09.05.22.
//

#include <vector>
#include "../headers/util.cuh"

__host__ __device__
int index_of(unsigned char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    return -1;
}

__host__
int host_make_trie(trieOptimized *root, const unsigned char *begin, const unsigned char *end,
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

__global__ void matchWordsStreams(const unsigned char *str, int *matched, trieOptimized *root, unsigned int streamSize,
                                  unsigned int size) {
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

__global__ void
matchWordsSharedMem(unsigned char *str, unsigned int *matched, trieOptimized *root, unsigned int size,
                    unsigned int sharedMemPerBlock) {
    const trieOptimized *node;
    int charIndex;
    unsigned int nextIndex, currentChar;
    unsigned int threadChunk = sharedMemPerBlock / blockDim.x;

    unsigned int startThread = threadChunk * threadIdx.x;
    unsigned int stopThread = startThread + threadChunk;

    extern __shared__ unsigned char sArray[];

    for (unsigned int globalIndex = blockIdx.x * sharedMemPerBlock;
         globalIndex < size; globalIndex += gridDim.x * sharedMemPerBlock) {

        for (unsigned int tSharedIndex = threadIdx.x, tGlobalIndex = globalIndex + threadIdx.x;
             (tSharedIndex < sharedMemPerBlock + M - 1) && tGlobalIndex < size;
             tSharedIndex += blockDim.x, tGlobalIndex += blockDim.x)
            sArray[tSharedIndex] = str[tGlobalIndex];

        __syncthreads();

        for (unsigned int startChar = startThread; (startChar < stopThread &&
                                                    globalIndex + startChar < size); ++startChar) {
            node = root;
            currentChar = startChar;
            while ((charIndex = index_of(sArray[currentChar])) != -1 && (nextIndex = node->next[charIndex]) != 0) {
                node = root + nextIndex;
                int nodeId = node->id;
                if (nodeId != -1) {
                    atomicAdd(&matched[nodeId], 1);
                }
                ++currentChar;
            }
        }
        __syncthreads();
    }
}

__global__ void
matchWordsSharedMem2(unsigned char *str, unsigned int *matched, trieOptimized *root, unsigned int size,
                     unsigned int sharedMemPerBlock) {
    const trieOptimized *node;
    int charIndex;
    unsigned int nextIndex, currentChar;
    unsigned int threadChunk = sharedMemPerBlock / blockDim.x;

    unsigned int startThread = threadChunk * threadIdx.x;
    unsigned int stopThread = startThread + threadChunk;

    extern __shared__ unsigned char sArray[];

    auto *uint4Str = reinterpret_cast < uint4 * > ( str );
    uint4 uint4Var;
    uchar4 c0, c4, c8, c12;

    for (unsigned int globalIndex = blockIdx.x * sharedMemPerBlock;
         globalIndex < size; globalIndex += gridDim.x * sharedMemPerBlock) {

        for (unsigned int tSharedIndex = threadIdx.x, tGlobalIndex = globalIndex / sizeof(uint4) + threadIdx.x;
             (tSharedIndex < sharedMemPerBlock / sizeof(uint4)) && tGlobalIndex < size / sizeof(uint4) + 1;
             tSharedIndex += blockDim.x, tGlobalIndex += blockDim.x) {
            uint4Var = uint4Str[tGlobalIndex];
            c0 = *reinterpret_cast<uchar4 *> ( &uint4Var.x );
            c4 = *reinterpret_cast<uchar4 *> ( &uint4Var.y );
            c8 = *reinterpret_cast<uchar4 *> ( &uint4Var.z );
            c12 = *reinterpret_cast<uchar4 *> ( &uint4Var.w );

            sArray[tSharedIndex * sizeof(uint4) + 0] = c0.x;
            sArray[tSharedIndex * sizeof(uint4) + 1] = c0.y;
            sArray[tSharedIndex * sizeof(uint4) + 2] = c0.z;
            sArray[tSharedIndex * sizeof(uint4) + 3] = c0.w;

            sArray[tSharedIndex * sizeof(uint4) + 4] = c4.x;
            sArray[tSharedIndex * sizeof(uint4) + 5] = c4.y;
            sArray[tSharedIndex * sizeof(uint4) + 6] = c4.z;
            sArray[tSharedIndex * sizeof(uint4) + 7] = c4.w;

            sArray[tSharedIndex * sizeof(uint4) + 8] = c8.x;
            sArray[tSharedIndex * sizeof(uint4) + 9] = c8.y;
            sArray[tSharedIndex * sizeof(uint4) + 10] = c8.z;
            sArray[tSharedIndex * sizeof(uint4) + 11] = c8.w;

            sArray[tSharedIndex * sizeof(uint4) + 12] = c12.x;
            sArray[tSharedIndex * sizeof(uint4) + 13] = c12.y;
            sArray[tSharedIndex * sizeof(uint4) + 14] = c12.z;
            sArray[tSharedIndex * sizeof(uint4) + 15] = c12.w;
        }

        // adding M - 1 chars (for word per-block overlap)
        if (threadIdx.x < M - 1)
            sArray[sharedMemPerBlock + threadIdx.x] = str[globalIndex + sharedMemPerBlock + threadIdx.x];

        __syncthreads();


        for (unsigned int startChar = startThread; (startChar < stopThread &&
                                                    globalIndex + startChar < size); ++startChar) {
            node = root;
            currentChar = startChar;
            while ((charIndex = index_of(sArray[currentChar])) != -1 && (nextIndex = node->next[charIndex]) != 0) {
                node = root + nextIndex;
                int nodeId = node->id;
                if (nodeId != -1) {
                    atomicAdd(&matched[nodeId], 1);
                }
                ++currentChar;
            }
        }
        __syncthreads();
    }
}

__global__ void
matchWordsSharedMem3(unsigned char *str, unsigned int *matched, trieOptimized *root, unsigned int size,
                    unsigned int sharedMemPerBlock) {
    const trieOptimized *node;
    int charIndex;
    unsigned int nextIndex, currentChar;
    unsigned int threadChunk = sharedMemPerBlock / blockDim.x;

    unsigned int startThread = threadChunk * threadIdx.x;
    unsigned int stopThread = startThread + threadChunk;

    extern __shared__ unsigned char sArray[];

    for (unsigned int globalIndex = blockIdx.x * sharedMemPerBlock, globalIndexCopy =
            blockIdx.x * sharedMemPerBlock / sizeof(uint4);
         globalIndex < size; globalIndex += (gridDim.x * sharedMemPerBlock), globalIndexCopy +=
                                                                                     gridDim.x * sharedMemPerBlock /
                                                                                     sizeof(uint4)) {

        ((uint4 *) sArray)[threadIdx.x] = ((uint4 *) str)[threadIdx.x + globalIndexCopy];

        if (threadIdx.x == 0)
            ((uint4 *) sArray)[sharedMemPerBlock / sizeof(uint4) + threadIdx.x] = ((uint4 *) str)[threadIdx.x +
                                                                                                  globalIndexCopy +
                                                                                                  sharedMemPerBlock /
                                                                                                  sizeof(uint4)];

        __syncthreads();

        for (unsigned int startChar = startThread; (startChar < stopThread &&
                                                    globalIndex + startChar < size); ++startChar) {
            node = root;
            currentChar = startChar;
            while ((charIndex = index_of(sArray[currentChar])) != -1 &&
                   (nextIndex = node->next[charIndex]) != 0) {
                node = root + nextIndex;
                int nodeId = node->id;
                if (nodeId != -1) {
                    atomicAdd(&matched[nodeId], 1);
                }
                ++currentChar;
            }
        }
        __syncthreads();
    }
}

bool
validateResult(const char *csvPath, std::unordered_map<std::string, int> &patternIdMap, const unsigned int *matches) {
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