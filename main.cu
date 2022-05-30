#include <iostream>
#include "headers/util.cuh"
#include "cuda_profiler_api.h"

int main() {
    // setup the device
    check(cudaSetDevice(0));
    cudaDeviceProp deviceProp{};
    check(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "GPU used : " << deviceProp.name << std::endl;

    // read input data
    std::basic_string<char, std::char_traits<char>, managed_allocator<char>> input;
    std::vector<std::string> files{
            "../data/2600-0.txt", "../data/2701-0.txt", "../data/35-0.txt", "../data/84-0.txt", "../data/8800.txt",
            "../data/pg1727.txt", "../data/pg55.txt", "../data/pg6130.txt", "../data/pg996.txt", "../data/1342-0.txt"
    };
    read_data(input, files);

    // build trie
    std::unordered_map<std::string, int> patternIdMap;
    std::vector<trieOptimized, managed_allocator<trieOptimized>> nodes(1 << 17);

    int numNodes = do_trie(nodes, input, patternIdMap);

    // perform matching
    int numPatterns = patternIdMap.size();
    auto root = nodes.data();

    const int blockSize = 256;
    const int numBlocks = 1024;
    const int numStreams = 16;
    unsigned long streamBytes = input.size() / numStreams;
    cudaStream_t streams[numStreams];
    for (auto &stream : streams) check(cudaStreamCreate(&stream));

    int *matches = (int *) malloc(numPatterns * sizeof(int));
    int *d_matched;
    check(cudaMalloc(&d_matched, numPatterns * sizeof(int)));

    const char *inputPtr = input.data();

    float matchingTime = cudaEventProfile([&]() {
        // prefetch trie
        check(cudaMemPrefetchAsync(nodes.data(), numNodes * sizeof(trieOptimized), 0));

        // run staged copy-execute of input string and kernel
        for (int i = 0; i < numStreams; ++i) {
            unsigned long offset = i * streamBytes;
            unsigned long inputBytes = i == numStreams - 1 ? input.size() - offset : streamBytes;

            check(cudaMemPrefetchAsync(&inputPtr[offset], inputBytes, 0, streams[i]));
            matchWords<<<numBlocks / numStreams, blockSize, 0, streams[i]>>>(&inputPtr[offset], d_matched, root,
                                                                             inputBytes, input.size());
        }
        // copy results back to host
        check(cudaMemcpy(matches, d_matched, numPatterns * sizeof(int), cudaMemcpyDeviceToHost));
    });

    // validate results
    if (validateResult("../validation/results.csv", patternIdMap, matches))
        std::cout << "Matching completed successfully in: " << matchingTime << " ms. copying: " << std::endl;
    else
        std::cout << "Invalid results" << std::endl;

    free(matches);
    cudaFree(d_matched);

    return 0;
}
