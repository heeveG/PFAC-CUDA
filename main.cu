#include <iostream>
#include "headers/util.cuh"
#include "cuda_profiler_api.h"

int main() {
    // setup the device
    check(cudaSetDevice(0));
    cudaDeviceProp deviceProp{};
    check(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "GPU used : " << deviceProp.name << " " << deviceProp.sharedMemPerMultiprocessor << " "
              << deviceProp.sharedMemPerBlock << std::endl;

    // read input data
    std::basic_string<unsigned char, std::char_traits<unsigned char>, managed_allocator<unsigned char>> input;
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

    const int sharedMemPerBlock = 16384;
    const int blockSize = 1024;
    const int numBlocks = 24;

    auto *matches = (unsigned int *) malloc(numPatterns * sizeof(unsigned int));
    unsigned int *d_matched;
    unsigned char *d_input;
    trieOptimized *d_trie;

    check(cudaMalloc(&d_matched, numPatterns * sizeof(int)));
    check(cudaMalloc(&d_input, input.size()));
    check(cudaMalloc(&d_trie, numNodes * sizeof(trieOptimized)));

    const unsigned char *inputPtr = input.data();

    float matchingTime = cudaEventProfile([&]() {
        // prefetch trie
        check(cudaMemcpy(d_trie, nodes.data(), numNodes * sizeof(trieOptimized), cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));

//        matchWordsSharedMem<<<numBlocks, blockSize, sharedMemPerBlock + M - 1>>>(
//                d_input, d_matched, d_trie, input.size(), sharedMemPerBlock);

        matchWordsSharedMem2<<<numBlocks, blockSize, sharedMemPerBlock + 16 * ((M - 1) / 16 + 1)>>>(
                d_input, d_matched, d_trie, input.size(), sharedMemPerBlock);
        check(cudaDeviceSynchronize());
        // copy results back to host
        check(cudaMemcpy(matches, d_matched, numPatterns * sizeof(int), cudaMemcpyDeviceToHost));
    });

    // validate results
    if (validateResult("../validation/results.csv", patternIdMap, matches))
        std::cout << "Matching completed successfully in: " << matchingTime << " ms." << std::endl;
    else
        std::cout << "Invalid results" << std::endl;

    free(matches);
    check(cudaFree(d_matched));
    check(cudaFree(d_input));
    check(cudaFree(d_trie));
    return 0;
}
