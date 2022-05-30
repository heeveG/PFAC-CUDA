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
//    std::string input;
    std::vector<std::string> files{
            "../data/2600-0.txt", "../data/2701-0.txt", "../data/35-0.txt", "../data/84-0.txt", "../data/8800.txt",
            "../data/pg1727.txt", "../data/pg55.txt", "../data/pg6130.txt", "../data/pg996.txt", "../data/1342-0.txt"
    };
    read_data(input, files);

    // build trie
    std::unordered_map<std::string, int> patternIdMap;
    std::vector<trieOptimized, managed_allocator<trieOptimized>> nodes(1 << 17);

    do_trie(nodes, input, patternIdMap);

    int numPatterns = patternIdMap.size();
    auto root = nodes.data();

    const int blockSize = 256;
    const int numBlocks = 1024;
//    const int numBlocks = ((int) input.size() + blockSize - 1) / blockSize;

//    cudaProfilerStart();
    int *matches = (int *) malloc(numPatterns * sizeof(int));
    int *d_matched;
//    char *d_input;
//
//    check(cudaMalloc(&d_input, input.size()));
    check(cudaMalloc(&d_matched, numPatterns * sizeof(int)));
//    check(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));

    check(cudaMemPrefetchAsync(nodes.data(), 80564 * sizeof(trieOptimized), 0));
    check(cudaMemPrefetchAsync(input.data(), input.size(), 0));

    // perform matching
    float matchingTime = cudaEventProfile([&]() {
        matchWords<<<numBlocks, blockSize>>>(input.data(), d_matched, root, input.size());
    });

    check(cudaMemcpy(matches, d_matched, numPatterns * sizeof(int), cudaMemcpyDeviceToHost));
//    cudaProfilerStop();
    // validate results
    if (validateResult("../validation/results.csv", patternIdMap, matches))
        std::cout << "Matching completed successfully in: " << matchingTime << " ms. " << std::endl;
    else
        std::cout << "Invalid results" << std::endl;

//    do_trie(input, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);

    free(matches);
    cudaFree(d_matched);
//    cudaFree(d_input);
    return 0;
}
