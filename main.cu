#include <iostream>
#include "headers/util.cuh"


int main() {
    std::basic_string<char, std::char_traits<char>, managed_allocator<char>> input;
    std::vector<std::string> files{
            "../data/2600-0.txt", "../data/2701-0.txt", "../data/35-0.txt", "../data/84-0.txt", "../data/8800.txt",
            "../data/pg1727.txt", "../data/pg55.txt", "../data/pg6130.txt", "../data/pg996.txt", "../data/1342-0.txt"
    };


    read_data(input, files);

    std::unordered_map<std::string, int> patternIdMap;
    std::vector<trie, managed_allocator<trie>> nodes(1 << 17);
    int numPatterns = do_trie(nodes, input, patternIdMap, 1, 1);


    assert(cudaSuccess == cudaSetDevice(0));
    cudaDeviceProp deviceProp{};
    assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0));

    auto root = nodes.data();

    const int blockSize = 256;
//    const int numBlocks = ((int) input.size() + blockSize - 1) / blockSize;
    const int numBlocks = 24;

    int *matches = (int *) malloc(numPatterns * sizeof(int));
    int *d_matched;
    check(cudaMalloc(&d_matched, numPatterns * sizeof(int)));

    float matchingTime = cudaEventProfile([&]() {
        matchWords<<<numBlocks, blockSize>>>(input.data(), d_matched, root, input.size());
    });
    check(cudaMemcpy(matches, d_matched, numPatterns * sizeof(int), cudaMemcpyDeviceToHost));

    if (validateResult("../validation/results.csv", patternIdMap, matches))
        std::cout << "Matching completed successfully in: " << matchingTime << " ms. " << std::endl;
    else
        std::cout << "Invalid results" << std::endl;

//    do_trie(input, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);

    return 0;
}
