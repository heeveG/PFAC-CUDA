#include <iostream>
#include "headers/util.cuh"

int main() {

    std::basic_string<char, std::char_traits<char>, managed_allocator<char>> input;

    std::vector<std::string> files{
            "../data/2600-0.txt", "../data/2701-0.txt", "../data/35-0.txt", "../data/84-0.txt", "../data/8800.txt",
            "../data/pg1727.txt", "../data/pg55.txt", "../data/pg6130.txt", "../data/pg996.txt", "../data/1342-0.txt"
    };

    read_data(input, files);

    do_trie(input, 1, 1);
    do_trie(input, 1, 1);
    do_trie(input, 1, std::thread::hardware_concurrency());
    do_trie(input, 1, std::thread::hardware_concurrency());

    assert(cudaSuccess == cudaSetDevice(0));
    cudaDeviceProp deviceProp{};
    assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0));

//    do_trie(input, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);
//    do_trie(input, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);

    return 0;
}
