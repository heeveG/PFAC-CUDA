//
// Created by heeve on 07.05.22.
//

#ifndef CMAKE_AND_CUDA_UTIL_TCCH
#define CMAKE_AND_CUDA_UTIL_TCCH

#include "managed_allocator.hpp"

template<class String>
int do_trie(std::vector<trie, managed_allocator<trie>> &nodes, String const &input,
            std::unordered_map<std::string, int> &patternIdMap, int blocks, int threads) {

    // $TODO - allow for parallel host construction
//    if(use_simt) check(cudaMemset(nodes.data(), 0, nodes.size()*sizeof(trie)));

    auto t = nodes.data();
    auto b = make_allocator<trie *>(nodes.data() + 1);

    auto const begin = std::chrono::steady_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
#ifdef TURING
    device_make_trie<<<blocks,threads>>>(*t, b, input.data(), input.data() + input.size());
    check(cudaDeviceSynchronize());
#else
    int numPatterns = host_make_trie(*t, *b, input.data(), input.data() + input.size(), patternIdMap);
//    assert(blocks == 1);
//    std::vector<std::thread> tv(threads);
//    for(auto count = threads; count; --count)
//        tv[count - 1] = std::thread([&, count]() {
//            make_trie(*t, *b, input.data(), input.data() + input.size(), count - 1, threads);
//        });
//    for(auto& t : tv)
//        t.join();
#endif
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto const end = std::chrono::steady_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto const count = *b - nodes.data();
    std::cout << "Assembled " << count << " nodes on " << blocks << "x" << threads << " " << " threads in " << time
              << "ms." << std::endl;

    return numPatterns;
}

template<class String>
void read_data(String &input, std::vector<std::string> const &files) {
    for (const auto &ptr : files) {
        auto const cur = input.size();
        std::ifstream in(ptr.data());
        in.seekg(0, std::ios_base::end);
        auto const pos = in.tellg();
        input.resize(cur + pos);
        in.seekg(0, std::ios_base::beg);
        in.read((char *) input.data() + cur, pos);
    }
}

template <typename F>
float cudaEventProfile(const F &func) {
    cudaEvent_t start, stop;

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, 0));
    func();
    check(cudaEventRecord(stop, 0));
    check(cudaEventSynchronize(stop));

    float time;
    check(cudaEventElapsedTime(&time, start, stop));

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    return time;
}

#endif //CMAKE_AND_CUDA_UTIL_TCCH