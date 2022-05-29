//
// Created by heeve on 07.05.22.
//

#ifndef CMAKE_AND_CUDA_UTIL_TCCH
#define CMAKE_AND_CUDA_UTIL_TCCH

#include "managed_allocator.hpp"

template<class String>
void do_trie(std::vector<trie, managed_allocator<trie>> &nodes, String const &input,
             std::unordered_map<std::string, int> &patternIdMap) {
    // #TODO add turing support
    // if(use_simt) check(cudaMemset(nodes.data(), 0, nodes.size()*sizeof(trie)));

    auto t = nodes.data();
    auto b = make_allocator<trie *>(nodes.data() + 1);

    auto const begin = get_current_time_fenced();
#ifdef TURING
    device_make_trie<<<blocks,threads>>>(*t, b, input.data(), input.data() + input.size());
    check(cudaDeviceSynchronize());
#else
    host_make_trie(*t, *b, input.data(), input.data() + input.size(), patternIdMap);
    // $TODO - allow for parallel host construction
#endif
    auto const end = get_current_time_fenced();
    auto const time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto const count = *b - nodes.data();
    std::cout << "Assembled " << count << " nodes in " << time << "ms." << std::endl;

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

template<typename F>
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