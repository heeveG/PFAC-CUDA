//
// Created by heeve on 09.05.22.
//

#ifndef PFAC_MANAGED_ALLOCATOR_HPP
#define PFAC_MANAGED_ALLOCATOR_HPP

template <class T>
struct managed_allocator {
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T value_type;

    template< class U > struct rebind { typedef managed_allocator<U> other; };
    managed_allocator() = default;

    template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        void* out = nullptr;
        check(cudaMallocHost(&out, n*sizeof(T)));
        return static_cast<T*>(out);
    }
    void deallocate(T* p, std::size_t) noexcept {
        check(cudaFreeHost(p));
    }
};

template<class T, class... Args>
T* make_allocator(Args &&... args) {
    managed_allocator<T> ma;
    return new (ma.allocate(1)) T(std::forward<Args>(args)...);
}

#endif //PFAC_MANAGED_ALLOCATOR_HPP
