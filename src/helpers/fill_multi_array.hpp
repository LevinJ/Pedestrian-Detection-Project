#ifndef FILL_MULTI_ARRAY_HPP
#define FILL_MULTI_ARRAY_HPP

#include <algorithm>
#include <boost/multi_array.hpp>

template<std::size_t D>
struct fill_functor {
    template< class Array, class V >
    void operator()(Array x, const V &v) {

        fill_functor<D-1> f;
        for (typename Array::iterator i = x.begin(); i != x.end(); ++i) {
            typename Array::iterator::reference ri = *i;
            f(ri,v);
        }
    }
};

template<> struct fill_functor<1> {
    template< class Array, class V >
    void operator()(Array x, const V &v) {
        std::fill(x.begin(), x.end(), v);
    }
};

template<> struct fill_functor<0> { /* error on zero dimensional */ };

template<class Array, class V>
void fill(Array &x, const V &v) {
    fill_functor<Array::dimensionality>()(x,v);
}

template<class V1, std::size_t NumDims, class Allocator, class V2>
void fill(boost::multi_array<V1,NumDims,Allocator> &x, const V2 &v) {
        std::fill_n(x.data(), x.num_elements(), v);
}

template<class V1, std::size_t NumDims, class V2>
void fill(boost::multi_array_ref<V1,NumDims> &x, const V2 &v) {
        std::fill_n(x.data(), x.num_elements(), v);
}


#endif // FILL_MULTI_ARRAY_HPP
