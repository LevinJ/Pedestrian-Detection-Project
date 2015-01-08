#ifndef FOR_EACH_MULTI_ARRAY_HPP
#define FOR_EACH_MULTI_ARRAY_HPP

// code from http://agentzlerich.blogspot.com/2010/01/providing-fill-and-foreach-algorithms.html

#include <boost/static_assert.hpp>
#include <boost/multi_array.hpp>

namespace { // anonymous

template<std::size_t D>
struct for_each_functor {

    BOOST_STATIC_ASSERT(D != 0); // Nonsensical behavior for zero dimensions
    BOOST_STATIC_ASSERT(D > 1);  // Ensure not instantiated for specialized values

    // See http://groups.google.com/group/boost-list/browse_thread/thread/e16f32c4411dea08
    // for details about why MultiArray::iterator::reference is used below.
    template<class MultiArray,
             class UnaryFunction >
    void operator()(MultiArray &x, UnaryFunction &f) const {
        for_each_functor<D-1> functor;
        for (typename MultiArray::iterator i = x.begin(); i != x.end(); ++i) {
            typename MultiArray::iterator::reference ri = *i;
            functor(ri,f);
        }
    }
};

template<> struct for_each_functor<1> {
    template<class MultiArray,
             class UnaryFunction >
    void operator()(MultiArray &x, UnaryFunction &f) const {
        std::for_each(x.begin(), x.end(), f);
    }
};

} // namespace anonymous


/**
 * Invoke \c f for each element in MultiArray \c x.  The
 * order in which the invocations take place is undefined.
 *
 * @param x MultiArray over which to iterate.
 * @param f UnaryFunction to invoke on each element of \c x.
 *
 * @return \c f.
 *
 * @see SGI's <a href="http://www.sgi.com/tech/stl/UnaryFunction.html">
 *      UnaryFunction</a> concept for more information.
 */
template<class MultiArray,
         class UnaryFunction>
inline
UnaryFunction for_each(MultiArray &x,
                       UnaryFunction f) {
    for_each_functor<MultiArray::dimensionality>()(x,f);
    return f;
}


/**
 * Invoke \c f for each element in <tt>boost::multi_array</tt> \c x.  The order
 * in which the invocations take place is undefined.  This specialization
 * takes advantage of <tt>boost::multi_array</tt>'s contiguous storage.
 *
 * @param x <tt>boost::multi_array</tt> over which to iterate.
 * @param f UnaryFunction to invoke on each element of \c x.
 *
 * @return \c f.
 *
 * @see SGI's <a href="http://www.sgi.com/tech/stl/UnaryFunction.html">
 *      UnaryFunction</a> concept for more information.
 */
template<class ValueType, std::size_t NumDims, class Allocator,
         class UnaryFunction>
inline
UnaryFunction for_each(boost::multi_array<ValueType,NumDims,Allocator> &x,
                       UnaryFunction f) {
    return std::for_each(x.data(), x.data() + x.num_elements(), f);
}

/**
 * Invoke \c f for each element in <tt>boost::multi_array_ref</tt> \c x.  The
 * order in which the invocations take place is undefined.  This specialization
 * takes advantage of <tt>boost::multi_array_ref</tt>'s contiguous storage.
 *
 * @param x <tt>boost::multi_array_ref</tt> over which to iterate.
 * @param f UnaryFunction to invoke on each element of \c x.
 *
 * @return \c f.
 *
 * @see SGI's <a href="http://www.sgi.com/tech/stl/UnaryFunction.html">
 *      UnaryFunction</a> concept for more information.
 */
template<class ValueType, std::size_t NumDims,
         class UnaryFunction>
inline
UnaryFunction for_each(boost::multi_array_ref<ValueType,NumDims> &x,
                       UnaryFunction f)
{
    return std::for_each(x.data(), x.data() + x.num_elements(), f);
}

#endif // FOR_EACH_MULTI_ARRAY_HPP
