#ifndef DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSETSTAGE_HPP
#define DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSETSTAGE_HPP

#include "SoftCascadeOverIntegralChannelsStumpStage.hpp"

#include <boost/array.hpp>
#include <cmath>

namespace doppia {

// mpl (like in boost::mpl) meta-programming language
namespace mpl {

// based on http://stackoverflow.com/questions/2060735/boost-mpl-and-type-list-generation
template <size_t N, size_t Power>
struct pow
{
    static const size_t value = N * pow<N, Power - 1>::value;
};

template <size_t N>
struct pow<N, 0>
{
    static const size_t value = 1;
};


template <size_t N>
struct two_pow_N
{
    static const size_t value = pow<2, N>::value;
};

} // end mpl namespace


/// copy paste from boost::array
template<class T, std::size_t N>
class array {
  public:
    T elems[N];    // fixed-size array of elements of type T

  public:
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator begin() { return elems; }
    const_iterator begin() const { return elems; }
    iterator end() { return elems+N; }
    const_iterator end() const { return elems+N; }

    // reverse iterator support
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && !defined(BOOST_MSVC_STD_ITERATOR) && !defined(BOOST_NO_STD_ITERATOR_TRAITS)
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#elif defined(_MSC_VER) && (_MSC_VER == 1300) && defined(BOOST_DINKUMWARE_STDLIB) && (BOOST_DINKUMWARE_STDLIB == 310)
    // workaround for broken reverse_iterator in VC7
    typedef std::reverse_iterator<std::_Ptrit<value_type, difference_type, iterator,
                                  reference, iterator, reference> > reverse_iterator;
    typedef std::reverse_iterator<std::_Ptrit<value_type, difference_type, const_iterator,
                                  const_reference, iterator, reference> > const_reverse_iterator;
#else
    // workaround for broken reverse_iterator implementations
    typedef std::reverse_iterator<iterator,T> reverse_iterator;
    typedef std::reverse_iterator<const_iterator,T> const_reverse_iterator;
#endif

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    /// operator[]
    __host__ __device__
    reference operator[](size_type i)
    {
        BOOST_ASSERT( i < N && "out of range" );
        return elems[i];
    }

    __host__ __device__
    const_reference operator[](size_type i) const
    {
        BOOST_ASSERT( i < N && "out of range" );
        return elems[i];
    }

    // at() with range check
    reference at(size_type i) { rangecheck(i); return elems[i]; }
    const_reference at(size_type i) const { rangecheck(i); return elems[i]; }

    // front() and back()
    reference front()
    {
        return elems[0];
    }

    const_reference front() const
    {
        return elems[0];
    }

    reference back()
    {
        return elems[N-1];
    }

    const_reference back() const
    {
        return elems[N-1];
    }

    // size is constant
    static size_type size() { return N; }
    static bool empty() { return false; }
    static size_type max_size() { return N; }
    enum { static_size = N };

    // swap (note: linear complexity)
    void swap (array<T,N>& y) {
        for (size_type i = 0; i < N; ++i)
            boost::swap(elems[i],y.elems[i]);
    }

    // direct access to data (read-only)
    const T* data() const { return elems; }
    T* data() { return elems; }

    // use array as C array (direct read/write access to data)
    T* c_array() { return elems; }

    // assignment with type conversion
    template <typename T2>
    array<T,N>& operator= (const array<T2,N>& rhs) {
        std::copy(rhs.begin(),rhs.end(), begin());
        return *this;
    }

    // assign one value to all elements
    void assign (const T& value)
    {
        std::fill_n(begin(),size(),value);
    }

    // check range (may be private because it is static)
    static void rangecheck (size_type i) {
        if (i >= size()) {
            throw std::out_of_range("array<>: index out of range");
        }
    }

}; // end of class array


template<std::size_t N>
class SoftCascadeOverIntegralChannelsStumpSetStage
{
public:

    typedef SimpleDecisionStump stump_t;
    typedef stump_t::feature_t::rectangle_t rectangle_t;

    array<stump_t, N> stumps;
    array<float, mpl::two_pow_N<N>::value> weights;

    /// if (strong_classifier_score < cascade_threshold) answer is "not this class"
    float cascade_threshold;

    /// Bounding box for the features included in all the nodes,
    /// this box is used to do fast image border condition checking
    rectangle_t bounding_box;

    /// compute the bounding box based on the current nodes values
    void compute_bounding_box();

    /// obtain the corresponding weight
    float operator()(const boost::array<float, N> &features_values) const;

    /// helper function when doing partial objects detection (borders of the image)
    const rectangle_t &get_bounding_box() const;

};


template <std::size_t N>
inline float
SoftCascadeOverIntegralChannelsStumpSetStage<N>::operator()(const boost::array<float, N> &features_values) const
{
    size_t index = 0;

    // uses >= to be consistent with Markus Mathias code
    //return (feature_value >= feature_threshold)? weight_true_leaf : weight_false_leaf;

// #pragma unroll does not exist in GCC, but we compile with -funroll-loops
    for(size_t i=0, two_pow_i=1; i < N; i+=1, two_pow_i*=2)
    {
        //index += (features_values[i] >= stumps[i].feature_threshold)? mpl::pow2<i>::value : 0;
        // smart compiler unleash your magic !
        index += (features_values[i] >= stumps[i].feature_threshold)? two_pow_i : 0;
    }

    return weights[index];
}


template <std::size_t N>
void
SoftCascadeOverIntegralChannelsStumpSetStage<N>::compute_bounding_box()
{
    bounding_box = stumps[0].feature.box;

    rectangle_t &bb_a = bounding_box;

    for(size_t i=1; i < stumps.size(); i+=1)
    {
        const rectangle_t &bb_b = stumps[i].feature.box;

        bb_a.min_corner().x( std::min(bb_a.min_corner().x(), bb_b.min_corner().x()) );
        bb_a.min_corner().y( std::min(bb_a.min_corner().y(), bb_b.min_corner().y()) );
        bb_a.max_corner().x( std::max(bb_a.max_corner().x(), bb_b.max_corner().x()) );
        bb_a.max_corner().y( std::max(bb_a.max_corner().y(), bb_b.max_corner().y()) );

    } // end of "for each other rectangle"

    return;
}


template <std::size_t N>
inline
const typename SoftCascadeOverIntegralChannelsStumpSetStage<N>::rectangle_t &
SoftCascadeOverIntegralChannelsStumpSetStage<N>::get_bounding_box() const
{
    return bounding_box;
}


typedef SoftCascadeOverIntegralChannelsStumpSetStage<2> SoftCascadeOverIntegralChannelsTwoStumpsStage;
typedef SoftCascadeOverIntegralChannelsStumpSetStage<3> SoftCascadeOverIntegralChannelsThreeStumpsStage;
typedef SoftCascadeOverIntegralChannelsStumpSetStage<4> SoftCascadeOverIntegralChannelsFourStumpsStage;


} // namespace doppia

#endif // DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSETSTAGE_HPP
