#ifndef DOPPIA_ANGLEBINCOMPUTER_HPP
#define DOPPIA_ANGLEBINCOMPUTER_HPP

#include <boost/multi_array.hpp>
#include <cmath>
#include <cassert>

namespace doppia {

/// Helper class that computes the angle bin faster than using atan2
template<int num_bins>
class AngleBinComputer
{

public:
    AngleBinComputer();

    /// calling convention is the same of atan2
    int operator()(const float &y, const float &x) const;

    int get_num_bins() const;

    /// the returned value x is in the inclusive range [0, abs(x+y)]
    float soft_binning(const float &y, const float &x, const int bin_index) const;

public:
    // public for easy debugging, the object instances should be const
    boost::multi_array<float, 2> bin_vectors;
};

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
// The class is fully templated, so code should be in header

template <int num_bins>
AngleBinComputer<num_bins>::AngleBinComputer()
{

    bin_vectors.resize(boost::extents[num_bins][2]);

    const float angle_quantum = M_PI/(num_bins);

    float theta = 0; // theta is in the center of the angle bin
    for(int bin_index = 0; bin_index < num_bins; bin_index += 1, theta+= angle_quantum)
    {
        bin_vectors[bin_index][0] = std::cos(theta);
        bin_vectors[bin_index][1] = std::sin(theta);
    }

    return;
}


/// this method is speed sensistive
template <int num_bins>
inline
int AngleBinComputer<num_bins>::operator()(const float &y, const float &x) const
{
    // no need to explicitly handle the case y and x == 0,
    // this correspond to zero gradient areas, thus whatever the bin, the histograms will not be modified

    int index = 0;
    float max_dot_product = std::abs(x*bin_vectors[0][0] + y*bin_vectors[0][1]);

    // let us hope this gets unrolled
    for(int i = 1; i < num_bins; i += 1)
    {
        const float dot_product = std::abs(x*bin_vectors[i][0] + y*bin_vectors[i][1]);
        if(dot_product > max_dot_product)
        {
            max_dot_product = dot_product;
            index = i;
        }
    } // end of "for each bin"

    return index;
}


template <int num_bins>
int AngleBinComputer<num_bins>::get_num_bins() const
{
    return num_bins;
}

template <int num_bins>
float AngleBinComputer<num_bins>::soft_binning(const float &y, const float &x, const int bin_index) const
{
    assert(bin_index < num_bins);
    const float dot_product = std::abs(x*bin_vectors[bin_index][0] + y*bin_vectors[bin_index][1]);
    return dot_product;
}

} // end of namespace doppia

#endif // DOPPIA_ANGLEBINCOMPUTER_HPP
