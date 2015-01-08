#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP

#include <boost/version.hpp>

// Workaround for bug related to negative strides, see
// http://lists.boost.org/Archives/boost/2009/06/152833.php
// and SimpleTreesOptimizationStereo backwards_pass
#if BOOST_VERSION < 104100
#define BOOST_DISABLE_ASSERTS
#include <boost/multi_array.hpp>
#undef BOOST_DISABLE_ASSERTS
#else
#include <boost/multi_array.hpp>
#endif

#include <boost/cstdint.hpp>

#include <Eigen/Core>

namespace doppia
{

/// data is organized as y (rows), x (columns), disparity
/// CostType is expected to be float, boost::int32_t, boost::int16_t, or something else
template<typename CostType>
class CostVolume
{

public:

    typedef CostType cost_t;
    // data is organized as y (rows), x (columns), disparity
    typedef boost::multi_array<CostType, 3> data_t;

    typedef boost::multi_array_types::index_range range_t;
    typedef typename data_t::index index_t;

    // 3d --
    typedef typename data_t::template array_view<3>::type data_3d_view_t;
    typedef typename data_t::template const_array_view<3>::type const_data_3d_view_t;

    // 2d --
    typedef typename data_3d_view_t::reference data_2d_subarray_t;
    typedef typename data_3d_view_t::const_reference const_data_2d_subarray_t;

    typedef typename data_t::template array_view<2>::type data_2d_view_t;
    typedef typename data_t::template const_array_view<2>::type const_data_2d_view_t;

    // 1d --
    typedef typename data_2d_view_t::reference data_1d_subarray_t;
    typedef typename data_2d_view_t::const_reference const_data_1d_subarray_t;

    typedef typename data_t::template array_view<1>::type data_1d_view_t;
    typedef typename data_t::template const_array_view<1>::type const_data_1d_view_t;


    CostVolume(const int rows, const int columns, const int disparities);
    ~CostVolume();

    /// do memory allocation
    template<typename AnyCostType>
    void resize(const CostVolume<AnyCostType> &volume);

    void resize(const int rows, const int cols, const int disparities);
    //data_t &get_data();

    std::size_t rows() const;
    std::size_t columns() const;
    std::size_t disparities() const;
    const data_3d_view_t get_costs_views();
    const const_data_3d_view_t get_costs_views() const;

    /// returns a columns-disparities slice
    data_2d_subarray_t columns_disparities_slice(const int row_index);
    const_data_2d_subarray_t columns_disparities_slice(const int row_index) const;

    /// returns a rows-disparities slice
    /// @warning accessing data this way is slow, very slow (~10x)
    /// @note use columns_disparities_slice instead, much faster
    data_2d_view_t rows_disparities_slice(const int col_index);
    const_data_2d_view_t rows_disparities_slice(const int col_index) const;

    /// returns a columns-disparities slice
    /// @warning accessing data this way is slow, very slow (~10x)
    /// @note use columns_disparities_slice instead, much faster
    data_2d_view_t rows_columns_slice(const int disparity);
    const_data_2d_view_t rows_columns_slice(const int disparity) const;

protected:
    data_t data;

};


// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
// templated methods should be defined in the header


template<typename CostType>
CostVolume<CostType>::CostVolume(const int rows, const int cols, const int disparities)
{
    resize(rows, cols, disparities);
    return;
}

template<typename CostType>
CostVolume<CostType>::~CostVolume()
{
    // nothing to do here
    return;
}

template<typename CostType>
template<typename AnyCostType>
void CostVolume<CostType>::resize(const CostVolume<AnyCostType> &volume)
{
    // lazy allocation
    if((volume.rows() != rows()) or (volume.columns() != columns()) or (volume.disparities() != disparities()))
    {
        data.resize(boost::extents[volume.rows()][volume.columns()][volume.disparities()]);
    }
    return;
}


template<typename CostType>
void CostVolume<CostType>::resize(const int rows, const int cols, const int disparities)
{
    data.resize(boost::extents[rows][cols][disparities]);
    return;
}


template<typename CostType>
size_t CostVolume<CostType>::rows() const
{
    return data.shape()[0];
}

template<typename CostType>
size_t CostVolume<CostType>::columns() const
{
    return data.shape()[1];
}

template<typename CostType>
size_t CostVolume<CostType>::disparities() const
{
    return data.shape()[2];
}

template<typename CostType>
const typename CostVolume<CostType>::data_3d_view_t CostVolume<CostType>::get_costs_views()
{
    return data[ boost::indices[range_t()][range_t()][range_t()] ];
}

template<typename CostType>
const typename CostVolume<CostType>::const_data_3d_view_t CostVolume<CostType>::get_costs_views() const
{
    return data[ boost::indices[range_t()][range_t()][range_t()] ];
}



/// returns a columns-disparities slice
template<typename CostType>
typename CostVolume<CostType>::data_2d_subarray_t
CostVolume<CostType>::columns_disparities_slice(const int row_index)
{
    return data[row_index];
}

/// returns a columns-disparities slice
template<typename CostType>
typename CostVolume<CostType>::const_data_2d_subarray_t
CostVolume<CostType>::columns_disparities_slice(const int row_index) const
{
    return data[row_index];
}


/// returns a rows-disparities slice
/// @warning accessing data this way is slow, very slow (~10x)
/// @note use columns_disparities_slice instead, much faster
template<typename CostType>
typename CostVolume<CostType>::data_2d_view_t
CostVolume<CostType>::rows_disparities_slice(const int col_index)
{
    return data[ boost::indices[range_t()][col_index][range_t()] ];
}


/// returns a rows-disparities slice
/// @warning accessing data this way is slow, very slow (~10x)
/// @note use columns_disparities_slice instead, much faster
template<typename CostType>
typename CostVolume<CostType>::const_data_2d_view_t
CostVolume<CostType>::rows_disparities_slice(const int col_index) const
{
    return data[ boost::indices[range_t()][col_index][range_t()] ];
}


/// returns a columns-disparities slice
/// @warning accessing data this way is slow, very slow (~10x)
/// @note use columns_disparities_slice instead, much faster
template<typename CostType>
typename CostVolume<CostType>::data_2d_view_t
CostVolume<CostType>::rows_columns_slice(const int disparity)
{
    return data[ boost::indices[range_t()][range_t()][disparity] ];
}

/// returns a columns-disparities slice
/// @warning accessing data this way is slow, very slow (~10x)
/// @note use columns_disparities_slice instead, much faster
template<typename CostType>
typename CostVolume<CostType>::const_data_2d_view_t
CostVolume<CostType>::rows_columns_slice(const int disparity) const
{
    return data[ boost::indices[range_t()][range_t()][disparity] ];
}


} // end of namespace doppia


#endif // COSTVOLUME_HPP
