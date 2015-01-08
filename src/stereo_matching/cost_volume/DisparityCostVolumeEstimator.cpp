#include "DisparityCostVolumeEstimator.hpp"

#include "DisparityCostVolume.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/fill_multi_array.hpp"

#include <boost/gil/image_view_factory.hpp>

namespace doppia
{

using namespace boost;
using namespace boost::gil;
using namespace std;

typedef DisparityCostVolume::range_t range_t;
typedef DisparityCostVolume::const_data_2d_view_t const_data_2d_view_t;
typedef DisparityCostVolume::const_data_1d_view_t const_data_1d_view_t;
typedef DisparityCostVolume::const_data_2d_subarray_t const_data_2d_subarray_t;
typedef DisparityCostVolume::const_data_1d_subarray_t const_data_1d_subarray_t;
typedef DisparityCostVolume::data_3d_view_t data_3d_view_t;


program_options::options_description  DisparityCostVolumeEstimator::get_args_options()
{
    program_options::options_description desc("DisparityCostVolumeEstimator options");

    const bool simple_block_matcher_options_are_included = true;

    if(not simple_block_matcher_options_are_included)
    {
        desc.add_options()

                ("pixels_matching",
                 program_options::value<string>()->default_value("sad"),
                 "pixels matching method: sad, ssd, census or gradient")

                ("threshold",
                 program_options::value<float>()->default_value(0.5),
                 "minimum percent of pixels required to declare a match value between [0,1]")

                ;
    }

    return desc;
}

DisparityCostVolumeEstimator::DisparityCostVolumeEstimator()
    : AbstractDisparityCostVolumeEstimator()
{
    // this constructor should only be used for unit testing
    return;
}

DisparityCostVolumeEstimator::DisparityCostVolumeEstimator(const program_options::variables_map &options)
    : AbstractDisparityCostVolumeEstimator(options)
{

    threshold_percent = get_option_value<float>(options,"threshold");

    assert(threshold_percent > 0.0f);
    assert(threshold_percent <= 1.0f);

    return;
}

DisparityCostVolumeEstimator::~DisparityCostVolumeEstimator()
{
    // nothing to do here
    return;
}


void  DisparityCostVolumeEstimator::compute(gil::gray8c_view_t &left,
                                            gil::gray8c_view_t &right,
                                            DisparityCostVolume &cost_volume)
{
    compute_impl(left, right, cost_volume);
    return;
}

void  DisparityCostVolumeEstimator::compute(gil::rgb8c_view_t  &left,
                                            gil::rgb8c_view_t &right,
                                            DisparityCostVolume &cost_volume)
{
    compute_impl(left, right, cost_volume);
    return;
}


template <typename ImgView>
void DisparityCostVolumeEstimator::compute_impl( ImgView &left, ImgView &right, DisparityCostVolume &cost_volume)
{

    if (pixels_matching_method == "sad")
    {
        if(first_computation)
        {
            printf("DisparityCostVolumeEstimator::compute_impl will use sad matching over %zi disparities\n\n",
                   max_disparity);
        }
        SadCostFunctionT<uint8_t> pixels_distance;
        compute_costs_impl(left, right, pixels_distance, cost_volume);
    }
    else if (pixels_matching_method == "ssd")
    {
        if(first_computation)
        {
            printf("DisparityCostVolumeEstimator::compute_impl will use ssd matching over %zi disparities\n\n",
                   max_disparity);
        }
        SsdCostFunction pixels_distance;
        compute_costs_impl(left, right, pixels_distance, cost_volume);
    }
    else
    {
        printf("DisparityCostVolumeEstimator::compute_impl received an unknow pixels_matching_method value == %s\n",
               pixels_matching_method.c_str());

        throw std::invalid_argument("DisparityCostVolumeEstimator::compute_impl received an unknow pixels_matching_method");
    }

    first_computation = false;

    return;
}




template <typename ImgView, typename PixelsCostType>
inline void compute_costs_for_disparity(const ImgView &left, const ImgView &right,
                                        PixelsCostType &pixels_distance,
                                        const int disparity,
                                        DisparityCostVolume::data_2d_view_t &disparity_cost_view)
{
    typedef typename ImgView::value_type pixel_t;
    typedef  DisparityCostVolume::data_1d_subarray_t data_1d_subarray_t;

    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;

    for(int y=0; y < left.height(); y+=1)
    {
        typename ImgView::x_iterator left_row_it = left.x_at(disparity, y);
        typename ImgView::x_iterator right_row_it = right.row_begin(y);
        data_1d_subarray_t cost_row_subarray = disparity_cost_view[y];
        data_1d_subarray_t::iterator cost_it = cost_row_subarray.begin() + disparity;
        for(int left_x=disparity; left_x < left.width(); left_x+=1, ++left_row_it, ++right_row_it, ++cost_it)
        {
            const DisparityCostVolume::cost_t pixel_cost = pixels_distance(*left_row_it, *right_row_it);
            *cost_it = pixel_cost;
        } // end of 'for each row'
    } // end of 'for each column'

    return;
}



template <typename ImgView, typename PixelsCostType>
void DisparityCostVolumeEstimator::compute_costs_impl(const ImgView &left, const ImgView &right,
                                                      PixelsCostType &pixels_distance,
                                                      DisparityCostVolume &cost_volume)
{

    typedef typename ImgView::value_type pixel_t;
    maximum_cost_per_pixel = pixels_distance.template get_maximum_cost_per_pixel<pixel_t>();

    // lazy initialization
    this->resize_cost_volume(left.dimensions(), cost_volume);

    const bool crop_borders = false;

    if(crop_borders)
    {
        // FIXME hardcoded values
        const int top_bottom_margin = 25; // 20 // 10 // pixels
        const int left_right_margin = 40; //left.width()*0.1;  // *0.05 // pixels

        const int
                sub_width = left.width() - 2*left_right_margin,
                sub_height = left.height() - 2*top_bottom_margin;
        const ImgView
                left_subview = boost::gil::subimage_view(left,
                                                         left_right_margin, top_bottom_margin,
                                                         sub_width, sub_height),
                right_subview = boost::gil::subimage_view(right,
                                                          left_right_margin,top_bottom_margin,
                                                          sub_width, sub_height);


        const int y_min = top_bottom_margin, y_max = left.height() - top_bottom_margin;
        const int x_min = left_right_margin, x_max = left.width() - left_right_margin;
        data_3d_view_t data = cost_volume.get_costs_views();
        data_3d_view_t
                data_subview = data[ boost::indices[range_t(y_min, y_max)][range_t(x_min, x_max)][range_t()] ];

        // fill the original volumn with zero
        fill(cost_volume.get_costs_views(), 0);

#pragma omp parallel for
        for(size_t disparity=0; disparity < max_disparity; disparity +=1)
        {
            DisparityCostVolume::data_2d_view_t disparity_slice = data_subview[ boost::indices[range_t()][range_t()][disparity] ];
            compute_costs_for_disparity(left_subview, right_subview, pixels_distance, disparity, disparity_slice);
        }

    }
    else
    {

        // for each pixel and each disparity value
#pragma omp parallel for
        for(size_t disparity=0; disparity < max_disparity; disparity +=1)
        {
            data_3d_view_t data = cost_volume.get_costs_views();
            DisparityCostVolume::data_2d_view_t disparity_slice = data[ boost::indices[range_t()][range_t()][disparity] ];
            compute_costs_for_disparity(left, right, pixels_distance, disparity, disparity_slice);
        }

    } // end of "if else crop_borders"

    return;
}



} // end of namespace doppia

