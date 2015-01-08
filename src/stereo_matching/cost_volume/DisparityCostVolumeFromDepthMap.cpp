#include "DisparityCostVolumeFromDepthMap.hpp"
#include "DisparityCostVolume.hpp"


#include "helpers/fill_multi_array.hpp"

#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/dynamic_image/any_image_view.hpp>

namespace doppia {

DisparityCostVolumeFromDepthMap::DisparityCostVolumeFromDepthMap(
    const boost::program_options::variables_map &options,
    boost::shared_ptr<AbstractStereoMatcher> &stereo_matcher_p_)
    : AbstractDisparityCostVolumeEstimator(options),
      stereo_matcher_p(stereo_matcher_p_)
{
    // nothing to do here
    return;
}

DisparityCostVolumeFromDepthMap::~DisparityCostVolumeFromDepthMap()
{
    // nothing to do here
    return;
}



void DisparityCostVolumeFromDepthMap::compute(boost::gil::gray8c_view_t &left,
                                              boost::gil::gray8c_view_t &right,
                                              DisparityCostVolume &cost_volume)
{
    throw std::runtime_error("DisparityCostVolumeFromDepthMap::compute is not yet implemented for boost::gil::gray8c_view_t");
    return;
}

void DisparityCostVolumeFromDepthMap::compute(boost::gil::rgb8c_view_t  &left,
                                              boost::gil::rgb8c_view_t &right,
                                              DisparityCostVolume &cost_volume)
{
    assert(left.dimensions() == right.dimensions());

    this->resize_cost_volume(left.dimensions(), cost_volume);

    // compute the disparity map --
    AbstractStereoMatcher::disparity_map_t::const_view_t disparity_map;
    {
        boost::gil::any_image<AbstractStereoMatcher::input_images_t>::const_view_t
                left_view(left), right_view(right);
        stereo_matcher_p->set_rectified_images_pair(left_view, right_view);
        stereo_matcher_p->compute_disparity_map();
        disparity_map = stereo_matcher_p->get_disparity_map();
    }


    // set the cost volume --
    // we set the maximal cost on every pixel of the cost volume
    maximum_cost_per_pixel = 255;
    fill(cost_volume.get_costs_views(), maximum_cost_per_pixel);
    for(int y=0; y < disparity_map.height(); y+=1)
    {
        AbstractStereoMatcher::disparity_map_t::const_view_t::x_iterator row_it = disparity_map.row_begin(y);
        DisparityCostVolume::data_2d_subarray_t columns_disparities_slice = cost_volume.columns_disparities_slice(y);
        for(int x=0; x < disparity_map.width(); x+=1, ++row_it)
        {
            const size_t disparity_value = *row_it;
	    if(disparity_value < cost_volume.disparities())
	    {
              // we set zero cost to the depth map disparity values
              columns_disparities_slice[x][disparity_value] = 0;
	    }
	    else
	    {
		// disparity_value >= cost_volume.disparities()
		// we assume this value indicates "occluded pixel"
	    }
	   
        } // end of "for each col"
    } // end of "for each row"

    return;
}

DisparityCostVolumeFromDepthMap::disparity_map_view_t
DisparityCostVolumeFromDepthMap::get_disparity_map()
{
    return stereo_matcher_p->get_disparity_map();
}

} // end of namespace doppia
