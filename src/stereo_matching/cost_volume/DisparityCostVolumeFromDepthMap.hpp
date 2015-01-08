#ifndef DISPARITYCOSTVOLUMEFROMDEPTHMAP_HPP
#define DISPARITYCOSTVOLUMEFROMDEPTHMAP_HPP

#include "AbstractDisparityCostVolumeEstimator.hpp"
#include "stereo_matching/AbstractStereoMatcher.hpp"

#include <boost/shared_ptr.hpp>

namespace doppia {


/// Simple adaptor class that sets a cost volume from the output of a depth map
/// Cost is 0 at the values of the depth map and zero everywhere else
class DisparityCostVolumeFromDepthMap : public AbstractDisparityCostVolumeEstimator
{
public:
    DisparityCostVolumeFromDepthMap(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<AbstractStereoMatcher> &stereo_matcher_p);
    ~DisparityCostVolumeFromDepthMap();


    void compute(boost::gil::gray8c_view_t &left,
                         boost::gil::gray8c_view_t &right,
                         DisparityCostVolume &cost_volume);

    void compute(boost::gil::rgb8c_view_t  &left,
                         boost::gil::rgb8c_view_t &right,
                         DisparityCostVolume &cost_volume);

    typedef AbstractStereoMatcher::disparity_map_t::const_view_t  disparity_map_view_t;

    /// helper method for visualization
    disparity_map_view_t get_disparity_map();

protected:
    boost::shared_ptr<AbstractStereoMatcher> stereo_matcher_p;

};

} // end of namespace doppia

#endif // DISPARITYCOSTVOLUMEFROMDEPTHMAP_HPP
