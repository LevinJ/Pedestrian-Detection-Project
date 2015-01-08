#ifndef BASEVERYFASTINTEGRALCHANNELSDETECTOR_HPP
#define BASEVERYFASTINTEGRALCHANNELSDETECTOR_HPP

#include "BaseMultiscalesIntegralChannelsDetector.hpp"
#include "MultiScalesIntegralChannelsModel.hpp"

#include <boost/program_options/variables_map.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {


class BaseVeryFastIntegralChannelsDetector: public BaseMultiscalesIntegralChannelsDetector
{
public:
    typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stage_t fast_fractional_stage_t;
    typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stages_t fast_fractional_stages_t;

    typedef fast_fractional_stage_t fractional_cascade_stage_t;
    typedef fast_fractional_stages_t fractional_cascade_stages_t;

protected:
    BaseVeryFastIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p);

    ~BaseVeryFastIntegralChannelsDetector();

protected:

    /// updates the values inside detection_cascade_per_scale
    void compute_scaled_detection_cascades();

    void compute_extra_data_per_scale(const size_t input_width, const size_t input_height);

public:

    bool should_shuffle_the_scales;

    /// Helper debug variable
    std::vector<size_t> detector_index_per_scale;

protected:
    /// obtain the 'final' search range including scale, occlusion, detector size and shrinking factor information
    DetectorSearchRange compute_scaled_search_range(const size_t scale_index) const;

    /// helper function that validates the internal consistency of the extra_data_per_scale
    void check_extra_data_per_scale();
};

/// Helper method used in both CPU and GPU versions
/// for each detection windows, it shift the detection window such as the upper-left corner becomes the center
void recenter_detections(AbstractObjectsDetector::detections_t &detections);


} // end of namespace doppia

#endif // BASEVERYFASTINTEGRALCHANNELSDETECTOR_HPP
