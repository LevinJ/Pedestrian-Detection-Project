#ifndef DOPPIA_FASTESTPEDESTRIANDETECTORINTHEWESTV2_HPP
#define DOPPIA_FASTESTPEDESTRIANDETECTORINTHEWESTV2_HPP

#include "BaseFastestPedestrianDetectorInTheWest.hpp"
#include "IntegralChannelsDetector.hpp"

namespace doppia {

/// This is a new version of the FastestPedestrianDetectorInTheWest that shares more code with other detectors,
/// and follows the same convention as MultiscalesInegralChannelsDetector
/// @see FastestPedestrianDetectorInTheWest
class FastestPedestrianDetectorInTheWestV2:
        public IntegralChannelsDetector, public BaseFastestPedestrianDetectorInTheWest
{
public:
    FastestPedestrianDetectorInTheWestV2(
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~FastestPedestrianDetectorInTheWestV2();

protected:

    /// updates the values inside detection_cascade_per_scale
    /// this variant will also update search_ranges
    void compute_scaled_detection_cascades();

};


} //  end of namespace doppia

#endif // DOPPIA_FASTESTPEDESTRIANDETECTORINTHEWESTV2_HPP
