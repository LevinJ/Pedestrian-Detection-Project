#ifndef DOPPIA_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
#define DOPPIA_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP

#include "GpuIntegralChannelsDetector.hpp"
#include "BaseFastestPedestrianDetectorInTheWest.hpp"

namespace doppia {

/// This is is the GPU version of FastestPedestrianDetectorInTheWestV2,
/// this version exploits fractional channel features (using the GPU fast texture interpolation features)
/// @see FastestPedestrianDetectorInTheWestV2
class GpuFastestPedestrianDetectorInTheWest:
                public GpuIntegralChannelsDetector, public BaseFastestPedestrianDetectorInTheWest

{
public:

    GpuFastestPedestrianDetectorInTheWest(
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~GpuFastestPedestrianDetectorInTheWest();

protected:

    /// computes the detections directly on GPU, avoiding the score image transfer
    void compute_detections_at_specific_scale_v1(const size_t search_range_index,
                                                 const bool first_call = false);
};

} // end of namespace doppia

#endif // DOPPIA_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
