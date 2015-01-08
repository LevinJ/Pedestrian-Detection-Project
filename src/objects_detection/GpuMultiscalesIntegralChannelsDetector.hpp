#ifndef DOPPIA_GPUMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
#define DOPPIA_GPUMULTISCALESINTEGRALCHANNELSDETECTOR_HPP

#include "GpuIntegralChannelsDetector.hpp"
#include "BaseMultiscalesIntegralChannelsDetector.hpp"


namespace doppia {

/// This is the GPU variant of MultiscalesIntegralChannelsDetector
/// @see MultiscalesIntegralChannelsDetector
class GpuMultiscalesIntegralChannelsDetector :
        public GpuIntegralChannelsDetector, public BaseMultiscalesIntegralChannelsDetector
{
public:
    GpuMultiscalesIntegralChannelsDetector(const boost::program_options::variables_map &options,
                                           boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
                                           boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
                                           const float score_threshold, const int additional_border);
    ~GpuMultiscalesIntegralChannelsDetector();

};

} // end of namespace doppia

#endif // DOPPIA_GPUMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
