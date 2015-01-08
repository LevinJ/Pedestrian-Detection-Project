#ifndef DOPPIA_GPUINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
#define DOPPIA_GPUINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP

#include "GpuIntegralChannelsDetector.hpp"
#include "BaseIntegralChannelsModelsBundleDetector.hpp"

namespace doppia {


/// This is the generalization of GpuMultiscalesIntegralChannelsDetector,
/// that also handles occlusion levels
/// @see GpuMultiscalesIntegralChannelsDetector
class GpuIntegralChannelsModelsBundleDetector:
        public GpuIntegralChannelsDetector, public BaseIntegralChannelsModelsBundleDetector

{
public:

    GpuIntegralChannelsModelsBundleDetector(const boost::program_options::variables_map &options,
                                           boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p,
                                           boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
                                           const float score_threshold, const int additional_border);
    ~GpuIntegralChannelsModelsBundleDetector();
};

} // end of namespace doppia

#endif // DOPPIA_GPUINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
