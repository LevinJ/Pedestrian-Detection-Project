#ifndef DOPPIA_INTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
#define DOPPIA_INTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP

#include "IntegralChannelsDetector.hpp"
#include "BaseIntegralChannelsModelsBundleDetector.hpp"

namespace doppia {


/// This is the generalization of GpuMultiscalesIntegralChannelsDetector,
/// that also handles occlusion levels
/// @see MultiscalesIntegralChannelsDetector
class IntegralChannelsModelsBundleDetector:
        public IntegralChannelsDetector, public BaseIntegralChannelsModelsBundleDetector

{
public:

    IntegralChannelsModelsBundleDetector(const boost::program_options::variables_map &options,
                                           boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p,
                                           boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
                                           const float score_threshold, const int additional_border);
    ~IntegralChannelsModelsBundleDetector();
};

} // end of namespace doppia


#endif // DOPPIA_INTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
