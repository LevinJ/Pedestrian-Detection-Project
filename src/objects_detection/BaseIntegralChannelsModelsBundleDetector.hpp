#ifndef DOPPIA_BASEINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
#define DOPPIA_BASEINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP

#include "BaseMultiscalesIntegralChannelsDetector.hpp"

namespace doppia {


/// Extended version of BaseMultiscalesIntegralChannelsDetector that not only handles different scales,
/// but also different occlusion levels
class BaseIntegralChannelsModelsBundleDetector: public BaseMultiscalesIntegralChannelsDetector
{

protected:
    /// the constructor is protected because this base class is should not be instanciated directly
    BaseIntegralChannelsModelsBundleDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p);
    ~BaseIntegralChannelsModelsBundleDetector();


protected:

    /// updates the values inside detection_cascade_per_scale
    /// this variant will also update search_ranges,
    /// (since we will be shifting the actual scales)
    /// and add aditional search ranges for the different occlusion types
    void compute_scaled_detection_cascades();
};

} // namespace doppia

#endif // DOPPIA_BASEINTEGRALCHANNELSMODELSBUNDLEDETECTOR_HPP
