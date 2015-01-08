#ifndef DOPPIA_INTEGRALCHANNELSDETECTORMODELSBUNDLE_HPP
#define DOPPIA_INTEGRALCHANNELSDETECTORMODELSBUNDLE_HPP

#include "SoftCascadeOverIntegralChannelsModel.hpp"


// forward declaration
namespace doppia_protobuf {
class DetectorModelsBundle;
}

namespace doppia {

/// This is the application specific parallel of the protocol buffer message DetectorModelBundle
/// @see detector_model.proto
class IntegralChannelsDetectorModelsBundle
{

public:

    typedef SoftCascadeOverIntegralChannelsModel detector_t;
    typedef std::vector<detector_t> detectors_t;

protected:
    /// void constructor only for child classes
    IntegralChannelsDetectorModelsBundle();

public:
    /// this constructor will copy the protobuf data into a more efficient data structure
    IntegralChannelsDetectorModelsBundle(const doppia_protobuf::DetectorModelsBundle &model);
    virtual ~IntegralChannelsDetectorModelsBundle();

    const detectors_t& get_detectors() const;

    bool has_soft_cascade() const;

protected:

    detectors_t detectors;

    /// Check that the detectors are consistent amongst themselves
    virtual void sanity_check() const;
};

} // namespace doppia

#endif // DOPPIA_INTEGRALCHANNELSDETECTORMODELSBUNDLE_HPP
