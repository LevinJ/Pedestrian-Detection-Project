#ifndef DOPPIA_MULTISCALESINTEGRALCHANNELSMODEL_HPP
#define DOPPIA_MULTISCALESINTEGRALCHANNELSMODEL_HPP

#include "IntegralChannelsDetectorModelsBundle.hpp"

// forward declaration
namespace doppia_protobuf {
class MultiScalesDetectorModel;
class DetectorModelsBundle;
}

namespace doppia {

/// This is the application specific parallel of the protocol buffer message MultiScalesDetectorModel
/// @see detector_model.proto
class MultiScalesIntegralChannelsModel: public IntegralChannelsDetectorModelsBundle
{
public:

    /// this constructor will copy the protobuf data into a more efficient data structure
    MultiScalesIntegralChannelsModel(const doppia_protobuf::MultiScalesDetectorModel &model);
    ~MultiScalesIntegralChannelsModel();

protected:

    void sanity_check() const;
};

/// Helper function that normalizes the weights of multiple models
void normalized_bundle(const doppia_protobuf::DetectorModelsBundle &model,
                       doppia_protobuf::DetectorModelsBundle &rescaled_model);


} // namespace doppia

#endif // DOPPIA_MULTISCALESINTEGRALCHANNELSMODEL_HPP
