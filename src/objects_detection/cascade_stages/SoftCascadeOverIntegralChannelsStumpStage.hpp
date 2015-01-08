#ifndef DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSTAGE_HPP
#define DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSTAGE_HPP

#include "SoftCascadeOverIntegralChannelsFastStage.hpp" // defines DecisionStumpWithWeights

namespace doppia {


/// This class aims the same function of SoftCascadeOverIntegralChannelsFastStage,
/// however it uses a stump classifier as weak classifier, instead of the level2 decision tree
/// @see SoftCascadeOverIntegralChannelsFastStage
class SoftCascadeOverIntegralChannelsStumpStage
{
public:

    typedef DecisionStumpWithWeights weak_classifier_t;

    weak_classifier_t weak_classifier;

    /// if (strong_classifier_score < cascade_threshold) answer is "not this class"
    float cascade_threshold;

    const weak_classifier_t::rectangle_t &get_bounding_box() const;
};


inline
const SoftCascadeOverIntegralChannelsStumpStage::weak_classifier_t::rectangle_t &
SoftCascadeOverIntegralChannelsStumpStage::get_bounding_box() const
{
    return weak_classifier.feature.box;
}


} // end of namespace doppia

#endif // DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTUMPSTAGE_HPP
