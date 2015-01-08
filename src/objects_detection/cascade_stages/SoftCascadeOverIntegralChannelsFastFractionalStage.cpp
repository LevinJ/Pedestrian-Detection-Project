#include "SoftCascadeOverIntegralChannelsFastFractionalStage.hpp"

#include "SoftCascadeOverIntegralChannelsStage.hpp"

namespace doppia {

// This code is a copy-and-paste from SoftCascadeOverIntegralChannelsFastFractionalStage.cpp


SoftCascadeOverIntegralChannelsFastFractionalStage::SoftCascadeOverIntegralChannelsFastFractionalStage(
        const SoftCascadeOverIntegralChannelsStage &stage)
{

    cascade_threshold = stage.cascade_threshold;

    weak_classifier.level1_node.feature = stage.weak_classifier.level1_node.feature;
    weak_classifier.level1_node.feature_threshold = stage.weak_classifier.level1_node.feature_threshold;

    weak_classifier.level2_true_node = FractionalDecisionStumpWithWeights(stage.weak_classifier.level2_true_node, stage.weight);
    weak_classifier.level2_false_node = FractionalDecisionStumpWithWeights(stage.weak_classifier.level2_false_node, stage.weight);

    if(stage.weak_classifier.level1_node.larger_than_threshold == false)
    {
        std::swap(weak_classifier.level2_true_node, weak_classifier.level2_false_node);
    }

    return;
}

/*
SoftCascadeOverIntegralChannelsFastFractionalStage::~SoftCascadeOverIntegralChannelsFastFractionalStage()
{
    // nothing to do here
    return;
}
*/


FractionalDecisionStumpWithWeights::FractionalDecisionStumpWithWeights(const DecisionStump &stump, const float feature_weight)
{

    feature = stump.feature;
    feature_threshold = stump.feature_threshold;

    if(stump.larger_than_threshold)
    {
        weight_true_leaf = feature_weight;
        weight_false_leaf = -feature_weight;
    }
    else
    {
        weight_true_leaf = -feature_weight;
        weight_false_leaf = feature_weight;
    }
    return;
}

/*
FractionalDecisionStumpWithWeights::~FractionalDecisionStumpWithWeights()
{
    // nothing to do here
    return;
}*/



IntegralChannelsFractionalFeature::IntegralChannelsFractionalFeature(const IntegralChannelsFeature &feature)
{
    channel_index = feature.channel_index;

    // from integer to float convertions
    box.min_corner().x(feature.box.min_corner().x());
    box.min_corner().y(feature.box.min_corner().y());
    box.max_corner().x(feature.box.max_corner().x());
    box.max_corner().y(feature.box.max_corner().y());

    return;
}

/*
IntegralChannelsFractionalFeature::~IntegralChannelsFractionalFeature()
{
    // nothing to do here
    return;
}*/


} // namespace doppia
