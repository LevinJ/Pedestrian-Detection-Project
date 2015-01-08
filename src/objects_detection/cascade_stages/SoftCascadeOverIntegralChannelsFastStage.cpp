#include "SoftCascadeOverIntegralChannelsFastStage.hpp"

#include <algorithm> // for min and max

namespace doppia {


SoftCascadeOverIntegralChannelsFastStage::SoftCascadeOverIntegralChannelsFastStage(
        const SoftCascadeOverIntegralChannelsStage &stage)
{

    cascade_threshold = stage.cascade_threshold;

    weak_classifier.level1_node.feature = stage.weak_classifier.level1_node.feature;
    weak_classifier.level1_node.feature_threshold = stage.weak_classifier.level1_node.feature_threshold;

    weak_classifier.level2_true_node = DecisionStumpWithWeights(stage.weak_classifier.level2_true_node, stage.weight);
    weak_classifier.level2_false_node = DecisionStumpWithWeights(stage.weak_classifier.level2_false_node, stage.weight);

    if(stage.weak_classifier.level1_node.larger_than_threshold == false)
    {
        std::swap(weak_classifier.level2_true_node, weak_classifier.level2_false_node);
    }

    weak_classifier.compute_bounding_box();
    return;
}

/*
SoftCascadeOverIntegralChannelsFastStage::~SoftCascadeOverIntegralChannelsFastStage()
{
    // nothing to do here
    return;
}*/




DecisionStumpWithWeights::DecisionStumpWithWeights(const DecisionStump &stump, const float feature_weight)
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

void Level2DecisionTreeWithWeights::compute_bounding_box()
{
    bounding_box = level1_node.feature.box;

    rectangle_t &bb = bounding_box;
    const rectangle_t
            &bb_a = level2_true_node.feature.box,
            &bb_b = level2_false_node.feature.box;

    bb.min_corner().x( std::min(bb.min_corner().x(), bb_a.min_corner().x()) );
    bb.min_corner().x( std::min(bb.min_corner().x(), bb_b.min_corner().x()) );

    bb.min_corner().y( std::min(bb.min_corner().y(), bb_a.min_corner().y()) );
    bb.min_corner().y( std::min(bb.min_corner().y(), bb_b.min_corner().y()) );

    bb.max_corner().x( std::max(bb.max_corner().x(), bb_a.max_corner().x()) );
    bb.max_corner().x( std::max(bb.max_corner().x(), bb_b.max_corner().x()) );

    bb.max_corner().y( std::max(bb.max_corner().y(), bb_a.max_corner().y()) );
    bb.max_corner().y( std::max(bb.max_corner().y(), bb_b.max_corner().y()) );

    return;
}

/*
DecisionStumpWithWeights::~DecisionStumpWithWeights()
{
    // nothing to do here
    return;
}*/


} // end of namespace doppia
