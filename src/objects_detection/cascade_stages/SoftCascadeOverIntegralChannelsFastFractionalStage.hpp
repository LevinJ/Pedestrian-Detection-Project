#ifndef DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTFRACTIONALSTAGE_HPP
#define DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTFRACTIONALSTAGE_HPP

#include "helpers/geometry.hpp"
#include <boost/cstdint.hpp>

namespace doppia {

// forward declarations
class SoftCascadeOverIntegralChannelsStage;
class DecisionStump;
class IntegralChannelsFeature;

/// This is a variant of IntegralChannelsFeature,
/// where the coordinate_t is float
class IntegralChannelsFractionalFeature
{
public:
    typedef float coordinate_t;

    typedef doppia::geometry::point_xy<coordinate_t> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    __host__ __device__
    IntegralChannelsFractionalFeature();
    IntegralChannelsFractionalFeature(const IntegralChannelsFeature&feature);
    //~IntegralChannelsFractionalFeature();

    /// integral channel index
    boost::uint8_t channel_index;

    /// box over which compute the integral
    rectangle_t box;
};

/// Defined in the header to make CUDA happy
inline
IntegralChannelsFractionalFeature::IntegralChannelsFractionalFeature()
{
    // nothing to do here
    return;
}


/// just like DecisionStump, but assuming larger_than_threshold == true
class SimpleFractionalDecisionStump
{
public:

    typedef IntegralChannelsFractionalFeature feature_t;
    feature_t feature;

    /// thresholding the corresponding rectangle provides the a weak binary classifier
    float feature_threshold;

    inline bool operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        return (feature_value >= feature_threshold);
    }
};


class FractionalDecisionStumpWithWeights: public SimpleFractionalDecisionStump
{
public:

    __host__ __device__
    FractionalDecisionStumpWithWeights();
    FractionalDecisionStumpWithWeights(const DecisionStump &stump, const float feature_weight);
    //~FractionalDecisionStumpWithWeights();

    /// strong_classifier_score = sum_over_features(weight of weak_classifier_output)
    float weight_true_leaf, weight_false_leaf;

    inline float operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        return (feature_value >= feature_threshold)? weight_true_leaf : weight_false_leaf;
    }
};

/// Defined in the header to make CUDA happy
inline
FractionalDecisionStumpWithWeights::FractionalDecisionStumpWithWeights()
{
    // default constructor does nothing
    return;
}


class Level2FractionalDecisionTreeWithWeights
{
public:

    typedef SimpleFractionalDecisionStump::feature_t feature_t;

    SimpleFractionalDecisionStump level1_node;
    FractionalDecisionStumpWithWeights level2_true_node;
    FractionalDecisionStumpWithWeights level2_false_node;
};



/// This class is identical to SoftCascadeOverIntegralChannelsFastStage,
/// except that it is defined over fractional integral channel features.
/// These features are used the FastestDetectorInTheWest variants
/// @SoftCascadeOverIntegralChannelsFastStage
/// (yes, this name is ridiculously long)
class SoftCascadeOverIntegralChannelsFastFractionalStage
{
public:

    __host__ __device__
    SoftCascadeOverIntegralChannelsFastFractionalStage();
    SoftCascadeOverIntegralChannelsFastFractionalStage(const SoftCascadeOverIntegralChannelsStage &stage);
    //~SoftCascadeOverIntegralChannelsFastFractionalStage();


    //typedef DecisionStumpWithWeights weak_classifier_t;
    typedef Level2FractionalDecisionTreeWithWeights weak_classifier_t;

    weak_classifier_t weak_classifier;

    /// if (strong_classifier_score < cascade_threshold) answer is "not this class"
    float cascade_threshold;
};

/// Defined in the header to make CUDA happy
inline
SoftCascadeOverIntegralChannelsFastFractionalStage::SoftCascadeOverIntegralChannelsFastFractionalStage()
{
    // default constructor does nothing
    return;
}


} // namespace doppia

#endif // DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTFRACTIONALSTAGE_HPP
