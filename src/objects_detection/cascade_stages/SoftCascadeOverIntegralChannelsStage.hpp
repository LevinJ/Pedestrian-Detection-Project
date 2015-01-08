#ifndef DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTAGE_HPP
#define DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTAGE_HPP

#include "helpers/geometry.hpp"
#include <boost/cstdint.hpp>

namespace doppia {

/// efficient and compact version of the equivalent doppia_protobuf message
class IntegralChannelsFeature
{
public:
    typedef boost::int16_t coordinate_t;

    typedef doppia::geometry::point_xy<coordinate_t> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    /// integral channel index
    boost::uint8_t channel_index;

    /// box over which compute the integral
    rectangle_t box;
};


class DecisionStump
{
public:

    typedef IntegralChannelsFeature feature_t;
    feature_t feature;

    /// thresholding the corresponding rectangle provides the a weak binary classifier
    float feature_threshold;

    /// if true will evaluate (feature >= threshold),
    /// if false, will evaluate (feature < threshold)
    bool larger_than_threshold;

    inline bool operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        if(feature_value >= feature_threshold)
        {
            return larger_than_threshold;
        }
        else
        {
            return not larger_than_threshold;
        }
    }
};

class Level2DecisionTree
{
public:

    typedef DecisionStump::feature_t feature_t;

    DecisionStump level1_node;
    DecisionStump level2_true_node;
    DecisionStump level2_false_node;
};


class SoftCascadeOverIntegralChannelsStage
{
public:

    //typedef DecisionStump weak_classifier_t;
    typedef Level2DecisionTree weak_classifier_t;

    weak_classifier_t weak_classifier;

    /// strong_classifier_score = sum_over_features(weight*weak_classifier_output)
    float weight;

    /// if (strong_classifier_score < cascade_threshold) answer is "not this class"
    float cascade_threshold;

};

} // end of namespace doppia

#endif // DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSSTAGE_HPP
