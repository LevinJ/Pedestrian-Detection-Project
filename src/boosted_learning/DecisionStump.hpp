#ifndef DECISIONSTUMP_HPP
#define DECISIONSTUMP_HPP

#include "Feature.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <vector>
#include <iostream>

namespace boosted_learning {


class DecisionStump
{
public:
    typedef std::vector<size_t> indices_t;
    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef boost::shared_ptr<DecisionStump> decision_stump_p;

    DecisionStump();
    DecisionStump(const Feature &feature, const size_t featureIndex, const int threshold, const int alpha);

    void setAlpha(const int alpha);
    int getAlpha() const;

    Feature _feature;
    size_t _feature_index;
    int _threshold;
    int _alpha;

    int get_feature_response(const integral_channels_t &integral_image) const;
    virtual int apply_stump(const int feature_response) const;

    void print(string prefix) const;
};


} // end of namespace boosted_learning

#endif // DECISIONSTUMP_HPP


