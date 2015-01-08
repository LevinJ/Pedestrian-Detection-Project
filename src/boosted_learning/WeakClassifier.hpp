#ifndef WEAKCLASSIFER_HPP
#define WEAKCLASSIFER_HPP

#include "DecisionStump.hpp"
#include "DecisionTreeNode.hpp"

#include <boost/cstdint.hpp>
#include <boost/array.hpp>

#include <iostream>

namespace boosted_learning {

typedef boost::shared_ptr<DecisionStump> DecisionStumpPtr;
typedef boost::shared_ptr<const DecisionStump> ConstDecisionStumpPtr;
typedef boost::shared_ptr<DecisionTreeNode> DecisionTreeNodePtr;
typedef boost::shared_ptr<const DecisionTreeNode> ConstDecisionTreeNodePtr;
typedef std::vector<double> weights_t;

class WeakLearner;

enum WeakClassifierType {DECISION_TREE, STUMP_SET};

class WeakClassifier
{
public:

    typedef bootstrapping::integral_channels_t integral_channels_t;

    WeakClassifier();
    WeakClassifier(const bool silent_mode);

    virtual double predict_at_test_time(const WeakClassifier::integral_channels_t &integral_image) const = 0;
    virtual double predict_at_training_time(const FeaturesResponses &features_responses,
                                    const size_t training_sample_index) const = 0;

//protected:
public:
    bool _silent_mode; // both
    weights_t::value_type _cascade_threshold; // both??
    WeakClassifierType _tree_type;
};


}

#endif // WEAKCLASSIFER_HPP
