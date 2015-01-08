#ifndef WEAKCLASSIFIERDECISIONTREE_HPP
#define WEAKCLASSIFIERDECISIONTREE_HPP

#include "WeakClassifier.hpp"

namespace boosted_learning {

class WeakClassifierDecisionTree : public WeakClassifier
{
public:

    WeakClassifierDecisionTree();
    WeakClassifierDecisionTree(int depth);

    weights_t::value_type get_depth() const;
    weights_t::value_type get_beta() const;

    void set_beta(weights_t::value_type b);

    virtual weights_t::value_type predict_at_test_time(const WeakClassifier::integral_channels_t &integral_image) const;
    virtual weights_t::value_type predict_at_training_time(const FeaturesResponses &features_responses,
                                   const size_t training_sample_index) const;

//protected:
public:
    double _beta; // a weight for the classifier's decision
    std::vector<DecisionTreeNode> _decision_nodes;
    int _depth;
};


} // end of namespace boosted_learning

#endif // WEAKCLASSIFIERDECISIONTREE_HPP
