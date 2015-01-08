#ifndef WEAKCLASSIFIERSTUMPSET_HPP
#define WEAKCLASSIFIERSTUMPSET_HPP

#include "WeakClassifier.hpp"

namespace boosted_learning{

class WeakClassifierStumpSet : public WeakClassifier
{
public:
    WeakClassifierStumpSet(const bool silent_mode);

    weights_t::value_type predict_at_test_time(const WeakClassifier::integral_channels_t &integral_image) const;
    weights_t::value_type predict_at_training_time(const FeaturesResponses &features_responses,
                                   const size_t training_sample_index) const;

//protected:
public:
    std::vector<weights_t::value_type> _betas;
    std::vector<DecisionStump> _stumps;
    int _size; // decisiontree
    /*    bool _silent_mode; // both
    weights_t::value_type _cascade_threshold; // both??
    WeakClassifierType _tree_type; */
};


} // end of namespace boosted_learning

#endif // WEAKCLASSIFIERSTUMPSET_HPP
