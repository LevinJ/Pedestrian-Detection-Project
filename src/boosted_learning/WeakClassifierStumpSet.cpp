#include "WeakClassifierStumpSet.hpp"

using namespace boosted_learning;

WeakClassifierStumpSet::WeakClassifierStumpSet(const bool silent_mode) :
    WeakClassifier::WeakClassifier(silent_mode)
{
    _tree_type = STUMP_SET;
    return;
}


//TODO: inline
double WeakClassifierStumpSet::predict_at_test_time(const WeakClassifier::integral_channels_t &integral_image) const
{
    int pos = 0;
    for (size_t i = 0; i< _stumps.size(); ++i)
    {
        if(_stumps[i].apply_stump(_stumps[i].get_feature_response(integral_image)))
        {
            pos += pow(2, i);
        }
    }
    return _betas[pos];
}

//TODO: inline
double WeakClassifierStumpSet::predict_at_training_time(const FeaturesResponses &features_responses,
                               const size_t training_sample_index) const
{
    //start at the root
    int pos = 0;
    for (size_t i = 0; i< _stumps.size(); ++i)
    {
        if (_stumps[i].apply_stump(features_responses[_stumps[i]._feature_index][training_sample_index]))
        {
            pos += pow(2,i);
        }
    }    return _betas[pos];
}
