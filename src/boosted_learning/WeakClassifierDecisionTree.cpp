#include "WeakClassifierDecisionTree.hpp"

using namespace boosted_learning;

WeakClassifierDecisionTree::WeakClassifierDecisionTree(int depth) :
    _depth(depth)
{
    _decision_nodes.resize(pow(2.0, depth+1)-1);
    _tree_type = DECISION_TREE;

}

weights_t::value_type WeakClassifierDecisionTree::get_depth() const
{
    return _depth;
}


weights_t::value_type WeakClassifierDecisionTree::get_beta() const
{
    return _beta;
}


void WeakClassifierDecisionTree::set_beta(weights_t::value_type b)
{
    _beta = b;
}


// FIXME does this really help speed ? (do benchmarks)
// inlined for performance reasons
// FIXME does this really help speed ? (do benchmarks)
// inlined for performance reasons
inline
double WeakClassifierDecisionTree::predict_at_test_time(const WeakClassifier::integral_channels_t &integral_image) const
{
    int node_id = 1; // start at the root
    for (int i=0; i<_depth; i++)
    {
        int feature_response = _decision_nodes[node_id-1].get_feature_response(integral_image);
        node_id = 2*node_id + (1-_decision_nodes[node_id-1].apply_stump(feature_response));
    }
    bool decision = _decision_nodes[node_id-1].apply_stump(
                            _decision_nodes[node_id-1].get_feature_response(integral_image));
    int output_class = (1-decision) ? -1*_decision_nodes[node_id-1]._alpha : _decision_nodes[node_id-1]._alpha;
    return _beta*output_class;
}


inline
double WeakClassifierDecisionTree::predict_at_training_time(const FeaturesResponses &features_responses,
                               const size_t training_sample_index) const
{
    int node_id = 1; // start at the root
    for (int i=0; i<_depth; i++)
    {
        int feature_response = features_responses[_decision_nodes[node_id-1]._feature_index][training_sample_index];
        node_id = 2*node_id + (1-_decision_nodes[node_id-1].apply_stump(feature_response));

    }
    bool decision = _decision_nodes[node_id-1].apply_stump(
                            features_responses[_decision_nodes[node_id-1]._feature_index][training_sample_index]);
    int output_class = (1-decision) ? -1*_decision_nodes[node_id-1]._alpha : _decision_nodes[node_id-1]._alpha;
    return _beta*output_class;
}
