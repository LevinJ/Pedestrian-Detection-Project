#ifndef WeakLearner_H
#define WeakLearner_H

#include "TrainingData.hpp"
#include "Feature.hpp"

#include "WeakClassifierDecisionTree.hpp"
#include "WeakClassifierStumpSet.hpp"

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <vector>
#include <map>

namespace boosted_learning {


struct datum;
class WeakLearner
{

public:

    typedef std::vector<size_t> indices_t;
    typedef Eigen::VectorXd feature_vector_type;
    typedef TrainingData::objectClassLabel_t objectClassLabel_t;
    typedef std::vector<double> weights_t;

    WeakLearner();

    WeakLearner(const bool silent_mode, const int negClass,
                            TrainingData::const_shared_ptr_t trainingData,
                            const std::vector<objectClassLabel_t> & classes);

    void train_decision_tree(boost::shared_ptr<WeakClassifierDecisionTree> decision_tree, const weights_t &weights, const weights_t::value_type sumWeights=1.0);
    void train_stump_set(boost::shared_ptr<WeakClassifierStumpSet> stump_set, const weights_t &weights);
    weights_t determine_stump_set_weights(const weights_t &weights, boost::shared_ptr<WeakClassifierStumpSet> stump_set);

protected:

    void find_root_feature_threshold(
            const weights_t &weights,
            std::vector<bool> & indices,
            const size_t min_feature_index,
            weights_t::value_type &error_min, int &threshold_min, int &alphaMin) const;
    int find_leaf_feature_threshold(
            const weights_t &weights, const std::vector<bool> &indices ,const bool isLeft,
            const size_t min_feature_index,
            weights_t::value_type &error_min, int &threshold_min, int &alphaMin) const;

    void get_error_estimate_root(const weights_t &weights,
            const size_t featureIndex, weights_t::value_type &error) const;
    void getErrorEstimate_leaves(const weights_t &weights,
            const std::vector<bool> &indices, const size_t featureIndex,
            weights_t::value_type &error_left, weights_t::value_type &error_right) const;

    int create_root_node(const weights_t &weights, DecisionStump::decision_stump_p &stump, 
            weights_t::value_type &min_error, std::vector<bool> &indices) const;
    weights_t::value_type create_sibling_nodes(const weights_t &weights,
            std::vector<bool> &indices, DecisionStump::decision_stump_p &nodeLeft, DecisionStump::decision_stump_p &nodeRight) const;

    weights_t::value_type get_error_from_bins(const feature_vector_type &bin_neg, const feature_vector_type &bin_pos)const ;

    void sort_indices_based_on_data_positions(indices_t &positions, const size_t start) const;



    bool _silent_mode; // both
    const objectClassLabel_t _negative_class;
    TrainingData::const_shared_ptr_t _training_data;
    const std::vector<objectClassLabel_t> _classes;

private:
    void calc_error_thresh_alpha(std::vector<datum> & data,
                     const weights_t::value_type & sumpos,
                     const weights_t::value_type & sumNeg,
                     weights_t::value_type &error_min,
                     int &threshold_min,
                     int &alphaMin) const;
};

struct datum{
    FeaturesResponses::element feature_response;
    WeakLearner::objectClassLabel_t sample_class;
    weights_t::value_type weight;
};

} // end of namespace boosted_learning

#endif // WeakLearner_H
