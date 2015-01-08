#include "WeakLearner.hpp"

#include <boost/iterator/counting_iterator.hpp>

#include "helpers/Log.hpp"


#include <algorithm>
#include <limits>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

//#include <thrust/system/omp/vector.h>
//#include <thrust/sort.h>
//#include <thrust/extrema.h>

#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/get.hpp>


#include <omp.h>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "WeakLearner");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "WeakLearner");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "WeakLearner");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "WeakLearner");
}

} // end of anonymous namespace

namespace boosted_learning {

using boost::counting_iterator;


struct sort_pair {
    bool operator ()(std::pair<weights_t::value_type, size_t> const& a, std::pair<weights_t::value_type, size_t> const& b) {
        return a.first < b.first;
    }
};


WeakLearner::WeakLearner()
    :_negative_class(-1)
{
    // nothing to do here
    return;
}


WeakLearner::WeakLearner(const bool silent_mode, const int neg_class,
                                                 TrainingData::const_shared_ptr_t training_data,
                                                 const vector<objectClassLabel_t> &classes)
    : _silent_mode(silent_mode),
      _negative_class(neg_class),
      _training_data(training_data),
      _classes(classes)
{
    return;
}

struct comparator
{
    comparator(const FeaturesResponses &features_responses, const size_t feature_index):
        feature_responses(features_responses[feature_index])
    {
        // nothing to do here
        return;
    }

    bool operator()(const size_t training_sample_index_a, const size_t training_sample_index_b) const
    {
        return feature_responses[training_sample_index_a] < feature_responses[training_sample_index_b];
    }

    /// vector containing the responses of the the feature of interest
    FeaturesResponses::const_reference feature_responses;
};


void WeakLearner::sort_indices_based_on_data_positions(indices_t &positions,
                                                              const size_t feature_index) const
{
    const FeaturesResponses &features_responses = _training_data->get_feature_responses();
    std::sort(positions.begin(), positions.end(), comparator(features_responses, feature_index));

    return;
}


weights_t WeakLearner::determine_stump_set_weights(const weights_t &weights, boost::shared_ptr<WeakClassifierStumpSet> stump_set){

    const FeaturesResponses &features_responses = _training_data->get_feature_responses();
    std::vector< std::vector<bool> > classifications;


    //for all nodes get classification result
    for (size_t k=0; k< stump_set->_stumps.size(); ++k){
        std::vector<bool> classification(_training_data->get_num_examples(),false);
        DecisionStump &stump = stump_set->_stumps[k];
        const int feature_index = stump._feature_index;
        const FeaturesResponses::const_reference feature_responses = features_responses[feature_index];
        const int threshold = stump._threshold;
        for (size_t ne = 0; ne< _training_data->get_num_examples(); ne++){
            //correctly classified
            if (feature_responses[ne] < threshold){
                classification[ne] = true;
                //incorrectly classified
            }else{
                classification[ne] = false;
            }
        }
        classifications.push_back(classification);
    }
    std::vector<weights_t::value_type> beta(pow(2, stump_set->_stumps.size()), 0);
    std::vector<weights_t::value_type> weightsfg(pow(2, stump_set->_stumps.size()), 0);
    std::vector<weights_t::value_type> weightsbg(pow(2, stump_set->_stumps.size()), 0);
    for (size_t ne = 0; ne< _training_data->get_num_examples(); ne++){
        int pos = 1 * classifications[0][ne] + 2 * classifications[1][ne]+ 4 * classifications[2][ne];
        if (_classes[ne] != _negative_class)
        {
            weightsfg[pos] += weights[ne];
        }
        else
        {
            weightsbg[pos] +=weights[ne];
        }
    }

    for (size_t i = 0; i < pow(2,stump_set->_stumps.size()); ++i){
        if (weightsfg[i] < 1e-12)
            weightsfg[i] =1e-12;
        if (weightsbg[i] < 1e-12)
            weightsbg[i] =1e-12;


        weights_t::value_type wr = weightsfg[i]/(weightsfg[i]+ weightsbg[i]);


        beta[i] = 0.5 * log(wr/(1-wr));

    }

    return beta;
}



void WeakLearner::train_decision_tree(boost::shared_ptr<WeakClassifierDecisionTree> decision_tree,
        const weights_t &weights, const weights_t::value_type sum_weights)
{
    weights_t::value_type min_error = 0;
    std::vector<bool> samples_above_root_stump_threshold;

    DecisionStump::decision_stump_p root;
    create_root_node(weights, root, min_error, samples_above_root_stump_threshold);

    decision_tree->_decision_nodes[0] = DecisionTreeNode(root->_feature, root->_feature_index, root->_threshold, root->_alpha, 1);

    if (decision_tree->_depth==1)
    {
        DecisionStump::decision_stump_p root_left;
        DecisionStump::decision_stump_p root_right;        
        min_error = create_sibling_nodes(weights, samples_above_root_stump_threshold, root_left, root_right);

        decision_tree->_decision_nodes[1] = DecisionTreeNode(root_left->_feature, root_left->_feature_index, root_left->_threshold, root_left->_alpha, 2);
        decision_tree->_decision_nodes[2] = DecisionTreeNode(root_right->_feature, root_right->_feature_index, root_right->_threshold, root_right->_alpha, 3);

    }

    else if ((decision_tree->_depth > 1) || (decision_tree->_depth < 0)){
        throw std::runtime_error("ERROR: decision tree depth must be 0 or 1");
    }

    if (min_error < 1e-12)
    {
        decision_tree->set_beta(14);
    }
    else
    {
        decision_tree->set_beta(0.5 * log((sum_weights - min_error) / min_error));
    }

    if (!_silent_mode) // verbosity_level > 3
    {
        //root->print(log_debug());
    }

    return;
}


/// this is the function for the root node
int WeakLearner::create_root_node(
        const weights_t &weights,
        DecisionStump::decision_stump_p &stump,
        weights_t::value_type &min_error, std::vector<bool> &indices) const
{
    //for all features get responses on every image
    //find feature with lowest error
    const size_t num_features = _training_data->get_feature_pool_size();
    std::vector<std::pair<weights_t::value_type, size_t> >
            min_errors_for_search(num_features, std::make_pair(std::numeric_limits<weights_t::value_type>::max(), 0));

    //find max valid feature index
    size_t max_valid_feature_index= 0;
    for (size_t featureIndex = num_features-1; featureIndex >=0; --featureIndex)
    {
        if (_training_data->get_feature_validity(featureIndex))
        {
            max_valid_feature_index = featureIndex+1;
            break;
        }
    }

#pragma omp parallel for schedule(guided)
    for (size_t feature_index = 0; feature_index < max_valid_feature_index; ++feature_index)
    {
        if (_training_data->get_feature_validity(feature_index))
        {
            weights_t::value_type error = std::numeric_limits<weights_t::value_type>::max();
            get_error_estimate_root(weights, feature_index, error);
            min_errors_for_search[feature_index] = std::make_pair(error, feature_index);
        }
    } // end of "for each feature"

    int min_feature_index = -1;
    std::vector< std::pair<weights_t::value_type, size_t> >::iterator
            element_it  = std::min_element(min_errors_for_search.begin(),
                                           min_errors_for_search.end(), sort_pair());
    // min_element will find a valid iterator as long as the vector is not empty
    min_error = element_it->first;
    min_feature_index = element_it->second;

    if (min_feature_index == -1)
    {
        throw std::runtime_error("Error during creating the root node. Best Feature could not be set.");
        return -1;
    }


    int min_threshold = -1;
    int alpha_min = -1;

    find_root_feature_threshold(weights, indices, min_feature_index, min_error, min_threshold, alpha_min);

    // set the shared_ptr to point towards a new node instance
    stump.reset(new DecisionStump(_training_data->get_feature(min_feature_index), min_feature_index, min_threshold, alpha_min));

    return 0;
}


weights_t::value_type WeakLearner::create_sibling_nodes(
        const weights_t &weights,
        std::vector<bool> &indices,
        DecisionStump::decision_stump_p &node_left, DecisionStump::decision_stump_p &node_right) const
{
    //for all features get responses on every image
    //find feature with lowest error

    const size_t num_features = _training_data->get_feature_pool_size();
    std::vector<std::pair<weights_t::value_type, size_t> >
            min_errors_for_search_left(num_features, std::make_pair(std::numeric_limits<weights_t::value_type>::max(), 0));
    std::vector<std::pair<weights_t::value_type, size_t> >
            min_errors_for_search_right(num_features, std::make_pair(std::numeric_limits<weights_t::value_type>::max(), 0));

    //find max valid feature index
    size_t max_valid_feature_index= 0;
    for (size_t feature_index = num_features-1; feature_index >=0; --feature_index)
    {
        if (_training_data->get_feature_validity(feature_index))
        {
            max_valid_feature_index = feature_index+1;
            break;
        }
    }

#pragma omp parallel for schedule(guided)
    for (size_t feature_index = 0; feature_index < max_valid_feature_index; ++feature_index)
    {
        if (_training_data->get_feature_validity(feature_index)){

            weights_t::value_type error_left = std::numeric_limits<weights_t::value_type>::max();
            weights_t::value_type error_right = std::numeric_limits<weights_t::value_type>::max();
            getErrorEstimate_leaves(weights, indices, feature_index, error_left, error_right);
            min_errors_for_search_left[feature_index] = std::make_pair(error_left, feature_index);
            min_errors_for_search_right[feature_index] = std::make_pair(error_right, feature_index);
        }
    } // end of "for each feature"


    int min_feature_index_left = -1;
    int min_feature_index_right = -1;
    weights_t::value_type min_error_right=0;
    weights_t::value_type min_error_left =0;

    std::vector< std::pair<weights_t::value_type, size_t> >::iterator
            element_it_left  = std::min_element(min_errors_for_search_left.begin(),
                                        min_errors_for_search_left.end(), sort_pair());
    // min_element will find a valid iterator as long as the vector is not empty
    min_error_left = element_it_left->first;
    min_feature_index_left = element_it_left->second;

    std::vector< std::pair<weights_t::value_type, size_t> >::iterator
            element_it_right  = std::min_element(min_errors_for_search_right.begin(),
                                        min_errors_for_search_right.end(), sort_pair());
    // min_element will find a valid iterator as long as the vector is not empty
    min_error_right = element_it_right->first;
    min_feature_index_right = element_it_right->second;

    if (min_feature_index_left == -1 || min_feature_index_right ==-1)
    {
        throw(std::runtime_error("ERROR: something is terribly wrong here!"));
    }


    int min_threshold_left = -1;
    int alpha_min_left = -1;
    int min_threshold_right = -1;
    int alpha_min_right = -1;

    find_leaf_feature_threshold(weights,indices,true,
                      min_feature_index_left, min_error_left, min_threshold_left,
                      alpha_min_left);
    find_leaf_feature_threshold(weights,indices,false,
                      min_feature_index_right, min_error_right, min_threshold_right,
                      alpha_min_right);
    // set the shared_ptr to point towards a new node instance
    node_left.reset(new DecisionStump(_training_data->get_feature(min_feature_index_left),
                            min_feature_index_left, min_threshold_left, alpha_min_left));
    node_right.reset(new DecisionStump(_training_data->get_feature(min_feature_index_right),
                            min_feature_index_right, min_threshold_right, alpha_min_right));

    return min_error_right + min_error_left;
}



void WeakLearner::train_stump_set(boost::shared_ptr<WeakClassifierStumpSet> stump_set,
        const weights_t &weights)
{
    weights_t::value_type error_sum = 0;
    std::vector<bool> samples_above_root_stump_threshold;

    DecisionStump::decision_stump_p root;
    DecisionStump::decision_stump_p root_left;
    DecisionStump::decision_stump_p root_right;
    create_root_node(weights, root, error_sum, samples_above_root_stump_threshold);

    create_sibling_nodes(weights, samples_above_root_stump_threshold, root_left, root_right);

    if (!_silent_mode) // verbosity_level > 3
    {
         //root->print(log_debug());
    }

    stump_set->_stumps[0] =  DecisionStump(root->_feature, root->_feature_index, root->_threshold, root->_alpha);
    stump_set->_stumps[1] =  DecisionStump(root_left->_feature, root_left->_feature_index, root_left->_threshold, root_left->_alpha);
    stump_set->_stumps[2] =  DecisionStump(root_right->_feature, root_right->_feature_index, root_right->_threshold, root_right->_alpha);

    stump_set->_betas = determine_stump_set_weights(weights, stump_set);

    return;
}



class sort_data{
public:
    bool operator()(const datum &a, const datum &b) const{
        return a.feature_response < b.feature_response;
    }
};


int WeakLearner::find_leaf_feature_threshold(
        const weights_t &weights, const std::vector<bool> &indices ,const bool is_left,
        const size_t min_feature_index,
        weights_t::value_type &error_min, int &threshold_min, int &alpha_min) const
{

    //copy the feature values of the feature with minindex
    weights_t::value_type sum_pos = 0;
    weights_t::value_type sum_neg = 0;
    const size_t num_valid_samples = weights.size();

    FeaturesResponses::const_reference the_feature_responses = _training_data->get_feature_responses()[min_feature_index];
    std::vector<datum> data;
    data.reserve(num_valid_samples);

    for (size_t k = 0; k< num_valid_samples; ++k){
        if (is_left != indices[k])
            continue;
        datum d;
        d.feature_response = the_feature_responses[k];
        d.sample_class = _classes[k];
        d.weight = weights[k];
        if (d.sample_class == _negative_class)
        {
            sum_neg += d.weight;
        }
        else
        {
            sum_pos += d.weight;
        }
        data.push_back(d);
    }
    //add dummy
    datum d;
    d.feature_response = std::numeric_limits<int>::max();
    d.sample_class = _classes[num_valid_samples-1];
    d.weight = 0.0;
    data.push_back(d);

    //gets sorted in function
    //std::sort(data.begin(), data.end(), sort_data());

    if (!_silent_mode) // verbosity_level > 3
    {
        log_debug() << "sumneg: " << sum_neg << " sumpos: " << sum_pos << " both: " << sum_pos + sum_neg << "\n";
    }

     calc_error_thresh_alpha(data, sum_pos, sum_neg, error_min, threshold_min, alpha_min);

    if (!_silent_mode) // verbosity_level > 3
    {
        log_debug() << "threshold: " << threshold_min << " alpha: " << alpha_min << " error: " << error_min;
    }

    return 0;
}


void WeakLearner::getErrorEstimate_leaves(
        const weights_t &weights,
        const std::vector<bool> &indices, const size_t feature_index, weights_t::value_type &error_left, weights_t::value_type & error_right) const
{
    bintype num_bins = _training_data->get_bin_size();

    feature_vector_type
            bin_pos_left = feature_vector_type::Zero(num_bins + 1),
            bin_neg_left = feature_vector_type::Zero(num_bins + 1);
    feature_vector_type
            bin_pos_right = feature_vector_type::Zero(num_bins + 1),
            bin_neg_right = feature_vector_type::Zero(num_bins + 1);
    error_left = std::numeric_limits<weights_t::value_type>::max();
    error_right = std::numeric_limits<weights_t::value_type>::max();

    //const weights_t::value_type bin_scaling = num_bins / static_cast<weights_t::value_type>(maxv - minv);
    const FeaturesBinResponses &features_bin_responses = _training_data->get_feature_bin_responses();
    const FeaturesBinResponses::const_reference feature_bin_responses = features_bin_responses[feature_index];
    const size_t num_positive_examples = _training_data->get_num_positive_examples();

    for (size_t training_sample_index = 0; training_sample_index < num_positive_examples; ++training_sample_index)
    {
        const weights_t::value_type weight = weights[training_sample_index];
        const bintype bin = feature_bin_responses[training_sample_index];
        if (indices[training_sample_index]){
            bin_pos_left(bin) += weight;
        }else{
            bin_pos_right(bin) += weight;
        }

    } // end of "for each sample that falls into this node"

    for (size_t training_sample_index = num_positive_examples+1; training_sample_index < weights.size(); ++training_sample_index)
    {
        const weights_t::value_type weight = weights[training_sample_index];
        const bintype bin = feature_bin_responses[training_sample_index];
        if (indices[training_sample_index]){
            bin_neg_left(bin) += weight;
        }else{
            bin_neg_right(bin) += weight;
        }
    } // end of "for each sample that falls into this node"

    error_left = get_error_from_bins(bin_neg_left, bin_pos_left);
    error_right = get_error_from_bins(bin_neg_right, bin_pos_right);
    if (error_left < 0){
        log_warning() << "WARNING: error = " << error_left << " numeric problems?\n";
        error_left = 0;
    }
    if (error_right < 0){
        log_warning() << "WARNING: error = " << error_right << " numeric problems?\n";
        error_right = 0;
    }

    return;
} // end of getErrorEstimate_leaves


weights_t::value_type WeakLearner::get_error_from_bins(const feature_vector_type & bin_neg, const feature_vector_type & bin_pos)const {

    weights_t::value_type
            cum_neg = bin_neg.sum(),
            cum_pos = bin_pos.sum();
    weights_t::value_type error =std::numeric_limits<weights_t::value_type>::max();
    bintype num_bins = _training_data->get_bin_size();
    //run test by setting this to return 0 with error 0
//    if (cumPos == 0 || cumNeg == 0)
//    {
//        throw(std::runtime_error(""))
//    }
    //std::cout << "estimat sumpos: " << cumPos << " sumneg: " << cumNeg << " sum: \n" << cumPos+ cumNeg;
    weights_t::value_type
            //positives left
            positives_left_error = cum_pos,
            negatives_left_error = 0,
            //positives right
            negatives_right_error = cum_neg,
            positives_right_error = 0;
    //int minbin = -1;

    for (int i = 0; i < num_bins+1; ++i)
    {
        const weights_t::value_type bin_error = min( positives_right_error + negatives_right_error, positives_left_error + negatives_left_error);

        positives_left_error -= bin_pos(i);
        negatives_left_error += bin_neg(i);

        positives_right_error += bin_pos(i);
        negatives_right_error -= bin_neg(i);

        // we keep the min error
        if (bin_error < error)
        {
            //minbin = i;
            error = bin_error;
        }
    } // end of "or each bin"
    return error;
}

inline
void WeakLearner::get_error_estimate_root(
        const weights_t &weights, const size_t feature_index,
        weights_t::value_type &error) const
{

    bintype num_bins = _training_data->get_bin_size();

    feature_vector_type
            bin_pos = feature_vector_type::Zero(num_bins + 1),
            bin_neg = feature_vector_type::Zero(num_bins + 1);

    error = std::numeric_limits<weights_t::value_type>::max();

    //const weights_t::value_type bin_scaling = num_bins / static_cast<weights_t::value_type>(maxv - minv);
    const FeaturesBinResponses &features_bin_responses = _training_data->get_feature_bin_responses();
    const FeaturesBinResponses::const_reference feature_bin_responses = features_bin_responses[feature_index];
    const size_t numPositiveExamples = _training_data->get_num_positive_examples();

    //first only the positives
    for (size_t training_sample_index = 0; training_sample_index < numPositiveExamples; ++training_sample_index)
    {
        const weights_t::value_type weight = weights[training_sample_index];
        const bintype bin = feature_bin_responses[training_sample_index];
        bin_pos(bin) += weight;

    } // end of "for each sample that falls into this node"

    //now the negatives
    for (size_t training_sample_index = numPositiveExamples; training_sample_index < weights.size(); ++training_sample_index)
    {
        const weights_t::value_type weight = weights[training_sample_index];
        const bintype bin = feature_bin_responses[training_sample_index];
        bin_neg(bin) += weight;
    } // end of "for each sample that falls into this node"

    error = get_error_from_bins(bin_neg, bin_pos);
    if (error <0){
        log_warning() << "WARNING: error < 0, probably due to numeric instabilities\n";

        error = 0;
    }

}


void WeakLearner::find_root_feature_threshold(
        const weights_t &weights,
        std::vector<bool> & indices,
        const size_t min_feature_index,
        weights_t::value_type &error_min, int &threshold_min, int &alpha_min) const
{

    //copy the feature values of the feature with minindex
    weights_t::value_type sum_pos = 0;
    weights_t::value_type sum_neg = 0;
    const size_t num_valid_samples = weights.size();

    FeaturesResponses::const_reference the_feature_responses = _training_data->get_feature_responses()[min_feature_index];
    std::vector<datum> data(num_valid_samples+1);

    //generate structure to be sorted
    for (size_t k = 0; k< num_valid_samples; ++k){
        datum &d = data[k];
        d.feature_response = the_feature_responses[k];
        d.sample_class = _classes[k];
        d.weight = weights[k];
        if (d.sample_class == _negative_class)
        {
            sum_neg += d.weight;
        }
        else
        {
            sum_pos += d.weight;
        }
    }
    //add dummy in the end
    datum &d = data[num_valid_samples];
    d.feature_response = std::numeric_limits<int>::max();
    d.sample_class = _classes[num_valid_samples-1];
    d.weight = 0.0;

    indices.resize(num_valid_samples,false);
    error_min = std::numeric_limits<weights_t::value_type>::max();

    if (!_silent_mode) // verbosity_level > 3
    {
        log_debug() << "sumneg: " << sum_neg << " sumpos: " << sum_pos << " both: " << sum_pos + sum_neg << "\n";
    }

    calc_error_thresh_alpha(data, sum_pos, sum_neg, error_min, threshold_min, alpha_min);

    const bool min_than_threshold_value = true;

    for (size_t k = 0 ; k< num_valid_samples; ++k){

        if (the_feature_responses[k] < threshold_min){
            indices[k] = min_than_threshold_value;
        }
        else
        {
            indices[k] = not min_than_threshold_value;
        }
    }

    if (!_silent_mode) // verbosity_level > 3
    {

        log_debug() << "threshold: " << threshold_min << " alpha: " << alpha_min << " error: " << error_min;
    }

}


void WeakLearner::calc_error_thresh_alpha(std::vector<datum> & data,
                                                   const weights_t::value_type &sumPos,
                                                   const weights_t::value_type &sumNeg,
                                                   weights_t::value_type &error_min,
                                                   int &threshold_min,
                                                   int &alpha_min) const{
    error_min = std::numeric_limits<weights_t::value_type>::max();
//std::cout << "sumPos :" << sumPos << " sumNeg :" << sumNeg << " sum:" << sumPos+sumNeg << std::endl;
     std::sort(data.begin(), data.end(), sort_data());
     //set last feature response to (last-1) +1
     if (data.size() > 2)
        data[data.size()-1].feature_response = data[data.size()-2].feature_response +1;
     else
         data[data.size()-1].feature_response = std::numeric_limits<int>::max();
    //positives left
     weights_t::value_type positive_left_error = sumPos;
     weights_t::value_type negative_left_error = 0;

    //positives right
     weights_t::value_type negative_right_error = sumNeg;
     weights_t::value_type positive_right_error = 0;


    // go over all sorted data elements
    int previous_threshold = data[0].feature_response-1;

    for (size_t index = 0; index < data.size(); ++index)
    {
         weights_t::value_type error = 0;
        int alpha = 0;
        const int threshold = data[index].feature_response;

        if ((positive_left_error + negative_left_error) < (positive_right_error + negative_right_error))
        {
            alpha = 1;
            error = positive_left_error + negative_left_error;
        }
        else
        {
            alpha = -1;
            error = positive_right_error + negative_right_error;
        }

        if ((error < error_min) && (threshold != previous_threshold))
        {
            //std::cout << "current minimal Error " << error_min << std::endl;
            error_min = error;
            threshold_min = threshold;
            alpha_min = alpha;
           // std::cout << "Threshold: " << threshold << " Error: " << error << " error_min: " << error_min << " threshold_min: "<<threshold_min << std::endl;
        }
        previous_threshold = threshold;

        const weights_t::value_type weight = data[index].weight;
        const objectClassLabel_t sample_class = data[index].sample_class;

        if (sample_class == _negative_class)
        {
            negative_left_error += weight;
            negative_right_error -= weight;
        }
        else
        {
            positive_left_error -= weight;
            positive_right_error += weight;
        }
    }// end of "for each sorted data element"
    if (error_min <0){
        log_warning() << "WARNING: Error < 0 : " << error_min << " numeric instabilities?\n";
        error_min = 0;
    }
}


} // end of namespace boosted_learning


