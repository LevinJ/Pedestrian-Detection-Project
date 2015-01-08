#ifndef LIBLINEARTRAINING_HPP
#define LIBLINEARTRAINING_HPP

#include "LinearSvmModel.hpp"

#include <linear.h>

#include <Eigen/Core>

#include <boost/program_options.hpp>
#include <boost/multi_array.hpp>

#include <string>

namespace doppia {

/// C++ wrapper over the C liblinear library
/// http://www.csie.ntu.edu.tw/~cjlin/liblinear
/// this is the doppia equivalent of shogun's
/// https://github.com/shogun-toolbox/shogun/blob/master/src/libshogun/classifier/svm/LibLinear.h
class LibLinearTraining
{
public:

    static boost::program_options::options_description get_args_options();

    LibLinearTraining(const boost::program_options::variables_map &options);
    ~LibLinearTraining();

    typedef Eigen::VectorXf feature_vector_t;

    /// if capacity was already set this call will do the required memory allocation
    void set_feature_vectors_size(const int feature_vectors_size);

    /// the capacity will define the maximum number of features vectors
    /// this call does the memory allocation for the requested capacity
    void set_feature_vectors_capacity(const int capacity);

    void set_number_of_feature_vectors(const int num_feature_vectors);
    void set_feature_vector(const size_t index, const int class_value, const feature_vector_t &feature_vector);

    size_t get_number_of_feature_vectors() const;

    void train();

    const LinearSvmModel &get_model() const;

    /// will save the training data in the SvmLight format
    void save_training_data(const std::string filename) const;

    /// will save the learned model in the LibLinear format
    void save_model(const std::string filename) const;

    /// predict the training data
    void predict();

    const std::vector<int> &get_expected_labels() const;
    const std::vector<int> &get_predicted_labels() const;

    /// this are the "raw" values that supported the predicted labels
    const std::vector<double> &get_decision_values() const;

protected:

    LinearSvmModel learned_svm_model;

    struct problem the_problem;
    struct parameter the_parameters;
    struct model* model_p;

    typedef boost::multi_array<struct feature_node, 2> feature_vectors_t;
    feature_vectors_t feature_vectors;
    std::vector<struct feature_node *> feature_vectors_pointers;
    std::vector<int> feature_vectors_labels, predicted_feature_vectors_labels;

    std::vector<double> decision_values;
};

} // end of namespace doppia

#endif // LIBLINEARTRAINING_HPP
