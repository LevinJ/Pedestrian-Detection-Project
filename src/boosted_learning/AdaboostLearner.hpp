#ifndef __AdaboostLearner_H
#define __AdaboostLearner_H

#include "LabeledData.hpp"
#include "TrainingData.hpp"
#include "Feature.hpp"
#include "WeakLearner.hpp"
#include "ModelIO.hpp"
#include "IntegralChannelsComputer.hpp"

#include "applications/bootstrapping_lib/bootstrapping_lib.hpp"

#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/multi_array.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "linear.h"

namespace boosted_learning {

//#define DISCRETE_ADA	1
//#define GENTLE_ADA 2

class AdaboostLearner
{

public:
    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef Eigen::VectorXd feature_vector_t;
    typedef TrainingData::objectClassLabel_t object_class_label_t;
    typedef boosted_learning::weights_t weights_t;

    AdaboostLearner(bool silent_mode, TrainingData::shared_ptr_t data,
            const string type_adaboost, const int num_iterations, const string output_model_filename,
                    const boost::program_options::variables_map &options);
    ~AdaboostLearner();

    void train(int decision_tree_depth, std::string dataset_name, WeakClassifierType weak_classifier_type);
    weights_t::value_type obtain_error_rate(const std::vector<boost::shared_ptr<WeakClassifier> > & classifier);

    void set_num_training_rounds(const int i);
    void set_test_data(TrainingData::shared_ptr_t data);
    void set_validation_data(TrainingData::shared_ptr_t data);
    void set_output_model_filename(const std::string update);

    const TrainingData::shared_ptr_t get_training_data() const;
    std::string get_output_model_filename() const;

    /// DBP stands for Direct Backward Pruning. See C. Zang and P. Viola 2007
    static void to_soft_cascade_dbp(const LabeledData::shared_ptr data,
                                 const std::string input_model_filename,
                                 const std::string soft_cascade_filename,
                                 const  TrainingData::point_t  &model_window, const TrainingData::rectangle_t &object_window);

protected:
    bool _silent_mode;
    int _num_training_rounds;
    string _type_adaboost; ///<  default, GENTLE_ADA, other=DISCRETE_ADA
    string _output_model_filename; ///< name of file.

    TrainingData::shared_ptr_t _training_data; ///< this variable is used to hold all the information of the training set.
    TrainingData::shared_ptr_t _test_data; ///< this variable is used to hold all the information of the testing set.
    TrainingData::shared_ptr_t _validation_data; ///< this variable is used to hold all the information of the training set.

private:
    std::string get_timestamp_string();
};


void calc_minmax_feature_responses(TrainingData::shared_ptr_t train_data, MinOrMaxFeaturesResponsesSharedPointer minvs,
                                MinOrMaxFeaturesResponsesSharedPointer maxvs);

size_t filter_features(const Features & features_configurations,
                      std::vector<bool> &valid_features,
                      const TrainingData::point_t& model_window, size_t max_features =0);


} // end of namespace boosted_learning

#endif // __AdaboostLearner_H

