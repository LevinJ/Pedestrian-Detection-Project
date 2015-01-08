#include "AdaboostLearner.hpp"

#include "helpers/Log.hpp"

#include <sys/mman.h>
#include <omp.h>
//#include <gperftools/profiler.h>


namespace boosted_learning
{

using namespace std;

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

} // end of anonymous namespac


AdaboostLearner::AdaboostLearner(bool silent_mode, TrainingData::shared_ptr_t data,
            const string type_adaboost, const int num_training_rounds, const string output_model_filename,
                                 const boost::program_options::variables_map &options):
    _silent_mode(silent_mode),  _type_adaboost(type_adaboost), _training_data(data)
{
    set_num_training_rounds(num_training_rounds);
    set_output_model_filename(boost::str( boost::format("%s%s")
                             % AdaboostLearner::get_timestamp_string()
                             % output_model_filename ));
std::cout << "output model name: " << boost::str( boost::format("%s%s") % AdaboostLearner::get_timestamp_string() % output_model_filename ) << std::endl;
    return;
}


AdaboostLearner::~AdaboostLearner()
{
    // nothing to do here
    return;
}


void AdaboostLearner::set_num_training_rounds(const int i)
{
    _num_training_rounds = i;
    return;
}


void AdaboostLearner::set_test_data(TrainingData::shared_ptr_t data)
{
    _test_data = data;
    return;
}


void AdaboostLearner::set_validation_data(TrainingData::shared_ptr_t data)
{
    _validation_data = data;
    return;
}


void AdaboostLearner::set_output_model_filename(const std::string update)
{
    _output_model_filename = update;
    return;
}


const TrainingData::shared_ptr_t AdaboostLearner::get_training_data() const
{
    return _training_data;
}


std::string AdaboostLearner::get_output_model_filename() const
{
    return _output_model_filename;
}


std::string AdaboostLearner::get_timestamp_string()
{

    using namespace boost::posix_time;
    const ptime current_time(second_clock::local_time());

    const std::string timestamp =
                boost::str( boost::format("%i_%02i_%02i_%i_")
                            % current_time.date().year()
                            % current_time.date().month().as_number()
                            % current_time.date().day()
                            % current_time.time_of_day().total_seconds()
                            );
    return timestamp;
}


weights_t::value_type AdaboostLearner::obtain_error_rate(const std::vector<boost::shared_ptr<WeakClassifier> > &weak_classifiers)
{
    if (!_test_data)
    {
        return -1;
    }

    StrongClassifier strongClassifier(weak_classifiers);
    int tp, fp, fn, tn;

    strongClassifier.classify(*(_test_data), tp, fp, fn, tn);
    weights_t::value_type error_rate = weights_t::value_type(fp + fn) / (tp + tn + fp + fn);
    log_info() << "Classification Results (TestData): " << std::endl;
    log_info() << "Detection Rate: " << weights_t::value_type(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    log_info() << "Error Rate: " << error_rate * 100 << " %" <<  std::endl;
    log_info() << "Error Positives: " <<  weights_t::value_type(fn) / (tp + fn) * 100 << " %" <<  std::endl;
    log_info() << "Error Negatives: " <<  weights_t::value_type(fp) / (tn + fp) * 100 << " %" <<  std::endl;
    log_info() << std::endl;

    return error_rate;
}


//***************************************************************************************************

void AdaboostLearner::train(int decision_tree_depth, std::string dataset_name, WeakClassifierType weak_classifier_type)
{
    if(weak_classifier_type!=DECISION_TREE && weak_classifier_type!=STUMP_SET)
    {
        throw runtime_error("TRAINING STOP: Unknown weak classifier type.");
    }

    ModelIO model_writer(_silent_mode);
    const std::string trained_model_name = "Model created via boosted_learning";
    model_writer.init_write(dataset_name,
                          doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                          trained_model_name,
                          _training_data->get_model_window(), _training_data->get_object_window());

    printf("\nStarting training with %zi positive samples and %zi negative samples (%zi total)\n",
           _training_data->get_num_positive_examples(), _training_data->get_num_negative_examples(), _training_data->get_num_examples());

    weights_t training_sample_weights(_training_data->get_num_examples());
    std::vector<object_class_label_t> class_labels(_training_data->get_num_examples());

    std::vector<std::string> filenames_background;
    filenames_background.reserve(_training_data->get_num_examples()*0.75); // rough heuristic

    for (size_t training_sample_index = 0; training_sample_index < _training_data->get_num_examples(); ++training_sample_index)
    {
        class_labels[training_sample_index] = _training_data->get_class_label_for_sample(training_sample_index);

        if (_training_data->get_class_label_for_sample(training_sample_index) == -1)
        {
            training_sample_weights[training_sample_index] = 1.0 / (2.0 * _training_data->get_num_negative_examples());
            filenames_background.push_back(_training_data->get_file_name(training_sample_index));
        }
        else
        {
            training_sample_weights[training_sample_index] = 1.0 / (2.0 * _training_data->get_num_positive_examples());
        }
    }// end of "for each training example"

    // (see entropy regularized LPBoost paper): setting a minimum weight is an heuristic to avoid "extreme cases", didn't work
    // also: previously computed min, max feature responses here

    std::vector<boost::shared_ptr<WeakClassifier> > weak_classifiers;
    weights_t total_classifier_output(_training_data->get_num_examples(), 0);

    bool previous_error_rate_is_zero = false;
    const bool enable_printing = true;

    //ProfilerStart("boosted_learning.prof");
    weights_t::value_type start_wall_time = omp_get_wtime();

    WeakLearner weak_learner(_silent_mode, -1, _training_data, class_labels);

    //boost::shared_ptr<WeakClassifier> weak_classifier; //TODO: there has to be a more elegant way than this (necessary since weak classifiers are declared in after cond. statements)

    for(int training_round = 0; training_round < _num_training_rounds; training_round +=1)
    {
        if (enable_printing && !previous_error_rate_is_zero)
        {
            std::cout << "Training stage: " << training_round << flush << std::endl;
        }

        boost::shared_ptr<WeakClassifier> weak_classifier;

        if(weak_classifier_type==DECISION_TREE)
        {
            weak_classifier.reset(new WeakClassifierDecisionTree(decision_tree_depth));
            weights_t::value_type sum_weights = 1;
            weak_learner.train_decision_tree(boost::static_pointer_cast<WeakClassifierDecisionTree>(weak_classifier), training_sample_weights, sum_weights);

            weights_t::value_type beta = boost::static_pointer_cast<WeakClassifierDecisionTree>(weak_classifier)->get_beta();
            if (beta <= 0)
            {
                std::cout << "current Beta: " << beta << std::endl << std::flush;
                throw runtime_error("TRAINING STOP: Not possible to reduce error anymore");
            }
        }
        else if(weak_classifier_type==STUMP_SET)
        {
            weak_learner.train_stump_set(boost::static_pointer_cast<WeakClassifierStumpSet>(weak_classifier), training_sample_weights);
        }
        else
        {
            throw runtime_error("TRAINING STOP: Unknown weak classifier type.");
        }

        // classify the data save misclassification indices
        int tp = 0, fp = 0, tn = 0, fn = 0;
        weights_t::value_type weight_normalisation_factor = 0;

        // this loop takes care of reweighting the training examples, and accumulating simple error counts
        for (size_t training_sample_index = 0; training_sample_index < _training_data->get_num_examples(); ++training_sample_index)
        {
            // update classifier outputs

            const weights_t::value_type weak_classifier_output = weak_classifier->predict_at_training_time(_training_data->get_feature_responses(), training_sample_index);

            total_classifier_output[training_sample_index] += weak_classifier_output;

            // reweight training samples
            weights_t::value_type &sample_weight = training_sample_weights[training_sample_index];

            if(weak_classifier_type==DECISION_TREE)
            {
                if (weak_classifier_output*class_labels[training_sample_index] < 0)
                {
                    sample_weight *= exp(boost::dynamic_pointer_cast<WeakClassifierDecisionTree>(weak_classifier)->get_beta());
                }
                else
                {
                    sample_weight *= exp(-boost::dynamic_pointer_cast<WeakClassifierDecisionTree>(weak_classifier)->get_beta());
                }

            }
            else if(weak_classifier_type==STUMP_SET)
            {
                if (weak_classifier_output*class_labels[training_sample_index] < 0)
                {
                    sample_weight *= exp(class_labels[training_sample_index] * weak_classifier_output);
                }
                else
                {
                    sample_weight *= exp(-class_labels[training_sample_index] * weak_classifier_output);
                }  
            }

            weight_normalisation_factor += sample_weight;

            if (total_classifier_output[training_sample_index] >= 0)
            {
                (_training_data->get_class_label_for_sample(training_sample_index) == 1) ? tp++ : fp++ ;
            }
            else
            {
                (_training_data->get_class_label_for_sample(training_sample_index) == 1) ? fn++ : tn++ ;
            }

        }

        for (size_t i = 0; i < _training_data->get_num_examples(); ++i)
        {
            training_sample_weights[i] /= weight_normalisation_factor;
        }

        const weights_t::value_type error_rate = weights_t::value_type(fp + fn) / _training_data->get_num_examples() * 100;

        if ((error_rate == 0) && enable_printing == false && previous_error_rate_is_zero == false)
        {
            std::cout << "\rError rate zero for the first time at iteration " << training_round << std::endl;
        }

        if (enable_printing)
        {
            if((previous_error_rate_is_zero == false) and (error_rate == 0))
            {
                std::cout << std::endl;
            }

            if((previous_error_rate_is_zero == false) or (error_rate != 0))
            {
                std::cout << "Classification Results (Trainset): " << std::endl;
                std::cout << "Detection Rate: " << 100 - error_rate << " %" <<  std::endl;
                std::cout << "Error Rate: " << error_rate << " %" <<  std::endl;
                std::cout << "Error Positives: " <<  weights_t::value_type(fn) / (tp + fn) * 100 << " %" <<  std::endl;
                std::cout << "Error Negatives: " <<  weights_t::value_type(fp) / (tn + fp) * 100 << " %" <<  std::endl;
                std::cout << std::endl;
            }
            else
            {
                // no need to print the same data again and again, overwrite the previous output line
                std::cout << "\rError rate stable at zero until " << training_round << std::flush;
            }
        }

        // update in case the error rate fluctuated
        previous_error_rate_is_zero = (error_rate == 0);

        // store the new learned weak classifier --
        weak_classifiers.push_back(weak_classifier); // converted from WeakLearner to WeakDiscreteTree

        model_writer.add_classifier_stage(weak_classifier);

        // save a temporary model
        if ((training_round % 500) == 100) // == 10 to avoid saving an empty model
        {
            printf("\nAverage speed in the last %i iterations is %.3f [Hz]\n",
                   training_round, training_round / (omp_get_wtime() - start_wall_time) );
            //ProfilerFlush();

            const string temporary_filename = _output_model_filename + ".tmp";
            model_writer.write(temporary_filename);
            log_info() << std::endl << "Created " << temporary_filename << std::endl;

            obtain_error_rate(weak_classifiers);
        }

    } // end of "for all the iterations"

    std::cout << std::endl;
    // ProfilerStop();

    printf("Total time for the %i iterations is %.3f [seconds]\n",
           _num_training_rounds,  (omp_get_wtime() - start_wall_time) );

    model_writer.write(_output_model_filename);

    return;
}


void AdaboostLearner::to_soft_cascade_dbp(const LabeledData::shared_ptr data,
                                       const std::string input_model_filename,
                                       const std::string soft_cascade_filename,
                                       const TrainingData::point_t &model_window,
                                       const TrainingData::rectangle_t &object_window)
{

    printf("Starting computation to create a DBP (discrete backward prunning (?)) soft cascade. Please be patient...\n");
    ModelIO model_reader;
    model_reader.read_model(input_model_filename);
    StrongClassifier learner = model_reader.read();

    // modelReader.print();
    const weights_t::value_type detection_rate = 1.0;
    //const weights_t::value_type detectionRate = 0.92; // This value corresponds to the FPDW and LatentSVM responses at 1 FPPI on INRIA
    learner.convert_to_soft_cascade(data, detection_rate);
    learner.write_classifier(soft_cascade_filename,
                            model_reader.get_model_training_dataset_name(),
                            model_window, object_window);

    return;
}


void calc_minmax_feature_responses(TrainingData::shared_ptr_t train_data, MinOrMaxFeaturesResponsesSharedPointer minvs,
                                MinOrMaxFeaturesResponsesSharedPointer maxvs){
    const FeaturesResponses &featuresResponses = train_data->get_feature_responses();

    for (size_t feature_index = 0; feature_index < train_data->get_feature_pool_size(); ++feature_index)
    {
        if (train_data->_validFeatures[feature_index] == false)
            continue;
        int minv = std::numeric_limits<int>::max();
        int maxv = -std::numeric_limits<int>::max();

        for (size_t example_index = 0; example_index < train_data->get_num_examples(); ++example_index)
        {
            const int val = featuresResponses[feature_index][example_index];
            minv = std::min(val, minv);
            maxv = std::max(val, maxv);
        } // end of "for each example"

        (*maxvs)[feature_index] = maxv;
        (*minvs)[feature_index] = minv;

    } // end of "for each feature"

}


size_t filter_features(const Features &features_configurations, std::vector<bool> &valid_features,
                      const TrainingData::point_t &model_window, size_t max_features)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            model_width = model_window.x() / shrinking_factor ,
            model_height = model_window.y() / shrinking_factor;
    //set the filter according to the shrinking factor
    size_t unused_features = 0;
    size_t used_features = 0;

    valid_features.clear();
    size_t no_featureconf = features_configurations.size();
    valid_features.resize(no_featureconf, false);
    unused_features = 0;// no_featureconf;
    //std::cout << validFeatures.size() << std::endl;
    //std::cout << featuresConfigurations.size() << std::endl;

    for (size_t i =0; i< features_configurations.size(); ++i)
    {
        const Feature &ff = features_configurations[i];

        //bottom or right
        if (((ff.y + ff.height) > model_height)
            or ((ff.x + ff.width) > model_width ))
        {
            valid_features[i] = false;
            unused_features +=1;
        }
        else{

            valid_features[i] = true;
            used_features +=1;
            if (max_features != 0 && used_features == max_features){
                unused_features = features_configurations.size()-used_features;
                break;
            }
        }
    }
    // std::cout << "features that are set to valid: " << used_features << std::endl;

    return unused_features;
}


} // end of namespace boosted_learning
