#include "StrongClassifier.hpp"

#include "ModelIO.hpp"

#include "integral_channels_helpers.hpp"
#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method


#include <boost/filesystem.hpp>
#include <boost/array.hpp>
#include <boost/progress.hpp>

#include <cassert>

namespace boosted_learning {

namespace gil = boost::gil;
using boost::shared_ptr;


void adjust_cascade(std::vector<boost::shared_ptr<WeakClassifier> > &weak_classifiers)
{
    if (weak_classifiers[weak_classifiers.size()-1]->_cascade_threshold == -std::numeric_limits<float>::max())
    {
        weak_classifiers[weak_classifiers.size()-1]->_cascade_threshold = 0;
    }

    if (weak_classifiers.back()->_cascade_threshold <= -std::numeric_limits<int>::max())
        //throw runtime_error("there is a bug in the cascade: last stage threshold too negative");
    {
        weak_classifiers.back()->_cascade_threshold = 0;
    }

    return;
}


StrongClassifier::StrongClassifier(const std::vector<boost::shared_ptr<WeakClassifier> > &weak_classifiers)
{
    if (weak_classifiers.empty())
    {
        throw std::runtime_error("StrongClassifier must be constructed with "
                                 "a non-empty set of learners");
    }

    // copy data
    std::copy(weak_classifiers.begin(), weak_classifiers.end(), std::back_inserter(_weak_classifiers));

    adjust_cascade(_weak_classifiers);
    _weak_classifier_type = _weak_classifiers[0]->_tree_type; // we assume it is fixed for all nodes
    return;
}


void StrongClassifier::classify(const LabeledData &data,
                                         int &tp, int &fp, int &fn, int &tn,
                                         const bool use_cascade) const
{

    int tpp = 0, fpp = 0, fnn = 0, tnn = 0;

#pragma omp parallel for reduction(+:tpp, fpp, fnn, tnn) default(none) shared(data, std::cout)
    for (size_t i = 0; i < data.get_num_examples(); ++i)
    {
        float res = 0;
        bool goingthrough = true;

        for (size_t l = 0; l < _weak_classifiers.size(); ++l)
        {
            res += _weak_classifiers[l]->predict_at_test_time(data.get_integral_image(i));

            if (use_cascade and res < _weak_classifiers[l]->_cascade_threshold)
            {
                goingthrough = false;
                break;
            }

        }
        if (goingthrough)
        {
            (data.get_class_label_for_sample(i) == 1) ? tpp++ : fpp++ ;
        }
        else
        {
            (data.get_class_label_for_sample(i) == -1) ? tnn++ : fnn++ ;
        }
    }

    tn = tnn;
    tp = tpp;
    fp = fpp;
    fn = fnn;

    return;
}


void StrongClassifier::classify(const TrainingData &data,
                                int &tp, int &fp, int &fn, int &tn,
                                const bool use_cascade) const
{

    int tpp = 0, fpp = 0, fnn = 0, tnn = 0;

#pragma omp parallel for reduction(+:tpp, fpp, fnn, tnn) default(none) shared(data, std::cout)
    for (size_t i = 0; i < data.get_num_examples(); ++i)
    {
        float res = 0;
        bool goingthrough = true;

        for (size_t l = 0; l < _weak_classifiers.size(); ++l)
        {
            res += _weak_classifiers[l]->predict_at_training_time(data.get_feature_responses(), i);

            if (use_cascade and res < _weak_classifiers[l]->_cascade_threshold)
            {
                goingthrough = false;
                break;
            }

        }

        if (goingthrough)
        {
            data.get_class_label_for_sample(i) == 1 ? tpp++ : fpp++;
        }
        else
        {
            data.get_class_label_for_sample(i) == -1 ? tnn++ : fnn++;
        }

    }

    tn = tnn;
    tp = tpp;
    fp = fpp;
    fn = fnn;

    return;
}


float StrongClassifier::compute_classification_score(const integral_channels_t &integral_image) const
{
    float score = 0;

    for (size_t wc = 0; wc < _weak_classifiers.size(); wc += 1)
    {
        score += _weak_classifiers[wc]->predict_at_test_time(integral_image);
    }

    return score;
}



// enable re-ordering of features or not
#if 1
// no features re-ordering
void StrongClassifier::convert_to_soft_cascade(const LabeledData::shared_ptr data, const float detection_rate)
{
    const int num_examples = data->get_num_examples();

    // stores the score and the sample index
    std::vector<std::pair<float, int> > scores;
    std::vector<bool> valid(num_examples, true);

    boost::progress_display soft_cascade_progress(data->get_num_pos_examples() + _weak_classifiers.size());

#pragma omp parallel for default(none) shared(scores, soft_cascade_progress)
    for (int i = 0 ; i < num_examples; ++i)
    {
        float res = 0;

        if (data->get_class_label_for_sample(i) != -1) //data.getbackgroundClassLabel()){
        {
            //get result of every weak learner
            for (size_t l = 0; l < _weak_classifiers.size(); ++l)
            {
                res += _weak_classifiers[l]->predict_at_test_time(data->get_integral_image(i)); //, useModelSoftCascade);
            } //TODO: possible source of error, different classification method now?

#pragma omp critical
            {
                scores.push_back(std::make_pair<float, int>(res, i));
                ++soft_cascade_progress;
            }
        }
    }

    std::sort(scores.begin(), scores.end());
    //const int pos = int(scores.size() * (1 - detectionRate) + 0.5);

    //this is the cascasde threshold for the detection rate
    //const float theta = scores[pos].first - 1e-6;
    const float theta = 5; // FIXME hardcoded value
    std::cout << "Theta calculated: " << theta << flush << std::endl;

    for (int i = 0 ; i < (int)scores.size(); ++i)
    {
        if (scores[i].first < theta)
        {
            valid[scores[i].second] = false;
        }
    }

    std::vector<float> res(num_examples, 0);



    //find a valid positive sample with least score per stage
    for (size_t l = 0; l < _weak_classifiers.size(); ++l)
    {
        float stage_threshold = std::numeric_limits<float>::max();

        //#pragma omp parallel for default(none) shared(no, data, thisthr, valid, l, res)
        for (int i = 0 ; i < num_examples; ++i)
        {
            if (valid[i] and data->get_class_label_for_sample(i) != -1) //data.getbackgroundClassLabel()){
            {
                //std::string fn = data.get_file_name(i);
                res[i] += _weak_classifiers[l]->predict_at_test_time(data->get_integral_image(i));
                stage_threshold = min(stage_threshold, res[i]);
            }
        }

        std::cout << "stage_threshold: " << stage_threshold << std::endl;


        // remove a small value to make sure that < and <= work properly
        const float epsilon = 1e-6;
        _weak_classifiers[l]->_cascade_threshold = stage_threshold - epsilon;

        ++soft_cascade_progress;
    }

    return;
}
#else
// will do features re-ordering
void StrongClassifier::convert_to_soft_cascade(const LabeledData::shared_ptr data, const float detection_rate)
{
    const int num_examples = data->get_num_examples();
    std::vector<std::pair<float, int> > scores;
    std::vector<bool> valid(num_examples, true);

    boost::progress_display soft_cascade_progress(data->get_num_pos_examples() + _weak_classifiers.size());

#pragma omp parallel for default(none) shared(scores, soft_cascade_progress)
    for (int i = 0 ; i < num_examples; ++i)
    {
        float res = 0;

        if (data->get_class_label_for_sample(i) != -1) //data.getbackgroundClassLabel()){
        {
            //get result of every weak learner
            for (size_t l = 0; l < _weak_classifiers.size(); ++l)
            {
                res += _weak_classifiers[l]->predict_at_test_time(data->get_integral_image(i));
            }

#pragma omp critical
            {
                scores.push_back(std::make_pair<float, int>(res, i));
                ++soft_cascade_progress;
            }
        }
    }

    std::sort(scores.begin(), scores.end());
    int position = int(scores.size() * (1 - detection_rate) + 0.5);

    //this is the cascasde threshold for the detection rate dr
    float theta = scores[position].first - 1e-6;
    //std::cout << "Theta calculated: " << theta << flush << std::endl;

    for (size_t i = 0 ; i < scores.size(); ++i)
    {
        if (scores[i].first < theta)
        {
            valid[scores[i].second] = false;
        }
    }


    // Start doing the reordering of the features (more discriminative first) --

    std::vector<float> score_results(num_examples, 0);
    std::vector<bool> learners_already_used(_learners.size(), false);
    //find a valid positive sample with least score per stage
    std::vector<int> new_indices;
    for (size_t l = 0; l < _learners.size(); ++l)
    {

        float stage_threshold = std::numeric_limits<float>::max();
        float steepest_stage_threshold = -std::numeric_limits<float>::max();
        int steepest_learner_index = -1;
        for (size_t k = 0; k < _learners.size(); ++k)
        {
            if (learners_already_used[k] == false)
            {
                std::vector<float> tmp_res(score_results);
                float minimum_stage_threshold = std::numeric_limits<float>::max();

                for (int i = 0; i < num_examples; ++i)
                {
                    if (valid[i] and (data->get_class_label_for_sample(i) != -1)) //data.getbackgroundClassLabel()){
                    {
                        //std::string fn = data.get_file_name(i);
                        tmp_res[i] += _weak_classifiers[k].classify(data->get_integral_image(i));
                        if (tmp_res[i] < minimum_stage_threshold){
                            minimum_stage_threshold = tmp_res[i];
                        }
                    }
                } // end of "for each example"

                //now search the biggest one
                if (minimum_stage_threshold > steepest_stage_threshold)
                {
                    steepest_stage_threshold = minimum_stage_threshold;
                    steepest_learner_index = k;
                }
            }
        } // end of "for each learner"

        for (int i = 0 ; i < num_examples; ++i)
        {
            if (valid[i] and data->get_class_label_for_sample(i) != -1) //data.getbackgroundClassLabel()){
            {
                //std::string fn = data.get_file_name(i);
                score_results[i] += _learners[steepest_learner_index].predict_at_test_time(data->get_integral_image(i), use_model_soft_cascade);
            }
        }

        learners_already_used[steepest_learner_index] = true;
        stage_threshold = steepest_stage_threshold;
        std::cout << "stage_threshold: " << stage_threshold << std::endl;
        std::cout << "stage:  " << l << " steepest  idx: " << steepest_learner_index << std::endl;

        const float epsilon = 1e-6;
        _learners[steepest_learner_index]._cascade_threshold = stage_threshold - epsilon;
        new_indices.push_back(steepest_learner_index);
        ++soft_cascade_progress;

    } // end of "for each learner"

    for(size_t index=0; index < _learners.size(); index+=1)
    {
        if(std::count(new_indices.begin(), new_indices.end(), index) != 1)
        {
            throw std::runtime_error("StrongClassifier::convert_to_soft_cascade new_indices is flawed. "
                                     "Something went terribly wrong");
        }
    }

    //sort learners
    std::vector<WeakDiscreteTree> reordered_learners;
    for (size_t i = 0; i< new_indices.size(); ++i)
    {
        reordered_learners.push_back(_learners[new_indices[i]]);
    }

    _learners = reordered_learners;

    return;
}
#endif


void StrongClassifier::write_classifier(
        const std::string filename,
        const std::string trained_dataset_name,
        const point_t model_window, const rectangle_t object_window)
{
    ModelIO model_writer;

    const std::string trained_model_name = "Soft cascade Model created via boosted_learning";

    model_writer.init_write(trained_dataset_name,
                          doppia_protobuf::DetectorModel::SoftCascadeOverIntegralChannels,
                          trained_model_name,
                          model_window, object_window);

    for (size_t i = 0; i < _weak_classifiers.size(); ++i)
    {
        model_writer.add_classifier_stage(_weak_classifiers[i]);
    }

    model_writer.write(filename);
    google::protobuf::ShutdownProtobufLibrary();

    return;
}

} // end of namespace boosted_learning

