#include "LibLinearTraining.hpp"

#include "helpers/get_option_value.hpp"

#include <string>
#include <fstream>

namespace doppia {

using namespace std;

boost::program_options::options_description LibLinearTraining::get_args_options()
{
    boost::program_options::options_description desc("LibLinearTraining options");

    desc.add_options()

            //("fraction_of_positives,f",
            //program_options::value<float>()->default_value(0.25),
            //"which fraction of positives examples should be part the total training set. 0.5 indicates equal number of negative and positive samples")

            ("liblinear.use_primal",
             boost::program_options::value<bool>()->default_value(false),
             "use primal linear SVM solution instead of dual")

            ("liblinear.C",
             boost::program_options::value<float>()->default_value(1),
             "C value of the linear SVM solver which control the trade-off between large margin or small training error")

            ;

    return desc;
}


LibLinearTraining::LibLinearTraining(const boost::program_options::variables_map &options)
{
    // default values
    the_parameters.solver_type = L2R_L2LOSS_SVC_DUAL;

    const bool use_primal = get_option_value<bool>(options, "liblinear.use_primal");
    if(use_primal)
    {
        the_parameters.solver_type = L2R_L2LOSS_SVC;
    }

    //the_parameters.C = 1;
    the_parameters.C = get_option_value<float>(options, "liblinear.C");


    the_parameters.eps =  HUGE_VAL; // see setting below
    the_parameters.nr_weight = 0;
    the_parameters.weight_label = NULL;
    the_parameters.weight = NULL;

    the_problem.n = 0; // length of each feature vector
    the_problem.l = 0; // number of feature vectors
    the_problem.bias = -1;
    the_problem.x = NULL;
    the_problem.y = NULL;

    model_p = NULL;

    if(the_parameters.eps == HUGE_VAL)
    {
        if(the_parameters.solver_type == L2R_LR || the_parameters.solver_type == L2R_L2LOSS_SVC)
        {
            the_parameters.eps = 0.01;
        }
        else if(the_parameters.solver_type == L2R_L2LOSS_SVC_DUAL || the_parameters.solver_type == L2R_L1LOSS_SVC_DUAL || the_parameters.solver_type == MCSVM_CS || the_parameters.solver_type == L2R_LR_DUAL)
        {
            the_parameters.eps = 0.1;
        }
        else if(the_parameters.solver_type == L1R_L2LOSS_SVC || the_parameters.solver_type == L1R_LR)
        {
            the_parameters.eps = 0.01;
        }
    } // end of "if eps not set"


    if(the_problem.bias > 0)
    {
        throw std::runtime_error("Positive bias not yet handled");
    }

    return;
}


void free_model_content(struct model *model_ptr)
{
    if(model_ptr->w != NULL)
    {
        free(model_ptr->w);
    }
    if(model_ptr->label != NULL)
    {
        free(model_ptr->label);
    }
    return;
}


void free_and_destroy_model(struct model **model_ptr_ptr)
{
    struct model *model_ptr = *model_ptr_ptr;
    if(model_ptr != NULL)
    {
        doppia::free_model_content(model_ptr);
        free(model_ptr);
    }

    model_ptr = NULL;
    return;
}


void destroy_parameters(parameter& param)
{
    if(param.weight_label != NULL)
    {
        free(param.weight_label);
    }
    if(param.weight != NULL)
    {
        free(param.weight);
    }
    return;
}

LibLinearTraining::~LibLinearTraining()
{
    destroy_parameters(the_parameters);

    doppia::free_and_destroy_model(&model_p);

    return;
}


void LibLinearTraining::set_feature_vectors_size(const int feature_vectors_size)
{
    feature_vectors.resize(boost::extents[feature_vectors.shape()[0]][feature_vectors_size + 1]);
    the_problem.n = feature_vectors_size;
    return;
}


void LibLinearTraining::set_feature_vectors_capacity(const int capacity)
{
    feature_vectors.resize(boost::extents[capacity][feature_vectors.shape()[1]]);
    return;
}


void LibLinearTraining::set_number_of_feature_vectors(const int num_feature_vectors)
{
    size_t capacity = feature_vectors.shape()[0];
    if(num_feature_vectors > capacity)
    {
        throw std::runtime_error("LibLinearTraining::set_number_of_feature_vectors called with a larger number than LibLinearTraining::set_feature_vectors_capacity");
    }

    the_problem.l = num_feature_vectors;

    feature_vectors_pointers.resize(num_feature_vectors);
    for(int i=0; i < num_feature_vectors; i+=1)
    {
        feature_vectors_pointers[i] = feature_vectors[i].origin();
    }
    the_problem.x = feature_vectors_pointers.data();

    feature_vectors_labels.resize(num_feature_vectors);
    the_problem.y = feature_vectors_labels.data();
    return;
}


size_t LibLinearTraining::get_number_of_feature_vectors() const
{
    return feature_vectors_pointers.size();
}


void LibLinearTraining::set_feature_vector(const size_t index, const int class_value, const feature_vector_t &input_vector)
{

    if(feature_vectors.size() == 0)
    {
        throw std::runtime_error("LibLinearTraining::set_feature_vector called before setting the feature vectors size and the number of feature vectors");
    }

    feature_vectors_t::reference feature_vector = feature_vectors[index];

    assert(feature_vector.size() == (input_vector.size() + 1));

    for(int i=0; i < input_vector.size(); i+=1)
    {
        feature_vector[i].index = i + 1;
        feature_vector[i].value = input_vector(i);
    }

    feature_vector[feature_vector.size() - 1].index = -1; // mark the end of the list
    // FIXME what about the bias ?
    //feature_vector[feature_vector.size() - 1].value = the_problem.bias;

    feature_vectors_labels[index] = class_value;
    return;
}


LinearSvmModel model_from_model(struct model &model)
{

    Eigen::VectorXf w(model.nr_feature);
    for(int i=0; i < model.nr_feature; i+=1)
    {
        w(i) = model.w[i];
    }

    const float bias = model.bias;
    return LinearSvmModel(w, bias);
}


void LibLinearTraining::train()
{
    if(the_problem.x == NULL or the_problem.y == NULL)
    {
        throw std::invalid_argument("no data set before calling LibLinearTraining::train");
    }

    if(the_parameters.eps <= 0)
    {
        throw std::invalid_argument("the_parameters.eps should be  > 0");
    }

    if(the_parameters.C <= 0)
    {
        throw std::invalid_argument("the_parameters.C should be  > 0");
    }


    // check if previous model needs to be destroyed
    doppia::free_and_destroy_model(&model_p);

    model_p = ::train(&the_problem, &the_parameters);

    learned_svm_model = model_from_model(*model_p);

    return;
}


const LinearSvmModel &LibLinearTraining::get_model() const
{
    if(learned_svm_model.get_w().size() == 0)
    {
        throw std::runtime_error("Called LibLinearTraining::get_model before LibLinearTraining::train, no learned model available");
    }

    return learned_svm_model;
}


void LibLinearTraining::save_training_data(const std::string filename) const
{

    if(feature_vectors.empty())
    {
        throw std::runtime_error("Called LibLinearTraining::save_training_data before setting any training data");
    }

    std::ofstream training_file;

    training_file.open(filename.c_str());

    if(training_file.is_open() == false)
    {
        throw std::runtime_error("Failed to open training data recording file.");
    }

    // set number of digits used for floating points
    training_file.precision(5);

    for(size_t i=0; i < feature_vectors_labels.size(); i+= 1)
    {
        training_file << feature_vectors_labels[i] << " ";

        feature_vectors_t::const_reference feature_vector = feature_vectors[i];

        for(size_t c=0; c < (feature_vector.size() - 1); c+=1)
        { // size-1 since last element is just a -1:0 marker
            training_file << feature_vector[c].index << ":" << feature_vector[c].value << " ";
        } // end of "for each element in the feature vector"

        training_file << "\n"; // not std::endl since we do not want to force a flush (so things go faster)

    } // end of "for each feature vector"

    training_file.close();

    return;
}


void LibLinearTraining::save_model(const std::string filename) const
{

    if(model_p == NULL)
    {
        throw std::runtime_error("LibLinearTraining::save_model called before LibLinearTraining::train");
    }
    ::save_model(filename.c_str(), model_p);

    return;
}


/// slightly modified version from the one found inside liblinear's linear.cpp
void predict_values(const struct model *model_, const struct feature_node *x,
                    double *dec_values, int &label_index)
{
    int idx;
    int n;
    if(model_->bias>=0)
        n=model_->nr_feature+1;
    else
        n=model_->nr_feature;
    double *w=model_->w;
    int nr_class=model_->nr_class;
    int i;
    int nr_w;
    if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
        nr_w = 1;
    else
        nr_w = nr_class;

    const feature_node *lx=x;
    for(i=0;i<nr_w;i++)
        dec_values[i] = 0;
    for(; (idx=lx->index)!=-1; lx++)
    {
        // the dimension of testing data may exceed that of training
        if(idx<=n)
            for(i=0;i<nr_w;i++)
                dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
    }

    if(nr_class==2)
    {
        //return (dec_values[0]>0)?model_->label[0]:model_->label[1];
        label_index = (dec_values[0]>0)? 0: 1;
        return;
    }
    else
    {
        int dec_max_idx = 0;
        for(i=1;i<nr_class;i++)
        {
            if(dec_values[i] > dec_values[dec_max_idx])
                dec_max_idx = i;
        }
        //return model_->label[dec_max_idx];
        label_index = dec_max_idx;
        return;
    }
} // end of "predict"


/// predict the training data
void LibLinearTraining::predict()
{
    if(model_p == NULL)
    {
        throw std::runtime_error("LibLinearTraining::predict called before LibLinearTraining::train");
    }

    predicted_feature_vectors_labels.resize(feature_vectors_labels.size());
    decision_values.resize(predicted_feature_vectors_labels.size());

    //printf("model_p->nr_class == %i\n", model_p->nr_class);

#pragma omp parallel for
    for(size_t i=0; i < predicted_feature_vectors_labels.size(); i+=1)
    {
        std::vector<double> dec_values(model_p->nr_class);

        feature_vectors_t::const_reference feature_vector = feature_vectors[i];

        int label_index;
        predict_values(model_p, feature_vector.origin(),
                       dec_values.data(),
                       label_index);

        predicted_feature_vectors_labels[i] = model_p->label[label_index];
        decision_values[i] = dec_values[label_index];

    } // end of "for each feature

    return;
}

const std::vector<int> &LibLinearTraining::get_expected_labels() const
{
    return feature_vectors_labels;
}

const std::vector<int> &LibLinearTraining::get_predicted_labels() const
{
    return predicted_feature_vectors_labels;
}

/// this are the "raw" values that supported the predicted labels
const std::vector<double> &LibLinearTraining::get_decision_values() const
{
    return decision_values;
}




} // end of namespace doppia
