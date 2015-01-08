#ifndef LINEARSVMMODEL_HPP
#define LINEARSVMMODEL_HPP

#include <Eigen/Core>

#include <string>
#include <fstream>

// forward declaration
namespace doppia_protobuf {
  class DetectorModel;
}

namespace doppia {

class LinearSvmModel
{
public:
    LinearSvmModel();

    /// this constructor will copy the protobuf data into a more efficient data structure
    LinearSvmModel(const doppia_protobuf::DetectorModel &model);


    LinearSvmModel(const Eigen::VectorXf &w, const float bias);

    /// will read the data from a liblinear model file
    LinearSvmModel(const std::string &model_filename);
    ~LinearSvmModel();

    float get_bias() const;
    const Eigen::VectorXf &get_w() const;

    float compute_score(const Eigen::VectorXf &feature_vector) const;

protected:

    float bias;
    Eigen::VectorXf w;

    void parse_libsvm_model_file(std::ifstream &model_file);
};

} // end of namespace doppia




#endif // LINEARSVMMODEL_HPP
