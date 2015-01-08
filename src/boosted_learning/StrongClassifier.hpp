#ifndef _StrongClassifier_H
#define _StrongClassifier_H

#include "LabeledData.hpp"
#include "TrainingData.hpp"
#include "WeakClassifier.hpp"


namespace boosted_learning
{

class StrongClassifier
{
public:

    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef bootstrapping::integral_channels_const_view_t integral_channels_const_view_t;
    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    StrongClassifier(const std::vector<boost::shared_ptr<WeakClassifier> > &weak_classifiers);

    void convert_to_soft_cascade(const LabeledData::shared_ptr data, const float detection_rate);
    void write_classifier(const std::string filename,
                         const std::string trained_dataset_name,
                         const point_t model_window, const rectangle_t object_window);

    float compute_classification_score(const integral_channels_t &integral_image) const;

    void classify(const TrainingData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade=true) const;
    void classify(const LabeledData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade=true) const;

protected:
    WeakClassifierType _weak_classifier_type;
    std::vector< boost::shared_ptr<WeakClassifier> > _weak_classifiers;

};

} // end of namespace boosted_learning

#endif// StrongClassifier_H
