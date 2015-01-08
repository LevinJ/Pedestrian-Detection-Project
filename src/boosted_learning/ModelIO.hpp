#ifndef ModelIO_HPP
#define ModelIO_HPP

#include "WeakLearner.hpp"
#include "StrongClassifier.hpp"

#include "objects_detection/detector_model.pb.h"

#include <string>

namespace boosted_learning
{

class ModelIO
{
public:

    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;
    typedef ::doppia_protobuf::DetectorModel::OcclusionTypes occlusion_t;

    ModelIO(const bool silent_mode = true);
    ~ModelIO();
    void init_write(const std::string datasetName,
                   const doppia_protobuf::DetectorModel::DetectorTypes type,
                   const std::string detectorName,
                   const point_t modelWindow,
                   const rectangle_t objectWindow,float occlusionLevel=0 ,
                   occlusion_t occlusionType = ::doppia_protobuf::DetectorModel::BottomOcclusion);
    void write(const std::string fileName);
    void read_model(const std::string filename);
    void print();


    void read_full_cascade(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &sc, std::vector<boost::shared_ptr<WeakClassifier> > &wc_stages);
    void read_classifier_stage(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage, boost::shared_ptr<WeakClassifier> weak_classifier_p);
    void read_decision_tree(const doppia_protobuf::IntegralChannelBinaryDecisionTree &tree, boost::shared_ptr<WeakClassifierDecisionTree> wc_decisiontree_p);
    void read_stump_set(const doppia_protobuf::IntegralChannelStumpSet &stumpset, boost::shared_ptr<WeakClassifierStumpSet> wc_stumpset_p);
    DecisionStump read_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump);
    DecisionTreeNode read_decision_tree_node(const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode &node);
    StrongClassifier read();



    const std::string get_model_training_dataset_name();
    const TrainingData::point_t get_model_window_size();
    const TrainingData::rectangle_t get_object_window();

    void set_decision_tree_node(doppia_protobuf::IntegralChannelBinaryDecisionTreeNode *node, const DecisionTreeNode dec_node);
    void set_decision_stump(doppia_protobuf::IntegralChannelDecisionStump *stump, const DecisionStump dec_stump);
    void set_decision_tree(doppia_protobuf::IntegralChannelBinaryDecisionTree *tree, const boost::shared_ptr<WeakClassifierDecisionTree> dec_tree_p);
    void set_stump_set(doppia_protobuf::IntegralChannelStumpSet *stumpset, const boost::shared_ptr<WeakClassifierStumpSet> stump_set_p);
    void add_classifier_stage(const boost::shared_ptr<WeakClassifier> wc);



protected:
    doppia_protobuf::DetectorModel _model;
    bool _silent_mode;
};

} // end of namespace boosted_learning

#endif // ModelIO_HPP
