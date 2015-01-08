#include "ModelIO.hpp"

#include "applications/bootstrapping_lib/IntegralChannelsComputer.hpp"


#include <iostream>
#include <fstream>
#include <stdexcept>
#include <boost/make_shared.hpp>

namespace boosted_learning {

using namespace doppia_protobuf;


ModelIO::ModelIO(const bool silent_mode): _silent_mode(silent_mode)
{
    // set the shrinking factor in case we will write this model
    SoftCascadeOverIntegralChannelsModel *model_p =  _model.mutable_soft_cascade_model();
    model_p->set_shrinking_factor(bootstrapping::integral_channels_computer_t::get_shrinking_factor());

    return;
}


ModelIO::~ModelIO()
{
    //google::protobuf::ShutdownProtobufLibrary();
    return;
}


void ModelIO::read_model(const std::string filename)
{
    fstream input(filename.c_str(), ios::in | std::ios::binary);

    if( input.is_open() == false)
    {
        printf("Failed to open model file %s\n", filename.c_str());
        throw std::runtime_error("Failed to open model file");
    }

    if (!_model.ParseFromIstream(&input))
    {
        throw std::runtime_error("Failed to read Detector File.");
    }

    return;
}







DecisionTreeNode ModelIO::read_decision_tree_node(const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode &node)
{
    DecisionStump stump = read_decision_stump(node.decision_stump());
    int id = node.id();

    return DecisionTreeNode(stump._feature, stump._feature_index, stump._threshold, stump._alpha, id);
}
DecisionStump ModelIO::read_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump)
{
    int ch = stump.feature().channel_index();
    int x = stump.feature().box().min_corner().x();
    int y = stump.feature().box().min_corner().y();
    int w = stump.feature().box().max_corner().x() - stump.feature().box().min_corner().x();
    int h = stump.feature().box().max_corner().y() - stump.feature().box().min_corner().y();

    const Feature feature(x, y, w, h, ch);
    int feature_index = -1;
    float threshold = (float) stump.feature_threshold();
    int alpha = stump.larger_than_threshold()? -1 : 1;

    return DecisionStump(feature, feature_index, threshold, alpha);
}
void ModelIO::read_stump_set(const doppia_protobuf::IntegralChannelStumpSet &stumpset, boost::shared_ptr<WeakClassifierStumpSet> wc_stumpset_p)
{
    int noNodes = stumpset.nodes_size();
    int noBetas = stumpset.weights_size();
    wc_stumpset_p->_stumps.resize(noNodes);
    wc_stumpset_p->_betas.resize(noBetas);

    assert(pow(2.0, noNodes)==noBetas);

    for (int i = 0; i < noNodes; ++i)
    {
        wc_stumpset_p->_stumps[i] = read_decision_stump(stumpset.nodes(i));
    }
    for (int i = 0; i < noBetas; ++i)
    {
        wc_stumpset_p->_betas[i] = stumpset.weights(i);
    }

    wc_stumpset_p->_tree_type = STUMP_SET;
    wc_stumpset_p->_size = noNodes;

    return;
}
void ModelIO::read_decision_tree(const doppia_protobuf::IntegralChannelBinaryDecisionTree &tree, boost::shared_ptr<WeakClassifierDecisionTree> wc_decisiontree_p)
{
    std::vector<DecisionTreeNode> nodes;

    int noNodes = tree.nodes_size();
    nodes.resize(noNodes);

    for (int i = 0; i < noNodes; ++i)
    {
        //read node
        DecisionTreeNode node = read_decision_tree_node(tree.nodes(i));
        assert(i == tree.nodes(i).id());
        node._id = tree.nodes(i).id();
        nodes.push_back(node);
    }

    wc_decisiontree_p->set_beta(1);
    wc_decisiontree_p->_decision_nodes = nodes;
    wc_decisiontree_p->_depth = (int)ceil(log(nodes.size())/log(2));
    wc_decisiontree_p->_tree_type = DECISION_TREE;

    return;
}
void ModelIO::read_classifier_stage(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage, boost::shared_ptr<WeakClassifier> weak_classifier_p)
{

    weak_classifier_p->_cascade_threshold = (stage.cascade_threshold());

    if (stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_Level2DecisionTree
             || stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_LevelNDecisionTree)
    {

        boost::shared_ptr<WeakClassifierDecisionTree> wc_dt = boost::static_pointer_cast<WeakClassifierDecisionTree>(weak_classifier_p);
        wc_dt->set_beta(stage.weight());
        read_decision_tree(stage.level2_decision_tree(), wc_dt);
    }
    else if (stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_StumpSet
             || stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_Stumps)
    {
        boost::shared_ptr<WeakClassifierStumpSet> wc_ss = boost::static_pointer_cast<WeakClassifierStumpSet>(weak_classifier_p);
        read_stump_set(stage.stump_set(), wc_ss);
    }
    else
    {
        throw std::invalid_argument("Received an unknown stage.feature_type");
    }

    return;
}
void ModelIO::read_full_cascade(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &sc, std::vector<boost::shared_ptr<WeakClassifier> > & wc_stages)
{
    //typedef doppia_protobuf::Sof
    wc_stages.resize(sc.stages_size());

    if (!_silent_mode)
    {
        std::cout << "Cascade contains " << sc.stages_size() << " stages\n";
    }

    for (int i = 0; i < sc.stages_size(); ++i)
    {
        read_classifier_stage(sc.stages(i), wc_stages[i]);
    }

    return;
}
StrongClassifier ModelIO::read()
{
    std::vector<boost::shared_ptr<WeakClassifier> > out;

    if (!_silent_mode)
    {
        std::cout << " Reading Detector: " << _model.detector_name() << std::endl;
        std::cout << " Trained on: " << _model.training_dataset_name() << std::endl;
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_LinearSvm)
    {
        std::cout << " Type of the Detector: Linear SVM\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels)
    {
        std::cout << " Type of the Detector: SoftCascade over Integral Channels\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_HoughForest)
    {
        std::cout << " Type of the Detector: HoughForest\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels)
    {
        read_full_cascade(_model.soft_cascade_model(), out);
    }

    else
    {
        throw runtime_error("This type of detector has not been implemented yet");
    }

    StrongClassifier ret(out);

    return ret;
}


void ModelIO::print()
{
    _model.PrintDebugString();
    return;
}



const std::string ModelIO::get_model_training_dataset_name()
{
    if(_model.has_training_dataset_name() == false)
    {
        throw std::runtime_error("_model.has_training_dataset_name() == false");
    }
    return _model.training_dataset_name();
}
const TrainingData::point_t ModelIO::get_model_window_size()
{
    if(_model.has_model_window_size() == false)
    {
        throw std::runtime_error("_model.has_model_window_size() == false");
    }

    return TrainingData::point_t(_model.model_window_size().x(),
                                 _model.model_window_size().y());
}
const TrainingData::rectangle_t ModelIO::get_object_window()
{
    if(_model.has_object_window() == false)
    {
        throw std::runtime_error("_model.has_object_window() == false");
    }

    TrainingData::rectangle_t object_window;
    object_window.min_corner().x( _model.object_window().min_corner().x() );
    object_window.min_corner().y( _model.object_window().min_corner().y() );
    object_window.max_corner().x( _model.object_window().max_corner().x() );
    object_window.max_corner().y( _model.object_window().max_corner().y() );

    return object_window;
}
void ModelIO::init_write(const std::string dataset_name,
                        const DetectorModel::DetectorTypes type,
                        const std::string detector_name,
                        const point_t model_window,
                        const rectangle_t object_window,
                        float occlusion_level, ::doppia_protobuf::DetectorModel::OcclusionTypes occlusion_type)
{

    doppia_protobuf::Point2d *model_window_new = _model.mutable_model_window_size();
    model_window_new->set_x(model_window.x());
    model_window_new->set_y(model_window.y());
    _model.set_occlusion_level(occlusion_level);
    _model.set_occlusion_type(occlusion_type);

    doppia_protobuf::Box *b = _model.mutable_object_window();
    b->mutable_min_corner()->set_x(object_window.min_corner().x());
    b->mutable_min_corner()->set_y(object_window.min_corner().y());
    b->mutable_max_corner()->set_x(object_window.max_corner().x());
    b->mutable_max_corner()->set_y(object_window.max_corner().y());

    _model.set_training_dataset_name(dataset_name.c_str());
    _model.set_detector_type(type);
    _model.set_detector_name(detector_name);

    return;
}




void ModelIO::set_decision_tree_node(doppia_protobuf::IntegralChannelBinaryDecisionTreeNode *node, const DecisionTreeNode dec_node)
{
    doppia_protobuf::IntegralChannelDecisionStump *stump_p = node->mutable_decision_stump();
    set_decision_stump(stump_p, dec_node);

    if(dec_node._id==1)
    {
        stump_p->set_larger_than_threshold(false);
    }

    node->set_id(dec_node._id);
    node->set_parent_id((dec_node._id == 1) ? 1 : (dec_node._id / 2));
    node->set_parent_value(dec_node.is_left());

    return;
}
void ModelIO::set_decision_stump(doppia_protobuf::IntegralChannelDecisionStump *stump, const DecisionStump dec_stump)
{
    stump->set_feature_threshold(dec_stump._threshold);
    if (dec_stump._alpha == -1)
    {
        stump->set_larger_than_threshold(true);
    }
    else if (dec_stump._alpha == 1)
    {
        stump->set_larger_than_threshold(false);
    }

    IntegralChannelsFeature *feat = stump->mutable_feature();
    feat->set_channel_index(dec_stump._feature.channel);
    Box *b = feat->mutable_box();
    b->mutable_min_corner()->set_x(dec_stump._feature.x);
    b->mutable_min_corner()->set_y(dec_stump._feature.y);

    b->mutable_max_corner()->set_x(dec_stump._feature.x + dec_stump._feature.width);
    b->mutable_max_corner()->set_y(dec_stump._feature.y + dec_stump._feature.height);

    return;
}
void ModelIO::set_stump_set(doppia_protobuf::IntegralChannelStumpSet *stumpset, boost::shared_ptr<WeakClassifierStumpSet> stump_set_p)
{
    for (size_t i = 0; i< stump_set_p->_stumps.size(); ++i){
        set_decision_stump(stumpset->add_nodes(), stump_set_p->_stumps[i]);

    }

    for( size_t i = 0; i< stump_set_p->_betas.size();++i ){
        assert(stumpset->nodes(i).larger_than_threshold() == false);
        stumpset->add_weights(stump_set_p->_betas[i]);
    }

}
void ModelIO::set_decision_tree(doppia_protobuf::IntegralChannelBinaryDecisionTree *tree, boost::shared_ptr<WeakClassifierDecisionTree> dec_tree_p)
{
    int num_nodes = dec_tree_p->_decision_nodes.size();
    for(int i=0; i<num_nodes; i++)
    {
        IntegralChannelBinaryDecisionTreeNode *n = tree->add_nodes();
        set_decision_tree_node(n, dec_tree_p->_decision_nodes[i]);
    }

    return;
}


void ModelIO::add_classifier_stage(const boost::shared_ptr<WeakClassifier> wc)
{
    SoftCascadeOverIntegralChannelsModel *model_p =  _model.mutable_soft_cascade_model();
    doppia_protobuf::SoftCascadeOverIntegralChannelsStage *stage = model_p->add_stages();

    if (wc->_tree_type == STUMP_SET)
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_StumpSet);
        set_stump_set(stage->mutable_stump_set(), boost::static_pointer_cast<WeakClassifierStumpSet>(wc));
    }
    else if (wc->_tree_type == DECISION_TREE)
    {
        boost::shared_ptr<WeakClassifierDecisionTree> wc_dt = boost::static_pointer_cast<WeakClassifierDecisionTree>(wc);
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_Level2DecisionTree);
        set_decision_tree(stage->mutable_level2_decision_tree(), wc_dt);
        stage->set_weight(wc_dt->get_beta());
    }
    else
    {
        throw std::runtime_error("Unknown tree type.");
    }

    stage->set_cascade_threshold(wc->_cascade_threshold);

    if (false and (not _silent_mode)) // verbosity_level = 2
    {
        stage->PrintDebugString();
    }

    return;
}
void ModelIO::write(const std::string file_name)
{

    if(file_name.empty())
    {
        throw std::invalid_argument("ModelIO::write requires a non-empty file name");
    }

    fstream output(file_name.c_str(), ios::out | std::ios::binary);

    if(output.is_open() == false)
    {
        printf("ModelIO failed to open file %s for writing\n", file_name.c_str());
        throw std::invalid_argument("ModelIO::write failed to create the output file.");
    }

    _model.CheckInitialized(); // check that all messages are properly set
   // _model.PrintDebugString();
    if (!_model.SerializeToOstream(&output))
    {
        throw std::runtime_error("Failed to write Detector File.");
    }

    output.close();
    return;
}









} // end of namespace boosted_learning
