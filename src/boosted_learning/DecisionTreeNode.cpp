#include "DecisionTreeNode.hpp"

namespace boosted_learning
{


DecisionTreeNode::DecisionTreeNode():
    DecisionStump::DecisionStump(Feature(-1,-1,-1,-1,-1), -1, 0, 1), _id(0), _valid(false)
{
    return;
}


DecisionTreeNode::DecisionTreeNode(const Feature &feature, const size_t feature_index, const int threshold, const int alpha, const int id):
    DecisionStump::DecisionStump(feature, feature_index, threshold, alpha), _id(id), _valid(true)
{
    // nothing to do here
    return;
}


// setters
void DecisionTreeNode::set_id(int id)
{
    _id = id;
    return;
}


void DecisionTreeNode::set_invalid()
{
    _valid = false;
    return;
}


//getters
int DecisionTreeNode::get_id() const
{
    return _id;
}


bool DecisionTreeNode::is_invalid() const
{
    return _valid;
}


int DecisionTreeNode::get_parent_id() const
{
    return _id/2;
}


bool DecisionTreeNode::is_left() const
{
    return _id==1 || (_id%2)==0;
}


void DecisionTreeNode::print(std::ostream & log_func, const int depth)
{
    log_func << std::endl;
    log_func << " --------------level" << depth << std::endl;
    log_func << _feature.x << "\t" << _feature.y << "\t" << _feature.width << "\t" << _feature.height << "\t" << _feature.channel << "\n";
    log_func << _threshold << std::endl;
    log_func << _alpha << std::endl;
    log_func << " ----------------------------" <<  std::endl;
    log_func << std::endl;

    return;
}

}
