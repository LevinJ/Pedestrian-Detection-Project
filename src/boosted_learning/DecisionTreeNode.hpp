#ifndef DECISIONTREENODE_HPP
#define DECISIONTREENODE_HPP

#include "DecisionStump.hpp"

namespace boosted_learning {


class DecisionTreeNode : public DecisionStump
{
public:
    typedef boost::shared_ptr<DecisionTreeNode> decision_tree_node_p;

    DecisionTreeNode();
    DecisionTreeNode(const Feature &feature, const size_t feature_index, const int threshold, const int alpha, const int id);
     
    int _id;
    bool _valid;

    // setters
    void set_id(const int id);
    void set_invalid();

    // getters
    int get_id() const;
    bool is_invalid() const;
    int get_parent_id() const;
    bool is_left() const;

    void print(std::ostream & log_func, const int depth = 0);
};


} // end of namespace boosted_learning

#endif // DECISIONTREENODE_HPP
