#include "compute_cascade_window_size_visitor.hpp"

namespace doppia {


template<>
void compute_cascade_window_size_visitor::update_window_size
<SoftCascadeOverIntegralChannelsModel::fast_stage_t>
(const SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage, detection_window_size_t &window_size) const
{
    update_window_size(stage.weak_classifier.level1_node.feature, window_size);
    update_window_size(stage.weak_classifier.level2_true_node.feature, window_size);
    update_window_size(stage.weak_classifier.level2_false_node.feature, window_size);
    return;
}



void compute_cascade_window_size_visitor::update_window_size(
        const IntegralChannelsFeature &feature,
        detection_window_size_t &window_size) const
{
    if((feature.box.min_corner().x() < 0)
       or(feature.box.min_corner().y() < 0))
    {
        throw std::runtime_error("compute_cascade_window_size_visitor expects all features to have positive coordinates");
    }


    window_size.x( std::max<int>(feature.box.max_corner().x(), window_size.x()) );
    window_size.y( std::max<int>(feature.box.max_corner().y(), window_size.y()) );

    return;
}

} // end of namespace doppia
