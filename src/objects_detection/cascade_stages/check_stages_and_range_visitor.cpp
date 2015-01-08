#include "check_stages_and_range_visitor.hpp"

#include "objects_detection/SoftCascadeOverIntegralChannelsModel.hpp"

#include "helpers/Log.hpp"

namespace doppia {


typedef SoftCascadeOverIntegralChannelsModel::plain_stage_t plain_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t fast_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stage_t fast_fractional_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::stump_stage_t stump_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::two_stumps_stage_t two_stumps_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::three_stumps_stage_t three_stumps_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::four_stumps_stage_t four_stumps_stage_t;


check_stages_and_range_visitor::check_stages_and_range_visitor(
        const ptrdiff_t scale_index_,
        const DetectorSearchRange &search_range_,
        const int shrunk_width_, const int shrunk_height_,
        const bool should_touch_borders_)
    :
      scale_index(scale_index_),
      search_range(search_range_),
      shrunk_width(shrunk_width_),
      shrunk_height(shrunk_height_),
      should_touch_borders(should_touch_borders_),
      min_observed_x(shrunk_width), max_observed_x(0),
      min_observed_y(shrunk_height), max_observed_y(0)
{
    log_debug() << boost::str(
                       boost::format(
                           "Starting check for scale %i, search range min (x,y) == (%i, %i), max (x,y) == (%i, %i), "
                           "image (width, height) == (%i, %i)\n")
                       % scale_index
                       % search_range.min_x % search_range.min_y
                       % search_range.max_x % search_range.max_y
                       % shrunk_width % shrunk_height);
    return;
}


template<>
bool check_stages_and_range_visitor::check_stage<fast_stage_t>
(const fast_stage_t &stage)
{
    const bool everything_is_fine =
            check_feature(stage.weak_classifier.level1_node.feature)
            and check_feature(stage.weak_classifier.level2_true_node.feature)
            and check_feature(stage.weak_classifier.level2_false_node.feature);

    return everything_is_fine;
}


template<>
bool check_stages_and_range_visitor::check_stage<stump_stage_t>
(const stump_stage_t &stage)
{
    const bool everything_is_fine = check_feature(stage.weak_classifier.feature);

    return everything_is_fine;
}


template<>
bool check_stages_and_range_visitor::check_stage< SoftCascadeOverIntegralChannelsStumpSetStage<3> >
(const SoftCascadeOverIntegralChannelsStumpSetStage<3> &stage)
{
    const bool everything_is_fine =
            check_feature(stage.stumps[0].feature)
            and check_feature(stage.stumps[1].feature)
            and check_feature(stage.stumps[2].feature);

    return everything_is_fine;
}


/// checks if a single feature is inside the expected boundaries
bool check_stages_and_range_visitor::check_feature(const IntegralChannelsFeature &feature)
{
    // to be really sure, we will check all four corners of the search range
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    // this code should match get_feature_value_* as defined inside integral_channels_detector.cu
    const int
            top_left_x = search_range.min_x + box.min_corner().x(),
            top_left_y = search_range.min_y + box.min_corner().y(),
            top_right_x = search_range.max_x + box.max_corner().x(),
            top_right_y = top_left_y,
            bottom_left_x = top_left_x,
            bottom_left_y = search_range.max_y + box.max_corner().y(),
            bottom_right_x = top_right_x,
            bottom_right_y = bottom_left_y;

    {
        min_observed_x = std::min(min_observed_x, std::min(top_left_x, top_right_x));
        max_observed_x = std::max(max_observed_x, std::max(top_right_x, top_left_x));

        min_observed_y = std::min(min_observed_y, std::min(top_left_y, bottom_left_y));
        max_observed_y = std::max(max_observed_y, std::max(top_left_y, bottom_left_y));
    }

    // checks are >= 0 because reaching zero is fine
    // checks are  <= max_x/y because the features are evaluated over integral images,
    // which are max_x/y + 1.

    const bool top_left_is_fine =
            (top_left_x >= 0) and (top_left_x <= shrunk_width)
            and (top_left_y >= 0) and (top_left_y <= shrunk_height);
    if(not top_left_is_fine)
    {
        log_error() << boost::str(
                           boost::format("Top left corner failed; (x,y) == (%i, %i)\n")
                           % top_left_x % top_left_y);
    }

    const bool top_right_is_fine =
            (top_right_x >= 0) and (top_right_x <= shrunk_width)
            and (top_right_y >= 0) and (top_right_y <= shrunk_height);
    if(not top_right_is_fine)
    {
        log_error() << boost::str(
                           boost::format("Top right corner failed; (x,y) == (%i, %i)\n")
                           % top_right_x % top_right_y);
    }

    const bool bottom_left_is_fine =
            (bottom_left_x >= 0) and (bottom_left_x <= shrunk_width)
            and (bottom_left_y >= 0) and (bottom_left_y <= shrunk_height);
    if(not bottom_left_is_fine)
    {
        log_error() << boost::str(
                           boost::format("Bottom left corner failed; (x,y) == (%i, %i)\n")
                           % bottom_left_x % bottom_left_y);
    }

    const bool bottom_right_is_fine =
            (bottom_right_x >= 0) and (bottom_right_x <= shrunk_width)
            and (bottom_right_y >= 0) and (bottom_right_y <= shrunk_height);
    if(not bottom_right_is_fine)
    {
        log_error() << boost::str(
                           boost::format("Bottom right corner failed; (x,y) == (%i, %i)\n")
                           % bottom_right_x % bottom_right_y);
    }


    const bool everything_is_fine =
            top_left_is_fine and top_right_is_fine
            and bottom_left_is_fine and bottom_right_is_fine;

    if(not everything_is_fine)
    {
        log_error() << boost::str(
                           boost::format(
                               "Feature box min (x,y) == (%i, %i), max (x,y) == (%i, %i)\n"
                               "shrunk width, height == %i, %i\n")
                           % box.min_corner().x()
                           % box.min_corner().y()
                           % box.max_corner().x()
                           % box.max_corner().y()
                           % shrunk_width
                           % shrunk_height );
    }

    return everything_is_fine;
}


std::ostream & check_stages_and_range_visitor::log_info()
{
    return  logging::log(logging::InfoMessage, "check_stages_and_range_visitor");
}


std::ostream & check_stages_and_range_visitor::log_debug()
{
    return  logging::log(logging::DebugMessage, "check_stages_and_range_visitor");
}


std::ostream & check_stages_and_range_visitor::log_warning()
{
    return  logging::log(logging::WarningMessage, "check_stages_and_range_visitor");
}


std::ostream & check_stages_and_range_visitor::log_error()
{
    return  logging::log(logging::ErrorMessage, "check_stages_and_range_visitor");
}




} // end of namespace doppia
