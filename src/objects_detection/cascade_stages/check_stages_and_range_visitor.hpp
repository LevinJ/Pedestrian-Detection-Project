#ifndef DOPPIA_CHECK_STAGES_AND_RANGE_VISITOR_HPP
#define DOPPIA_CHECK_STAGES_AND_RANGE_VISITOR_HPP

#include "objects_detection/DetectorSearchRange.hpp"

#include <boost/variant/static_visitor.hpp>

#include <boost/format.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>


namespace doppia {

/// Helper visitor class used to verify if the stages lie inside an expected range
class check_stages_and_range_visitor : public boost::static_visitor<bool>
{
protected:
    const ptrdiff_t scale_index;
    const DetectorSearchRange &search_range;
    const int shrunk_width, shrunk_height;
    const bool should_touch_borders;

    int min_observed_x, max_observed_x, min_observed_y, max_observed_y;
public:

    /// @param should_touch_borders indicates that we will not only check that the model does not overflow
    /// but that we will also make sure that it does touches all the borders
    /// (which is the expected behaviour for all models except occlusion ones)
    check_stages_and_range_visitor(
            const ptrdiff_t scale_index,
            const DetectorSearchRange &search_range,
            const int shrunk_width, const int shrunk_height,
            const bool should_touch_borders = false);

    template<typename StageType>
    bool operator()(const std::vector<StageType> &stages);

protected:

    /// checks if a single stage is inside the expected boundaries
    template<typename StageType>
    bool check_stage(const StageType &stage);

    /// checks if a single feature is inside the expected boundaries
    bool check_feature(const IntegralChannelsFeature &feature);


    std::ostream & log_info();
    std::ostream & log_debug();
    std::ostream & log_warning();
    std::ostream & log_error();

}; // end of visitor class check_stages_and_range_visitor


// Templated implementations need to be on the header
template<typename StageType>
bool check_stages_and_range_visitor::operator()(const std::vector<StageType> &stages)
{
    bool everything_is_fine = true;

    const int
            range_width = search_range.max_x - search_range.min_x,
            range_height = search_range.max_y - search_range.min_y;
    if((scale_index >= 0) and ((range_width <= 0) or (range_height <=0)) )
    {
        // if scale_index -1 will do a check on empty range
        log_debug() << "check_stages_and_range_visitor is skipping an empty search range" << std::endl;
        return everything_is_fine;
    }

    if(stages.empty())
    {
        everything_is_fine = false;
        log_error() << "check_stages_and_range_visitor received an empty model. That is not good..." << std::endl;
        return everything_is_fine;
    }

    for(size_t i=0; i < stages.size(); i+=1)
    {
        const StageType &stage = stages[i];
        everything_is_fine &= check_stage(stage);
        if(not everything_is_fine)
        {
            log_error() << "Failed safety check at stage " << i << std::endl;
            break;
        }
    } // end of "for each stage"


    // check if we touch the borders
    const bool
            touches_left_side = (min_observed_x == search_range.min_x),
            touches_top_side = (min_observed_y == search_range.min_y),
            touches_right_side = (max_observed_x == shrunk_width),
            touches_bottom_side = (max_observed_y == shrunk_height);

    const bool touches_the_borders =
            touches_left_side and touches_top_side
            and touches_right_side and touches_bottom_side;

    if(should_touch_borders and (not touches_the_borders))
    {
        log_error() << "Failed safety check because features do not touch the borders (but they should)" << std::endl;
        everything_is_fine = false;
    }

    const bool print_debug_information = true;
    if(print_debug_information)
    {
        const bool touches_no_border =
                (not touches_left_side) and (not touches_top_side)
                and (not touches_right_side) and (not touches_bottom_side);

        std::string touches;
        if(touches_the_borders)
        {
            touches = "touches all borders";
        }
        else if(touches_no_border)
        {
            touches = "touches no border";
        }
        else
        {
            //touches += (should_touch_borders and (not touches_the_borders))? "only touches " : "touches ";
            touches += "only touches ";
            touches += (touches_left_side)? "left, ": "";
            touches += (touches_top_side)? "top, ": "";
            touches += (touches_right_side)? "right, ": "";
            touches += (touches_bottom_side)? "bottom ": "";
            touches += "border(s)";
        }

        std::string fine;
        if(not everything_is_fine)
        {
            fine = "not ";
        }
        else if(everything_is_fine and (not touches_the_borders))
        {
            fine = "almost ";
        }
        else
        {
            fine = "";
        }

        std::ostream *out_stream_p = &log_debug();
        if((not everything_is_fine) or (not touches_the_borders))
        {
            out_stream_p = &log_warning();
        }

        (*out_stream_p) << boost::str(
                               boost::format(
                                   "Scale %i is %sfine, min (x, y) == (%i, %i), max (x,y) == (%i, %i), "
                                   "width, height == (%i, %i), (%s)\n")
                               % scale_index
                               % fine
                               % min_observed_x % min_observed_y
                               % max_observed_x % max_observed_y
                               % shrunk_width % shrunk_height
                               % touches );

        static bool first_not_touches_the_border = true;
        if(first_not_touches_the_border and (not touches_the_borders))
        {
            log_warning() << "Not touching the borders indicates that the model is suboptimal regarding detection quality "
                             "(was trained with doppia v1, instead of v2)" << std::endl;
            first_not_touches_the_border = false;
        }

    } // end of "if print debug information"


    return everything_is_fine;
}


template<typename StageType>
bool check_stages_and_range_visitor::check_stage(const StageType &)
{
    log_error() << "StageType == " << typeid(StageType).name() << std::endl;
    throw std::runtime_error("check_stages_and_range_visitor::check_stage "
                             "received an unknown (not yet implemented?) stage type");
    return false;
}

// prototypes declarations ---
template<>
bool check_stages_and_range_visitor::check_stage
<SoftCascadeOverIntegralChannelsModel::fast_stage_t>
(const SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage);

template<>
bool check_stages_and_range_visitor::check_stage
<SoftCascadeOverIntegralChannelsModel::stump_stage_t>
(const SoftCascadeOverIntegralChannelsModel::stump_stage_t &stage);


template<>
bool check_stages_and_range_visitor::check_stage
<SoftCascadeOverIntegralChannelsModel::three_stumps_stage_t>
(const SoftCascadeOverIntegralChannelsModel::three_stumps_stage_t &stage);


} // end of namespace doppia

#endif // DOPPIA_CHECK_STAGES_AND_RANGE_VISITOR_HPP
