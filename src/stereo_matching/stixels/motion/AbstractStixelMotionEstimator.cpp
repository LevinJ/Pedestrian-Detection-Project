#include "AbstractStixelMotionEstimator.hpp"

#include <stdexcept>

#include <cstdio>

namespace doppia {

AbstractStixelMotionEstimator::AbstractStixelMotionEstimator()
{
    frame_a_is_current_frame = false;
    previous_stixels_p = NULL;
    current_stixels_p = NULL;
    return;
}

AbstractStixelMotionEstimator::~AbstractStixelMotionEstimator()
{
    // nothing to do here
    return;
}

void AbstractStixelMotionEstimator::set_new_rectified_image(input_image_const_view_t &input)
{
    frame_a_is_current_frame = not frame_a_is_current_frame;

    boost::gil::rgb8_view_t current_view;

    if(frame_a_is_current_frame)
    {
        previous_image_view = boost::gil::const_view(image_b);

        // lazy allocation
        image_a.recreate(input.dimensions());
        current_view = boost::gil::view(image_a);        
    }
    else
    {
        previous_image_view = boost::gil::const_view(image_a);

        // lazy allocation
        image_b.recreate(input.dimensions());
        current_view = boost::gil::view(image_b);        
    }

    // copy the content
    boost::gil::copy_pixels(input, current_view);

    current_image_view = current_view; // non-const to const view

    return;
}

void AbstractStixelMotionEstimator::set_estimated_stixels(const stixels_t &input_stixels)
{
    if(frame_a_is_current_frame)
    {
        previous_stixels_p = &stixels_b;
        current_stixels_p = &stixels_a;
    }
    else
    {
        previous_stixels_p = &stixels_a;
        current_stixels_p = &stixels_b;
    }

    *current_stixels_p = input_stixels;

    if(stixels_motion.empty())
    {
        // by default there is 0 motion everywhere
        stixels_motion.clear();
        stixels_motion.resize(current_stixels_p->size(), 0);

        for(size_t i=0; i < stixels_motion.size(); i+=1)
        {
            stixels_motion[i] = i;
        }
    }

    return;
}

const AbstractStixelMotionEstimator::input_image_const_view_t& AbstractStixelMotionEstimator::get_current_image_view() const
{
    return current_image_view;
}

const AbstractStixelMotionEstimator::input_image_const_view_t& AbstractStixelMotionEstimator::get_previous_image_view() const
{
    return previous_image_view;
}

const stixels_t& AbstractStixelMotionEstimator::get_current_stixels() const
{
    return *current_stixels_p;
}

const stixels_t& AbstractStixelMotionEstimator::get_previous_stixels() const
{
    return *previous_stixels_p;
}

const AbstractStixelMotionEstimator::input_image_view_t AbstractStixelMotionEstimator::get_stixel_tracks_image_view() const
{
    return stixel_tracks_view;
}

const AbstractStixelMotionEstimator::stixels_motion_t& AbstractStixelMotionEstimator::get_stixels_motion() const
{
    if(stixels_motion.empty())
    {
        throw std::runtime_error("Called AbstractStixelMotionEstimator::get_stixels_motion "
                                 "but stixels_motion vector is empty");
    }

    return stixels_motion;
}

const AbstractStixelMotionEstimator::motion_cost_matrix_t& AbstractStixelMotionEstimator::get_matching_cost_matrix() const
{
    return matching_cost_matrix;
}

const AbstractStixelMotionEstimator::matrix_3d_float_t& AbstractStixelMotionEstimator::get_c_cost_matrix() const
{
    return c_cost_matrix;
}

const AbstractStixelMotionEstimator::motion_cost_matrix_t& AbstractStixelMotionEstimator::get_motion_cost_matrix() const
{
    return motion_cost_matrix;
}

const AbstractStixelMotionEstimator::dynamic_matrix_boolean_t& AbstractStixelMotionEstimator::get_motion_cost_assignment_matrix() const
{
    return motion_cost_assignment_matrix;
}

const AbstractStixelMotionEstimator::motion_cost_matrix_t& AbstractStixelMotionEstimator::get_dynamic_programming_cost_matrix() const
{
    return M_cost_dp;
}

const AbstractStixelMotionEstimator::motion_cost_matrix_t& AbstractStixelMotionEstimator::get_visual_motion_cost_matrix() const
{
    return visual_motion_cost_matrix;
}

const Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 >& AbstractStixelMotionEstimator::get_visual_motion_cost_column_maxima_indices() const
{
    return indices_columnwise_minimum_visual_cost;
}

const Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >& AbstractStixelMotionEstimator::get_motion_cost_column_maxima_indices() const
{
    return indices_columnwise_minimum_cost;
}

const Eigen::MatrixXi& AbstractStixelMotionEstimator::get_stixels_one_to_many_assignment_map() const
{
    return stixels_one_to_many_assignment_map;
}

} // namespace doppia
