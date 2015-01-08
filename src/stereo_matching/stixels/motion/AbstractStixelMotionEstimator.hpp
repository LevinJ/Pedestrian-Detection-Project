#ifndef DOPPIA_ABSTRACTSTIXELMOTIONESTIMATOR_HPP
#define DOPPIA_ABSTRACTSTIXELMOTIONESTIMATOR_HPP

#include "stereo_matching/stixels/Stixel.hpp"

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/multi_array.hpp>

#include <Eigen/Core>

#include <vector>

namespace doppia {

class AbstractStixelMotionEstimator
{

public:

    AbstractStixelMotionEstimator();
    virtual ~AbstractStixelMotionEstimator();

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8_view_t input_image_view_t;
    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    typedef std::vector<int> stixels_motion_t;
    typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic  > dynamic_matrix_boolean_t;

    typedef Eigen::MatrixXf motion_cost_matrix_t;
    typedef boost::multi_array< float, 3 > matrix_3d_float_t;
    typedef boost::multi_array< Eigen::Vector3f, 2 > matrix_of_3d_vectors_t;

    /// set the new image at time t,
    /// motion will be estimated between time t-1 and time t.
    virtual void set_new_rectified_image(input_image_const_view_t& input);

    /// set the estimated stixels corresponding to the current rectified image
    virtual void set_estimated_stixels(const stixels_t& input_stixels);

    /// No extra setters should be written !
    virtual const input_image_const_view_t& get_previous_image_view() const;
    virtual const input_image_const_view_t& get_current_image_view() const;

    /// No extra setters should be written !
    virtual const stixels_t& get_previous_stixels() const;
    virtual const stixels_t& get_current_stixels() const;

    virtual void compute() = 0;

    virtual const input_image_view_t get_stixel_tracks_image_view() const;
    virtual void update_stixel_tracks_image() = 0;
    virtual void reset_stixel_tracks_image() = 0;

    /// Will return a vector as long as the input stixels (as many as columns in the image),
    /// and each value indicates the index of the stixel in the _previous_ frame
    /// Negative value indicate non-matching ("new") stixels.
    virtual const stixels_motion_t& get_stixels_motion() const;

    virtual const motion_cost_matrix_t& get_motion_cost_matrix() const;
    virtual const dynamic_matrix_boolean_t& get_motion_cost_assignment_matrix() const;
    virtual const motion_cost_matrix_t& get_dynamic_programming_cost_matrix() const;
    virtual const motion_cost_matrix_t& get_visual_motion_cost_matrix() const;

    virtual const motion_cost_matrix_t& get_matching_cost_matrix() const;

    virtual const matrix_3d_float_t& get_c_cost_matrix() const;

    virtual const Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 >& get_visual_motion_cost_column_maxima_indices() const;
    virtual const Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >& get_motion_cost_column_maxima_indices() const;

    virtual const Eigen::MatrixXi& get_stixels_one_to_many_assignment_map() const;



protected:

    virtual void compute_motion_cost_matrix() = 0;
    virtual void compute_motion() = 0;

    bool frame_a_is_current_frame;

    input_image_t image_a, image_b;
    input_image_const_view_t previous_image_view, current_image_view;    

    stixels_t stixels_a;
    stixels_t stixels_b;

    stixels_t* previous_stixels_p;
    stixels_t* current_stixels_p;

    stixels_motion_t stixels_motion;

    motion_cost_matrix_t matching_cost_matrix;

    motion_cost_matrix_t motion_cost_matrix;
    dynamic_matrix_boolean_t motion_cost_assignment_matrix; // boolean
    motion_cost_matrix_t M_cost_dp; // Recursively filled dynamic programming matrix

    motion_cost_matrix_t visual_motion_cost_matrix; // For visualization purposes

    Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > indices_columnwise_minimum_cost;
    Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 > indices_columnwise_minimum_visual_cost;

    Eigen::MatrixXi stixels_one_to_many_assignment_map;

    /// Stores the "difference" between neighbouring stixels
    /// The difference between stixels[i] and stixels[i + 1] is stored in neighbour_differences(i)
    /// Hence, its size should be 1 less than the total number of stixels
    Eigen::Matrix< float, Eigen::Dynamic, 1 > neighbour_differences;

    /// Stores the c_cost values obtained in the first pass of the dynamic programming
    /// They are needed in the second pass.
    matrix_3d_float_t c_cost_matrix;

    /// For visualization of stixel tracks
    input_image_t stixel_tracks_image;
    input_image_view_t stixel_tracks_view;



};

} // namespace doppia

#endif // DOPPIA_ABSTRACTSTIXELMOTIONESTIMATOR_HPP
