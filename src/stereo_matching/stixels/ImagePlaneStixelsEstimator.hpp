#ifndef DOPPIA_IMAGEPLANESTIXELSESTIMATOR_HPP
#define DOPPIA_IMAGEPLANESTIXELSESTIMATOR_HPP

#include "BaseStixelsEstimator.hpp"

#include "opencv2/core/mat.hpp"

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>


namespace doppia {

// forward declarations
class AlignedImage;
class MetricStereoCamera;

class ImagePlaneStixelsEstimator: public BaseStixelsEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    ImagePlaneStixelsEstimator(
            const boost::program_options::variables_map &options,
            const MetricStereoCamera &camera,
            const float expected_object_height,
            const int minimum_object_height_in_pixels,
            const int stixel_width);
    ~ImagePlaneStixelsEstimator();

    /// Set the pair of rectified images corresponding to the computed cost volume
    void set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

    /// Provide the best estimate available for the ground plane
    void set_ground_plane_estimate(const GroundPlane &ground_plane,
                                   const GroundPlaneEstimator::line_t &v_disparity_ground_line);

    void compute();

    /// helpers for visualization
    /// @{
    typedef Eigen::MatrixXf cost_per_stixel_and_row_step_t;

    const cost_per_stixel_and_row_step_t &get_object_cost_per_stixel_and_row_step() const;
    const cost_per_stixel_and_row_step_t &get_ground_cost_per_stixel_and_row_step() const;
    const cost_per_stixel_and_row_step_t &get_cost_per_stixel_and_row_step() const;
    const std::vector<int> &get_stixel_and_row_step_ground_obstacle_boundary() const;
    /// @}

protected:

    friend void draw_stixels_estimation(const ImagePlaneStixelsEstimator &stixels_estimator,
                                        const boost::gil::rgb8c_view_t &left_input_view,
                                        boost::gil::rgb8_view_t &screen_left_view,
                                        boost::gil::rgb8_view_t &screen_right_view);


    /// multiple parameters of the method
    /// @{
    const int disparity_offset;
    const int num_disparities;

    const size_t num_row_steps;
    const bool should_estimate_stixel_bottom;
    const int stixel_support_width;

    float ground_cost_weight;
    float ground_cost_threshold;

    float u_disparity_boundary_diagonal_weight;
    /// @}

    input_image_const_view_t input_left_view, input_right_view;
    boost::scoped_ptr<AlignedImage>
    transposed_left_image_p,
    transposed_slim_left_image_p, // contains only the columns of interest
    transposed_right_image_p,
    rectified_right_image_p,
    transposed_rectified_right_image_p;

    cv::Mat gray_left_mat, df_dy_mat;

public:
    /// for each stixel, for each vertical step, which is the expected bottom row
    typedef boost::uint16_t row_t;
    typedef boost::multi_array<row_t, 2> row_given_stixel_and_row_step_t;

    typedef boost::int32_t disparity_t;
    typedef boost::multi_array<disparity_t, 2> disparity_given_stixel_and_row_step_t;

protected:
    row_given_stixel_and_row_step_t bottom_v_given_stixel_and_row_step, top_v_given_stixel_and_row_step;
    disparity_given_stixel_and_row_step_t disparity_given_stixel_and_row_step;

    /// for each column, and for each row_step find the most likely object bottom
    void find_stixels_bottom_candidates();
    void find_stixels_bottom_candidates_v0_baseline();
    void find_stixels_bottom_candidates_v1_compute_only_what_is_used();
    void set_fix_stixels_bottom_candidates();

    cost_per_stixel_and_row_step_t
    object_cost_per_stixel_and_row_step,
    ground_cost_per_stixel_and_row_step,
    cost_per_stixel_and_row_step;

    /// for each column, and for each row_step; compute the ground and object evidence
    void collect_stereo_evidence();
    void compute_transposed_rectified_right_image();
    void compute_object_cost(cost_per_stixel_and_row_step_t &object_cost) const;
    void compute_ground_cost(cost_per_stixel_and_row_step_t &ground_cost) const;

    cost_per_stixel_and_row_step_t M_cost;

    typedef boost::multi_array<int, 2> min_M_minus_c_indices_t;
    min_M_minus_c_indices_t min_M_minus_c_indices;


    /// estimate the distance using dynamic programming
    void estimate_stixels_bottom();

    /// simplistic method, good for a preliminary test
    void estimate_stixels_bottom_using_argmax_per_stixel();

    void estimate_stixels_bottom_using_dynamic_programming();
    void estimate_stixels_bottom_using_dynamic_programming_v0_backtracking();

    /// this vector stores the resulting boundary line
    std::vector<int> stixel_and_row_step_ground_obstacle_boundary;

    /// for each u value, give the boundary disparity
    std::vector<int> u_v_ground_obstacle_boundary;

    void u_v_disparity_boundary_to_stixels();


};

} // emd of namespace doppia

#endif // DOPPIA_IMAGEPLANESTIXELSESTIMATOR_HPP
