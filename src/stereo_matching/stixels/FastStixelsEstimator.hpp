#ifndef FASTSTIXELSESTIMATOR_HPP
#define FASTSTIXELSESTIMATOR_HPP

#include "StixelsEstimator.hpp"

#include "helpers/AlignedImage.hpp"

namespace doppia {

/// A SIMD enabled implementation of StixelsEstimator
class FastStixelsEstimator: public StixelsEstimator
{
public:
    //static boost::program_options::options_description get_args_options();

    FastStixelsEstimator(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width);
    ~FastStixelsEstimator();

    /// Set the pair of rectified images corresponding to the computed cost volume
    void set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

    /// Provide the best estimate available for the ground plane
    void set_ground_plane_estimate(const GroundPlane &ground_plane,
                                   const GroundPlaneEstimator::line_t &v_disparity_ground_line);

    void compute();


protected:

    const int disparity_offset;
    const int num_disparities;
    input_image_const_view_t input_left_view, input_right_view;
    boost::scoped_ptr<AlignedImage>
    transposed_left_image_p,
    transposed_right_image_p,
    rectified_right_image_p,
    transposed_rectified_right_image_p;

    void compute_disparity_space_cost();

    void compute_transposed_rectified_right_image();
    void compute_object_cost(u_disparity_cost_t &object_cost) const;
    void compute_ground_cost_v0(u_disparity_cost_t &ground_cost) const;

    /// v1 is faster than v0 (even without simd)
    void compute_ground_cost_v1(u_disparity_cost_t &ground_cost) const;

    void compute_object_and_ground_cost(u_disparity_cost_t &object_cost, u_disparity_cost_t &ground_cost) const;

};

/// Helper methods reused in ImagePlaneStixelsEstimator
/// @{
void compute_transposed_rectified_right_image(
        const AbstractStixelsEstimator::input_image_const_view_t &input_right_view,
        const AlignedImage::view_t &rectified_right_view,
        const AlignedImage::view_t &transposed_rectified_right_view,
        const int disparity_offset,
        const std::vector<int> &v_given_disparity,
        const std::vector<int> &disparity_given_v);

void sum_object_cost_baseline(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    float &object_cost_float);

void sum_object_cost_simd(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    float &object_cost_float);

/// @}

} // end of namespace doppia

#endif // FASTSTIXELSESTIMATOR_HPP
