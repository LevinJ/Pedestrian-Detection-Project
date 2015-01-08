#ifndef FASTGROUNDPLANEESTIMATOR_HPP
#define FASTGROUNDPLANEESTIMATOR_HPP

#include "BaseGroundPlaneEstimator.hpp"
#include "image_processing/IrlsLinesDetector.hpp"

#include <Eigen/Core>

#include <boost/scoped_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/cstdint.hpp>


namespace doppia {

// forward declarations
class AlignedImage;
class ResidualImageFilter;

class FastGroundPlaneEstimator : public BaseGroundPlaneEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    FastGroundPlaneEstimator(
        const boost::program_options::variables_map &options,
        const StereoCameraCalibration &stereo_calibration);
    ~FastGroundPlaneEstimator();

    typedef boost::gil::rgb8c_view_t input_image_view_t;
    void set_rectified_images_pair(input_image_view_t &left, input_image_view_t &right);

    void compute();

    typedef boost::multi_array<uint32_t, 2> v_disparity_data_t;
    typedef v_disparity_data_t::reference v_disparity_row_slice_t;
    const v_disparity_data_t &get_v_disparity() const;

    typedef boost::gil::gray32c_view_t v_disparity_const_view_t;
    const v_disparity_const_view_t &get_v_disparity_view() const;

    typedef IrlsLinesDetector::points_t points_t;
    const points_t &get_points() const;

    /// get the confidence of the current estimate,
    /// condifence is expected to be in the range [0, 1]
    const float get_confidence();

    bool is_computing_residual_image() const;
    const input_image_view_t get_left_half_view() const;

protected:

    bool should_do_residual_computation;
    bool silent_mode;
    boost::uint32_t cost_sum_saturation;

    // number of disparities used to estimate the ground plane
    const size_t max_disparity;
    boost::uint8_t y_stride;

    input_image_view_t input_left_view, input_right_view;
    boost::scoped_ptr<AlignedImage> left_image_p, right_image_p;

    v_disparity_data_t v_disparity_data;
    v_disparity_const_view_t v_disparity_view;

    boost::shared_ptr<IrlsLinesDetector> irls_lines_detector_p;
    IrlsLinesDetector::points_t points;
    Eigen::VectorXf row_weights, points_weights;

    void compute_v_disparity_data();
    void compute_v_disparity_row_simd(const input_image_view_t &left, const input_image_view_t &right,
        const int row,
        v_disparity_row_slice_t v_disparity_row);
    void compute_v_disparity_row_baseline(
        const input_image_view_t &left, const input_image_view_t &right,
        const int row,
        v_disparity_row_slice_t &v_disparity_row);

    /// Find the ground line in the v_disparity image
    /// @returns true if found a dominant line, false otherwise
    bool find_ground_line(AbstractLinesDetector::line_t &ground_line) const;

    int num_ground_plane_estimation_failures;
    void estimate_ground_plane();

    boost::scoped_ptr<ResidualImageFilter> residual_image_filter_p;

    bool confidence_is_up_to_date;
    float estimated_confidence;

};

} // end of namespace doppia

#endif // FASTGROUNDPLANEESTIMATOR_HPP
