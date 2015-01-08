#ifndef GROUNDPLANEESTIMATOR_HPP
#define GROUNDPLANEESTIMATOR_HPP

#include "BaseGroundPlaneEstimator.hpp"

#include "GroundPlane.hpp"
#include "image_processing/AbstractLinesDetector.hpp"

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace doppia {

// forward declarations
class DisparityCostVolume;
class StereoCameraCalibration;

class GroundPlaneEstimator: public BaseGroundPlaneEstimator
{
public:

    typedef AbstractLinesDetector::line_t line_t;

    static boost::program_options::options_description get_args_options();

    GroundPlaneEstimator(
        const boost::program_options::variables_map &options,
        const StereoCameraCalibration &stereo_calibration,
        const bool cost_volume_is_from_residual_image);
    virtual ~GroundPlaneEstimator();


    /// Set the disparity cost volume computed
    void set_ground_disparity_cost_volume(const boost::shared_ptr<DisparityCostVolume> &cost_volume_p);

    void compute();

    typedef boost::gil::gray8c_view_t v_disparity_const_view_t;

    /// get the v_disparity obtained from the stereo matching cost volume
    const v_disparity_const_view_t &get_raw_v_disparity_view() const;

    /// get the v_disparity after preprocessing
    const v_disparity_const_view_t &get_v_disparity_view() const;


protected:

    const bool cost_volume_is_from_residual_image;

    boost::shared_ptr<DisparityCostVolume> cost_volume_p;

    int num_ground_plane_estimation_failures;


    Eigen::MatrixXf v_disparity_data;
    boost::gil::gray8_image_t v_disparity_image, raw_v_disparity_image;
    boost::gil::gray8_view_t v_disparity_image_view, raw_v_disparity_image_view;
    v_disparity_const_view_t v_disparity_image_const_view, raw_v_disparity_image_const_view;
    float v_disparity_x_offset;

    void compute_v_disparity_data();
    void compute_v_disparity_image();
    void estimate_ground_plane();

    /// Find the ground line in the v_disparity image
    /// @returns true if found a dominant line, false otherwise
    bool find_ground_line(AbstractLinesDetector::line_t &ground_line) const;

    Eigen::MatrixXi v_disparity_mask;
    void compute_v_disparity_mask();

private:

    /// orientation in [radians]
    void set_orientation_filter(const float orientation);
    float current_orientation_filter_phase;
    cv::Mat_<float> orientation_filter_kernel;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // end of namespace doppia

#endif // GROUNDPLANEESTIMATOR_HPP
