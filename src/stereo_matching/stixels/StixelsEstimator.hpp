#ifndef STIXELSESTIMATOR_HPP
#define STIXELSESTIMATOR_HPP

#include "BaseStixelsEstimator.hpp"
#include "Stixel.hpp"

#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/multi_array.hpp>

#include <Eigen/Core>

#include <vector>

namespace doppia {

// forward declarations
class DisparityCostVolume;
class MetricStereoCamera;

/// Class dedicated to estimate the stixels once all required elements are available
/// This class is based on the u-v disparity estimation methods
/// In this context we mention u and v elements as
/// synomyms for x, y coordinates elements (in the image plane).
///
/// Part of this class is based on the work from Kubota et al.
/// "A global optimization algorithm for real-time on-board stereo obstacle detection system"
/// Susumu Kubota, T. Nakano and Y. Okamoto
/// Proceedings of the Intelligent Vehicles Symposium 2007
class StixelsEstimator: public BaseStixelsEstimator
{

protected:
    /// helper constructor for unit testing
    /// child class can create non-initializing constructors
    StixelsEstimator();

public:

    static boost::program_options::options_description get_args_options();

    StixelsEstimator(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width);
    virtual ~StixelsEstimator();


    /// Set a reference to disparity cost volume (computed assuming frontal objects)
    void set_disparity_cost_volume(const boost::shared_ptr<DisparityCostVolume> &cost_volume_p, const float max_cost_value);

    /// Set the pair of rectified images corresponding to the computed cost volume
    void set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

    /// Provide the best estimate available for the ground plane
    void set_ground_plane_estimate(const GroundPlane &ground_plane,
                                           const GroundPlaneEstimator::line_t &v_disparity_ground_line);

    void compute();

    typedef Eigen::MatrixXf u_disparity_cost_t;

    /// used to gui, debugging and testing
    /// @{
    const u_disparity_cost_t &get_u_disparity_cost() const;
    const u_disparity_cost_t &get_object_u_disparity_cost() const;
    const u_disparity_cost_t &get_ground_u_disparity_cost() const;
    const u_disparity_cost_t &get_M_cost() const;
    const std::vector<int> &get_u_disparity_ground_obstacle_boundary() const;
    const std::vector<int> &get_u_v_ground_obstacle_boundary() const;
    /// @}

protected:

    /// max_cost_value = cost_volume_estimator_p->get_maximum_cost_per_pixel
    float max_cost_value;
    boost::shared_ptr<DisparityCostVolume> pixels_cost_volume_p;

    virtual void compute_disparity_space_cost();

    /// mini fix to the "left area initialization issue"
    void fix_u_disparity_cost();

    float u_disparity_boundary_diagonal_weight;
    virtual void compute_ground_obstacle_boundary();

    void compute_ground_obstacle_boundary_v0();
    void compute_ground_obstacle_boundary_v1();
    void compute_ground_obstacle_boundary_v2();

    void post_process_object_u_disparity_cost(u_disparity_cost_t &cost) const;

    bool use_ground_cost_mirroring;
    float ground_cost_weight;
    float ground_cost_threshold;
    void post_process_ground_u_disparity_cost(u_disparity_cost_t &cost, const int num_rows) const;

    /// set u_v_ground_obstacle_boundary and the stixels outputs
    virtual void u_disparity_boundary_to_stixels();

    void high_pass_vertical_cost_filter(u_disparity_cost_t &cost) const;
    void low_pass_horizontal_cost_filter(u_disparity_cost_t &cost) const;

    float min_object_cost, max_object_cost;
    float min_ground_cost, max_ground_cost;

    u_disparity_cost_t
    object_u_disparity_cost, ground_u_disparity_cost,
    u_disparity_cost;

    u_disparity_cost_t M_cost;

    typedef boost::multi_array<int, 2> min_M_minus_c_indices_t;
    min_M_minus_c_indices_t min_M_minus_c_indices;

    /// for each u value, give the boundary disparity
    std::vector<int> u_disparity_ground_obstacle_boundary, u_v_ground_obstacle_boundary;


public:
    // GroundPlane is a Eigen structure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


};

} // end of namespace doppia

#endif // STIXELSESTIMATOR_HPP
