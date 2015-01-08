#ifndef STIXELSESTIMATORWITH3DCOST_HPP
#define STIXELSESTIMATORWITH3DCOST_HPP

#include "StixelsEstimator.hpp"
#include <boost/multi_array.hpp>

namespace doppia {

/// StixelsEstimator variant that uses a 3d cost volume.
/// Using this 3d cost we can simultaneously estimate the
/// position and height of the objects
class StixelsEstimatorWith3dCost : public StixelsEstimator
{
public:
    typedef boost::multi_array<float, 3> cost_volume_t;

public:

    StixelsEstimatorWith3dCost(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int num_height_levels,
        const int stixel_width);

    ~StixelsEstimatorWith3dCost();

    /// Provide the best estimate available for the ground plane
    void set_ground_plane_estimate(const GroundPlane &ground_plane,
                                   const GroundPlaneEstimator::line_t &v_disparity_ground_line);

    void compute();

protected:

    const int num_height_levels;

    /// for each height_level and disparity we get the minimum relevant v value
    /// using the ground plane and an expected maximum height
    typedef boost::multi_array<int, 2> minimum_v_t;
    minimum_v_t minimum_v;
    void set_mininum_v_given_disparity_and_height_level();

    /// for each u value, give the height level to the corresponding stixel
    /// this is a companion of the u_disparity_ground_obstacle_boundary vector
    std::vector<int> u_height_ground_obstacle_boundary;

    cost_volume_t object_cost_volume;
    cost_volume_t M_cost_volume;
    cost_volume_t u_disparity_cost_volume;
    /// compute the u-height-disparity cost volume
    void compute_cost_volume();

    /// estimate the stixels using dynamic programming over the cost volume
    void compute_stixels();


    void high_pass_vertical_cost_filter(cost_volume_t &cost_volume);
    void low_pass_horizontal_cost_filter(cost_volume_t &cost_volume);
};

} // end of namespace doppia

#endif // STIXELSESTIMATORWITH3DCOST_HPP
