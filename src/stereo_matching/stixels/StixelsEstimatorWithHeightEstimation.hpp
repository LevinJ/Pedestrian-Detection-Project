#ifndef STIXELSESTIMATORWITHHEIGHTESTIMATION_HPP
#define STIXELSESTIMATORWITHHEIGHTESTIMATION_HPP

#include "StixelsEstimator.hpp"

#include <boost/shared_ptr.hpp>

namespace doppia {

// forward declarations
class AbstractPreprocessor;
class CpuPreprocessor;
class DisparityCostVolume;
class AbstractDisparityCostVolumeEstimator;

typedef StixelsEstimator::u_disparity_cost_t stixel_height_cost_t;

// helper methods that will be reused by other (non-inheriting) classes
void compute_stixels_heights_using_local_minima(
    const stixel_height_cost_t &stixel_height_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    stixels_t &the_stixels);

void compute_stixels_heights_using_dynamic_programming(
    const stixel_height_cost_t &stixel_height_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    const MetricStereoCamera &stereo_camera,
    stixels_t &the_stixels);

void depth_cost_to_stixel_height_cost(const Eigen::MatrixXf &depth_cost,
                                      const std::vector<int> &u_v_ground_obstacle_boundary,
                                      Eigen::MatrixXf &stixel_height_cost);

void apply_horizontal_smoothing(Eigen::MatrixXf &cost, const int smoothing_iterations = 3);

void vertical_normalization(Eigen::MatrixXf &cost,
                            const std::vector<int> &u_v_ground_obstacle_boundary);

void do_horizontal_averaging(const int kernel_size,
                             const DisparityCostVolume &input_cost_volume,
                             const float max_cost_value,
                             DisparityCostVolume &output_cost_volume);

// helper class that will be reused by other (non-inheriting) classes
class StixelsHeightPostProcessing
{
public:
    static boost::program_options::options_description get_args_options();

    StixelsHeightPostProcessing(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width);

    ~StixelsHeightPostProcessing();

    void operator()(
        const std::vector<int> &expected_v_given_disparity,
        const GroundPlane &the_ground_plane,
        stixels_t &stixels);

protected:

    const MetricStereoCamera &stereo_camera;
    const float expected_object_height; ///< [meters]
    const int minimum_object_height_in_pixels; ///< [pixels]
    const int stixel_width;

    int enforce_height_with_pixels_margin;
    float enforce_height_with_meters_margin;

    bool should_enforce_reasonable_stixels_heights;

};

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class StixelsEstimatorWithHeightEstimation : public StixelsEstimator
{
public:

    typedef StixelsEstimator::u_disparity_cost_t stixel_height_cost_t;

    static boost::program_options::options_description get_args_options();

    StixelsEstimatorWithHeightEstimation(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const boost::shared_ptr<AbstractPreprocessor> preprocessor_p,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width);
    ~StixelsEstimatorWithHeightEstimation();

    /// Set the pair of rectified images corresponding to the computed cost volume
    void set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

    void compute();

    /// used to gui, debugging and testing
    /// @{
    const stixel_height_cost_t &get_depth_map() const;
    const stixel_height_cost_t &get_stixel_height_cost() const;
    /// @}

protected:


    /// Compute the heights cost matrix
    void compute_stixel_height_cost();
    void compute_stixel_height_cost_using_partial_disparity_map();
    void compute_stixel_height_cost_directly();
    void compute_stixel_height_cost_deprecated();

    void compute_disparity_map(const DisparityCostVolume &cost_volume,
                               Eigen::MatrixXf &disparity_map);

    void compute_disparity_likelihood_map(const DisparityCostVolume &cost_volume,
                                          Eigen::MatrixXf &disparity_likelihood_map);


    void compute_stixel_height_cost(const DisparityCostVolume &cost_volume);

    void compute_stixel_height_cost_column(const int column_index,
                                           const DisparityCostVolume &cost_volume);

    /// Apply dynamic programming over the stixels heights
    void compute_stixels_heights();
    void enforce_reasonable_stixels_heights();

    boost::scoped_ptr<StixelsHeightPostProcessing> stixels_height_post_processing_p;

    boost::shared_ptr<AbstractDisparityCostVolumeEstimator> cost_volume_estimator_p;
    boost::shared_ptr<DisparityCostVolume> original_pixels_cost_volume_p;
    boost::shared_ptr<CpuPreprocessor> preprocessor_p;

    stixel_height_cost_t stixel_height_cost, depth_map;
    input_image_const_view_t input_left_view, input_right_view;

    bool use_partial_disparity_map;
    int max_disparity_margin, min_disparity_margin;
    int horizontal_kernel_size, vertical_kernel_size;


    // temporary cost volumes used during computation
    boost::scoped_ptr<DisparityCostVolume> cost_volume_one_p, cost_volume_two_p;
};

} // end of namespace doppia

#endif // STIXELSESTIMATORWITHHEIGHTESTIMATION_HPP
