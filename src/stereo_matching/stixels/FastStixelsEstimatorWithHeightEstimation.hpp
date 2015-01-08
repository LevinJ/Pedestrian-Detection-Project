#ifndef FASTSTIXELSESTIMATORWITHHEIGHTESTIMATION_HPP
#define FASTSTIXELSESTIMATORWITHHEIGHTESTIMATION_HPP

#include "FastStixelsEstimator.hpp"

#include "StixelsEstimatorWithHeightEstimation.hpp" // for the stixel_height_cost_t definition

#include "helpers/AlignedImage.hpp"

namespace doppia {

class FastStixelsEstimatorWithHeightEstimation : public FastStixelsEstimator
{
public:

    //static boost::program_options::options_description get_args_options();

    FastStixelsEstimatorWithHeightEstimation(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width);

    ~FastStixelsEstimatorWithHeightEstimation();

    void compute();

    const stixel_height_cost_t &get_stixel_height_cost() const;

    const stixel_height_cost_t &get_disparity_likelihood_map() const;

protected:

    int minimum_disparity_margin, maximum_disparity_margin, disparity_range;
    stixel_height_cost_t stixel_height_cost, disparity_likelihood_map, M_cost;
    AlignedImage disparity_likelihood_helper_image;

    // FIXME should I make column_disparity_cost_t 3d ?
    typedef Eigen::aligned_allocator<boost::uint16_t> aligned_uint16_allocator;
    typedef boost::multi_array< boost::uint16_t, 2, aligned_uint16_allocator> column_disparity_cost_t;
    column_disparity_cost_t column_disparity_cost, column_disparity_horizontal_cost;

    /// Compute the heights cost matrix, without using any depth map
    void compute_stixel_height_cost();

    /// disparity_likelihood_map is in range [0,1]

    /// v0 is the original version meant to be SIMD effective
    /// (seems to be inefficient in memory access)
    void compute_disparity_likelihood_map_v0(Eigen::MatrixXf &disparity_likelihood_map);

    /// v1 we try to copy the data first and them apply SIMD
    /// (copying the data turned out to be super slow)
    void compute_disparity_likelihood_map_v1(Eigen::MatrixXf &disparity_likelihood_map);

    /// v2 we improve the memory access, but skip the SIMD usage
    /// (v2 is much faster than v1, but slightly slower than v0)
    void compute_disparity_likelihood_map_v2(Eigen::MatrixXf &disparity_likelihood_map);

    /// just like v0, but we do some stragegic copies to memory aligned buffers
    /// (v3 seems slightly faster than v0)
    void compute_disparity_likelihood_map_v3(Eigen::MatrixXf &disparity_likelihood_map);

    void load_data_into_disparity_likelihood_helper();
    void compute_colum_disparity_cost_from_disparity_likelihood_helper();
    void disparity_likelihood_from_colum_disparity_cost(Eigen::MatrixXf &disparity_likelihood_map);

    void compute_column_disparity_cost(const int column,
                                       const int num_columns,
                                       const int disparity,
                                       std::vector<uint16_t> &disparity_cost);

    void compute_column_disparity_cost_and_copy_data(const int column,
                                                     const int num_columns,
                                                     const int disparity,
                                                     AlignedImage &left_data,
                                                     std::vector<uint16_t> &disparity_cost);

    void compute_column_disparity_cost_using_data(const int column,
                                                  const int num_columns,
                                                  const int disparity,
                                                  const AlignedImage &left_data,
                                                  std::vector<uint16_t> &disparity_cost);

    void compute_column_disparity_cost_v2(const int column,
                                          const int num_columns,
                                          const int disparity,
                                          std::vector<uint16_t> &horizontal_sad_cost,
                                          std::vector<uint16_t> &disparity_cost);

    void compute_horizontal_5px_sad_border_case(const int column, const int num_columns,
                                                const int object_bottom_v,
                                                const int disparity, std::vector<uint16_t> &disparity_cost);

    void compute_horizontal_5px_sad_baseline(const int column,
                                             const int object_bottom_v,
                                             const int disparity,
                                             std::vector<uint16_t> &disparity_cost);

    void compute_transposed_horizontal_5px_sad_baseline(const int column,
                                                        const int object_bottom_v,
                                                        const int disparity,
                                                        std::vector<uint16_t> &disparity_cost);

    void compute_horizontal_5px_sad_simd(const int column,
                                         const int object_bottom_v,
                                         const int disparity,
                                         std::vector<uint16_t> &disparity_cost);


    void compute_horizontal_5px_sad_simd_and_copy_data(const int column,
                                                       const int object_bottom_v,
                                                       const int disparity,
                                                       AlignedImage &left_data,
                                                       std::vector<uint16_t> &disparity_cost);

    void compute_horizontal_5px_sad_simd_using_data(const int column,
                                                    const int object_bottom_v,
                                                    const int disparity,
                                                    const AlignedImage &left_data,
                                                    std::vector<uint16_t> &disparity_cost);


    /// Apply dynamic programming over the stixels heights
    void compute_stixels_heights();

    /// post process the heights
    void enforce_reasonable_stixels_heights();

    boost::scoped_ptr<StixelsHeightPostProcessing> stixels_height_post_processing_p;

    /// helper method to retrieve commonly used values: the range of disparities to explore
    void get_colum_disparities_range(
        const int column,
        int &minimum_disparity, int &maximum_disparity, int &stixel_disparity);
};

} // end namespace doppia

#endif // FASTSTIXELSESTIMATORWITHHEIGHTESTIMATION_HPP
