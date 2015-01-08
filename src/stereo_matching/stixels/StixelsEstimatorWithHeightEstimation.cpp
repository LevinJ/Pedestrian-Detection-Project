#include "StixelsEstimatorWithHeightEstimation.hpp"

#include "video_input/preprocessing/AbstractPreprocessor.hpp"
#include "video_input/preprocessing/CpuPreprocessor.hpp"
#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"
#include "stereo_matching/cost_volume/AbstractDisparityCostVolumeEstimator.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <omp.h>

#include <cstdio>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelsEstimatorWithHeightEstimation");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelsEstimatorWithHeightEstimation");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelsEstimatorWithHeightEstimation");
}

} // end of anonymous namespace


namespace doppia {

using namespace boost;

typedef DisparityCostVolume::range_t range_t;
typedef DisparityCostVolume::const_data_3d_view_t const_data_3d_view_t;
typedef DisparityCostVolume::data_3d_view_t data_3d_view_t;
typedef DisparityCostVolume::const_data_2d_view_t const_data_2d_view_t;
typedef DisparityCostVolume::const_data_1d_view_t const_data_1d_view_t;
typedef DisparityCostVolume::const_data_2d_subarray_t const_data_2d_subarray_t;
typedef DisparityCostVolume::const_data_1d_subarray_t const_data_1d_subarray_t;

// FIXME where should I put this ?
/// min_float_disparity will be used to replace integer disparity == 0
const float min_float_disparity = 0.8f;


// ------------------------------------------------------

program_options::options_description StixelsHeightPostProcessing::get_args_options()
{
    program_options::options_description desc("StixelsHeightPostProcessing options");

    desc.add_options()

            ("stixel_world.enforce_height_with_pixels_margin",
             program_options::value<int>()->default_value(20),
             "if this value is > 0, the stixels height estimate will be enforced to "
             "be in the given margin (in pixels) from the expected object height."
             "if this value is <= 0, no height enforcing will be done."
             "this option is incompatible with enforce_height_with_meters_margin")

            ("stixel_world.enforce_height_with_meters_margin",
             program_options::value<float>()->default_value(-1),
             "if this value is > 0, the stixels height estimate will be enforced to "
             "be in the given margin (in meters) from the expected object height."
             "if this value is <= 0, no height enforcing will be done."
             "this option is incompatible with enforce_height_with_pixels_margin")
            ;

    return desc;
}


StixelsHeightPostProcessing::StixelsHeightPostProcessing(
    const boost::program_options::variables_map &options,
    const MetricStereoCamera &stereo_camera_,
    const float expected_object_height_,
    const int minimum_object_height_in_pixels_,
    const int stixel_width_)
    : stereo_camera(stereo_camera_),
      expected_object_height(expected_object_height_),
      minimum_object_height_in_pixels(minimum_object_height_in_pixels_),
      stixel_width(stixel_width_)
{

    enforce_height_with_pixels_margin = get_option_value<int>(options, "stixel_world.enforce_height_with_pixels_margin");
    enforce_height_with_meters_margin = get_option_value<float>(options, "stixel_world.enforce_height_with_meters_margin");

    should_enforce_reasonable_stixels_heights =
            enforce_height_with_pixels_margin > 0 or enforce_height_with_meters_margin > 0;

    if(enforce_height_with_pixels_margin > 0 and enforce_height_with_meters_margin > 0)
    {
        throw std::invalid_argument("enforce_height_with_pixels_margin and enforce_height_with_meters_margin cannot be used together");
    }

    return;
}


StixelsHeightPostProcessing::~StixelsHeightPostProcessing()
{
    // nothing to do here
    return;
}


void StixelsHeightPostProcessing::operator()(const std::vector<int> &expected_v_given_disparity,
                                             const GroundPlane &the_ground_plane,
                                             stixels_t &the_stixels)
{

    if(should_enforce_reasonable_stixels_heights == false)
    {
        // nothing to do here
        return;
    }

    //const int num_rows = stixel_height_cost.rows();
    //const int num_columns = stixel_height_cost.cols();
    assert(stixel_width == 1);
    assert(the_stixels.empty() == false);
    const int num_columns = the_stixels.size();

    const bool use_meter_based_limits = enforce_height_with_meters_margin > 0;
    const bool use_pixels_based_limits = enforce_height_with_pixels_margin > 0;

    if(use_meter_based_limits)
    {

        // fix the top of each stixel to be within the high confidence part of the object
        //const float minimum_height = expected_object_height*0.5; // [meters]
        //const float maximum_height = expected_object_height*2.0; // [meters]
        const float minimum_height = std::max(0.0f, expected_object_height - enforce_height_with_meters_margin);
        const float maximum_height = expected_object_height + enforce_height_with_meters_margin;

#pragma omp parallel for
        for(int col = 0; col < num_columns; col += 1)
        {
            const float stixel_disparity =
                    std::max<float>(min_float_disparity, the_stixels[col].disparity);
            const float stixel_distance = stereo_camera.disparity_to_depth(stixel_disparity);

            // FIXME getting v for height should be a function of StixelsEstimator
            Eigen::Vector2f uv_point =
                    stereo_camera.get_left_camera().project_ground_plane_point(the_ground_plane,
                                                                               0, stixel_distance, minimum_height);
            const int max_v = static_cast<int>(uv_point[1]);

            uv_point =
                    stereo_camera.get_left_camera().project_ground_plane_point(the_ground_plane,
                                                                               0, stixel_distance, maximum_height);
            const int min_v = static_cast<int>(uv_point[1]);

            uv_point =
                    stereo_camera.get_left_camera().project_ground_plane_point(the_ground_plane,
                                                                               0, stixel_distance, expected_object_height);
            const int expected_v = static_cast<int>(uv_point[1]);

            const int estimated_top_y = the_stixels[col].top_y;

            if(estimated_top_y < min_v or estimated_top_y > max_v)
            {
                the_stixels[col].top_y = expected_v;
                the_stixels[col].default_height_value = true;
            }
            else
            {
                // we keep the current estimate
            }

        } // end of "for each column"

    } // end of "if use_meter_based_limits"
    else if (use_pixels_based_limits)
    {
        // we will use pixel based limits instead of meters based ones
        const int max_pixels_margin = enforce_height_with_pixels_margin; // [pixels]
        // enforce_height_with_pixels_margin should be based on the cumulative histogram of the absolute error

#pragma omp parallel for
        for(int col = 0; col < num_columns; col += 1)
        {
            Stixel &the_stixel = the_stixels[col];
            const int &expected_top_v = expected_v_given_disparity[the_stixel.disparity];

            const int v_margin = std::abs(expected_top_v - the_stixel.top_y);
            if(v_margin > max_pixels_margin)
            {
                the_stixel.top_y = expected_top_v;
                the_stixel.default_height_value = true;
            }
            else
            {
                // we keep the current estimate
            }

        } // end of "for each column"

    } // end of "if use_pixels_based_limits"
    else
    {
        throw std::invalid_argument("StixelsEstimatorWithHeightEstimation::enforce_reasonable_stixels_heights called but no margin was set");
    }


    return;
}


// ------------------------------------------------------

program_options::options_description StixelsEstimatorWithHeightEstimation::get_args_options()
{
    program_options::options_description desc("StixelsEstimatorWithHeightEstimation options");

    desc.add_options()

            ("stixel_world.use_partial_disparity_map",
             program_options::value<bool>()->default_value(false),
             "use partial disparity map or directly computing the height, without ever estimating the disparity")

            ("stixel_world.max_disparity_margin",
             program_options::value<int>()->default_value(5),
             "positive disparity margin used to estimate the stixel heigth. Value is in [pixels]. "
             "Set to -1 to use the full disparity (upper) range")

            ("stixel_world.min_disparity_margin",
             program_options::value<int>()->default_value(15),
             "negative disparity margin used to estimate the stixel heigth. Value is in [pixels]. "
             "Set to -1 to use the full disparity (lower) range")

            ("stixel_world.cost_volume_horizontal_kernel_size",
             program_options::value<int>()->default_value(5),
             "when computing the partial depth map or the disparity likelihood map, "
             "a horizontal smoothing of size kernel_size will be applied if kernel_size > 1. "
             "kernel_size must be an odd number")

            ("stixel_world.cost_volume_vertical_kernel_size",
             program_options::value<int>()->default_value(5),
             "when computing the partial depth map or the disparity likelihood map, "
             "a vertical smoothing of size kernel_size will be applied if kernel_size > 1. "
             "kernel_size must be an odd number")


            ("stixel_world.enforce_height_with_pixels_margin",
             program_options::value<int>()->default_value(20),
             "if this value is > 0, the stixels height estimate will be enforced to "
             "be in the given margin (in pixels) from the expected object height. "
             "if this value is <= 0, no height enforcing will be done. "
             "this option is incompatible with enforce_height_with_meters_margin")

            ("stixel_world.enforce_height_with_meters_margin",
             program_options::value<float>()->default_value(-1),
             "if this value is > 0, the stixels height estimate will be enforced to "
             "be in the given margin (in meters) from the expected object height. "
             "if this value is <= 0, no height enforcing will be done. "
             "this option is incompatible with enforce_height_with_pixels_margin")
            ;


    return desc;

}


void check_kernel_size(const int kernel_size, std::string kernel_name )
{
    if((kernel_size > 1) and ((kernel_size % 2) != 1))
    {
        log_error() << kernel_name << " has an invalid value" << std::endl;
        throw std::invalid_argument("kernel_size must be an odd number");
    }
    else
    {
        // nothing to do
    }

    return;
}


StixelsEstimatorWithHeightEstimation::StixelsEstimatorWithHeightEstimation(
    const boost::program_options::variables_map &options,
    const MetricStereoCamera &camera,
    const boost::shared_ptr<AbstractPreprocessor> preprocessor_p_,
    const float expected_objects_height,
    const int minimum_object_height_in_pixels,
    //const int num_height_levels_,
    const int stixel_width)
    : StixelsEstimator(options, camera, expected_objects_height, minimum_object_height_in_pixels, stixel_width)//,
    //num_height_levels(num_height_levels_)
{
    use_partial_disparity_map = get_option_value<bool>(options, "stixel_world.use_partial_disparity_map");
    max_disparity_margin = get_option_value<int>(options, "stixel_world.max_disparity_margin");
    min_disparity_margin = get_option_value<int>(options, "stixel_world.min_disparity_margin");


    stixels_height_post_processing_p.reset(new StixelsHeightPostProcessing(options, camera,
                                                                           expected_objects_height,
                                                                           minimum_object_height_in_pixels,
                                                                           stixel_width));



    if(use_partial_disparity_map)
    {
        log_info() << "Will use (partial or full) disparity map to compute the stixels height" << std::endl;
    }
    else
    {
        log_info() << "Will compute the stixels height directly from the cost volume (without estimating disparity)" << std::endl;
    }

    if(max_disparity_margin >=0 or min_disparity_margin >= 0)
    {
        log_info() << "Will use partial disparities information compute the stixels height" << std::endl;
    }
    else
    {
        log_info() << "Will use full disparity range to compute the stixels height" << std::endl;
    }


    preprocessor_p = boost::dynamic_pointer_cast<CpuPreprocessor>(preprocessor_p_);

    if(preprocessor_p == false)
    {
        throw std::invalid_argument("StixelsEstimatorWithHeightEstimation requires to receive an CpuPreprocessor instance");
    }


    horizontal_kernel_size = get_option_value<int>(options, "stixel_world.cost_volume_horizontal_kernel_size");
    vertical_kernel_size = get_option_value<int>(options, "stixel_world.cost_volume_vertical_kernel_size");

    check_kernel_size(horizontal_kernel_size, "stixel_world.cost_volume_horizontal_kernel_size");
    check_kernel_size(vertical_kernel_size, "stixel_world.cost_volume_vertical_kernel_size");


    original_pixels_cost_volume_p.reset(new DisparityCostVolume());

    const bool compute_original_pixels_cost_volume = false;
    if(compute_original_pixels_cost_volume)
    {
        cost_volume_estimator_p.reset(DisparityCostVolumeEstimatorFactory::new_instance(options));
    }
    else
    {
        // cost_volume_estimator_p is left to null
    }

    return;
}


StixelsEstimatorWithHeightEstimation::~StixelsEstimatorWithHeightEstimation()
{
    // nothing to do here
    return;
}


void StixelsEstimatorWithHeightEstimation::set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right)
{
    input_left_view = left;
    input_right_view = right;


    if(cost_volume_estimator_p)
    { // compute_original_pixels_cost_volume
        original_pixels_cost_volume_p->resize(*pixels_cost_volume_p);
        cost_volume_estimator_p->compute(left, right, *original_pixels_cost_volume_p);
    }
    else
    {
        original_pixels_cost_volume_p = pixels_cost_volume_p;
    }

    return;
}


void disparity_map_to_stixel_height_cost(const Eigen::MatrixXf &depth_map,
                                         const MetricStereoCamera &stereo_camera,
                                         const GroundPlane &the_ground_plane,
                                         const std::vector<int> &u_disparity_ground_obstacle_boundary,
                                         const std::vector<int> &u_v_ground_obstacle_boundary,
                                         Eigen::MatrixXf &stixel_height_cost);


void vertical_normalization(Eigen::MatrixXf &cost,
                            const std::vector<int> &u_v_ground_obstacle_boundary)
{
    const int num_rows = cost.rows();
    const int num_columns = cost.cols();

    //const int top_limit = 100; //0 // 50
    const int top_limit = 10;
#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const int a = top_limit, b = std::max(bottom_v - top_limit, 1);

        const float col_max = cost.col(col).segment(a,b).maxCoeff();
        const float col_min = cost.col(col).segment(a,b).minCoeff();
        cost.col(col).segment(a,b).array() -= col_min;
        if(col_max != col_min)
        {
            cost.col(col).segment(a,b) /= (col_max - col_min);
        }

        if(top_limit > 0)
        {
            cost.col(col).segment(0, top_limit).setConstant(1);
        }
        cost.col(col).segment(bottom_v, num_rows - bottom_v).setConstant(1);
    }

    return;
}


void apply_horizontal_smoothing(Eigen::MatrixXf &cost, const int smoothing_iterations)
{

    if(smoothing_iterations <= 0)
    {
        return;
    }

    //const int num_rows = cost.rows();
    const int num_columns = cost.cols();


    for(int i=0; i < smoothing_iterations; i+=1)
    {
        Eigen::MatrixXf t_cost = cost;

#pragma omp parallel for
        for(int col = 1; col < (num_columns - 1); col += 1)
        {
            const int previous_col = col - 1;
            const int next_col = col+1;
            t_cost.col(col) += cost.col(previous_col);
            t_cost.col(col) += cost.col(next_col);
            t_cost.col(col) /= 3;
        }
        cost = t_cost;
    }

    return;
}


void StixelsEstimatorWithHeightEstimation::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();

    // compute the stixels distances --
    StixelsEstimator::compute();

    // find the optimal stixels height --
    // (using dynamic programming)    
    compute_stixel_height_cost();
    compute_stixels_heights();

    // timing ---
    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;

    if((num_iterations % num_iterations_for_timing) == 0)
    {
        printf("Average StixelsEstimatorWithHeightEstimation::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
        cumulated_time = 0;
    }

    return;
}


const StixelsEstimatorWithHeightEstimation::stixel_height_cost_t &
StixelsEstimatorWithHeightEstimation::get_depth_map() const
{
    return depth_map;
}


const StixelsEstimatorWithHeightEstimation::stixel_height_cost_t &
StixelsEstimatorWithHeightEstimation::get_stixel_height_cost() const
{
    return stixel_height_cost;
}


void do_horizontal_averaging(const int kernel_size,
                             const DisparityCostVolume &input_cost_volume,
                             const float max_cost_value,
                             DisparityCostVolume &output_cost_volume)
{
    // pixels_cost_volume_p data is organized as y (rows), x (columns), disparity


    const int num_rows = input_cost_volume.rows();
    const int num_columns = input_cost_volume.columns();
    const int num_disparities = input_cost_volume.disparities();

    assert(&input_cost_volume != &output_cost_volume);
    assert(kernel_size % 2 == 1);

    if(kernel_size == 1)
    {
        output_cost_volume = input_cost_volume;
        return;
    }

    const_data_3d_view_t input_data = input_cost_volume.get_costs_views();
    data_3d_view_t output_data = output_cost_volume.get_costs_views();

    assert(input_data.dimensionality == output_data.dimensionality );
    assert(input_data.size() == output_data.size() );
    assert(input_data.shape()[0] == output_data.shape()[0] );
    assert(input_data.shape()[1] == output_data.shape()[1] );
    assert(input_data.shape()[2] == output_data.shape()[2] );

    const int half_kernel_size = (kernel_size - 1) / 2;


#pragma omp parallel for
    for(int v=0; v < num_rows; v+=1)
    {
        for(int u=0; u < num_columns; u+=1)
        {
            for(int d=0; d < num_disparities; d+=1)
            {
                float t_cost = 0;
                for(int u_offset= -half_kernel_size; u_offset <= half_kernel_size; u_offset +=1 )
                {
                    const int t_u = u + u_offset;
                    if(t_u < 0 or t_u >= num_columns)
                    {
                        t_cost += max_cost_value;
                    }
                    else
                    {
                        t_cost += input_data[v][t_u][d];
                    }
                }

                t_cost /= kernel_size;
                output_data[v][u][d] = t_cost;

            } // end of "for each disparity"
        } // end of "for each column"
    } // end of "for each row"

    return;
}


void do_vertical_averaging(const int kernel_size,
                           const DisparityCostVolume &input_cost_volume,
                           const float max_cost_value,
                           DisparityCostVolume &output_cost_volume)
{
    // pixels_cost_volume_p data is organized as y (rows), x (columns), disparity

    const int num_rows = input_cost_volume.rows();
    const int num_columns = input_cost_volume.columns();
    const int num_disparities = input_cost_volume.disparities();

    assert(&input_cost_volume != &output_cost_volume);
    assert(kernel_size % 2 == 1);

    if(kernel_size == 1)
    {
        output_cost_volume = input_cost_volume;
        return;
    }

    const_data_3d_view_t input_data = input_cost_volume.get_costs_views();
    data_3d_view_t output_data = output_cost_volume.get_costs_views();

    assert(input_data.dimensionality == output_data.dimensionality );
    assert(input_data.size() == output_data.size() );
    assert(input_data.shape()[0] == output_data.shape()[0] );
    assert(input_data.shape()[1] == output_data.shape()[1] );
    assert(input_data.shape()[2] == output_data.shape()[2] );

    //const_data_2d_view_t rows_disparities_cost_slice =
    //        cost_volume.rows_disparities_slice(column_index);

    const int half_kernel_size = (kernel_size - 1) / 2;

#pragma omp parallel for
    for(int v=0; v < num_rows; v+=1)
    {
        for(int u=0; u < num_columns; u+=1)
        {
            for(int d=0; d < num_disparities; d+=1)
            {
                float t_cost = 0;
                for(int v_offset= -half_kernel_size; v_offset <= half_kernel_size; v_offset +=1 )
                {
                    const int t_v = v + v_offset;
                    if(t_v < 0 or t_v >= num_rows)
                    {
                        t_cost += max_cost_value;
                    }
                    else
                    {
                        t_cost += input_data[t_v][u][d];
                    }
                }

                t_cost /= kernel_size;
                output_data[v][u][d] = t_cost;

            } // end of "for each disparity"
        } // end of "for each column"
    } // end of "for each row"


    return;
}


void compute_winner_take_all_disparity_map(const DisparityCostVolume &cost_volume,
                                           const int minimum_disparity_margin,
                                           const int maximum_disparity_margin,
                                           const float max_cost_value,
                                           const std::vector<int> &disparity_given_v,
                                           const std::vector<int> &u_disparity_ground_obstacle_boundary,
                                           const std::vector<int> &u_v_ground_obstacle_boundary,
                                           Eigen::MatrixXf &disparity_map)
{
    const int num_rows = cost_volume.rows();
    const int num_columns = cost_volume.columns();
    const int num_disparities = cost_volume.disparities();

#pragma omp parallel for
    for(int v=0; v < num_rows; v+=1)
    {
        for(int u=0; u < num_columns; u+=1)
        {
            const int bottom_v = u_v_ground_obstacle_boundary[u];
            if(v > bottom_v)
            {
                // ground area
                disparity_map(v, u) = disparity_given_v[v];
                //depth_map(v, u) = 0; // FIXME just for visualization
            }
            else
            { // above the ground area

                int maximum_disparity = num_disparities;
                int minimum_disparity = 0;

                const int stixel_disparity = u_disparity_ground_obstacle_boundary[u];

                // will use_partial_depth_range if
                // maximum_disparity_margin >=0 or minimum_disparity_margin >= 0
                if(maximum_disparity_margin >= 0)
                {
                    //const int maximum_disparity_margin = 5;
                    //const int maximum_disparity = disparity_given_v[v];
                    maximum_disparity = std::min(stixel_disparity + maximum_disparity_margin, num_disparities);
                }

                if(minimum_disparity_margin >= 0)
                {
                    //const int minimum_disparity_margin = 15;
                    minimum_disparity = std::max(0, stixel_disparity - minimum_disparity_margin);
                }

                const float maximum_possible_mismatch = max_cost_value;
                float min_cost_found = maximum_possible_mismatch;
                int min_cost_index = 0;
                for(int d=minimum_disparity; d < maximum_disparity; d+=1)
                {
                    const float &cost = cost_volume.get_costs_views()[v][u][d];
                    if(cost < min_cost_found)
                    {
                        min_cost_found = cost;
                        min_cost_index = d;
                    }

                } // end of "for each disparity between infinity and nearest stixel"

                disparity_map(v, u) = min_cost_index;

            } // end of "if above the ground or not"

        } // end of "for each column"
    } // end of "for each row"

    return;
}


void StixelsEstimatorWithHeightEstimation::compute_disparity_map(
    const DisparityCostVolume &cost_volume,
    Eigen::MatrixXf &disparity_map)
{

    const int num_rows = cost_volume.rows();
    const int num_columns = cost_volume.columns();
    //const int num_disparities = cost_volume.disparities();

    // Compute depth map for every pixel above the ground (using stixel constraint)
    if(cost_volume_one_p == false)
    {
        cost_volume_one_p.reset(new DisparityCostVolume());
    }

    if(cost_volume_two_p == false)
    {
        cost_volume_two_p.reset(new DisparityCostVolume());
    }
    DisparityCostVolume &cost_volume_one = *cost_volume_one_p, &cost_volume_two = *cost_volume_two_p;

    cost_volume_one.resize(cost_volume);
    cost_volume_two.resize(cost_volume);

    const bool should_do_horizontal_averaging = horizontal_kernel_size > 1;
    const bool should_do_vertical_averaging = vertical_kernel_size > 1;

    const float max_cost_value = this->max_cost_value; // cost_volume_estimator_p->get_maximum_cost_per_pixel();

    const DisparityCostVolume *input_p = &cost_volume;
    DisparityCostVolume *output_p = &cost_volume_one;


    // (for every pixel above the ground)
    if(should_do_horizontal_averaging)
    {
        // do horizontal averaging
        do_horizontal_averaging(horizontal_kernel_size, *input_p, max_cost_value, *output_p);
        input_p = output_p;
        output_p = &cost_volume_two;
    }

    if(should_do_vertical_averaging)
    {
        // do vertical averaging
        do_vertical_averaging(vertical_kernel_size, *input_p, max_cost_value, *output_p);
        input_p = output_p;
    }


    // find minima above the ground
    disparity_map = Eigen::MatrixXf::Zero(num_rows, num_columns);
    compute_winner_take_all_disparity_map(*input_p,
                                          min_disparity_margin, max_disparity_margin,
                                          max_cost_value,
                                          disparity_given_v,
                                          u_disparity_ground_obstacle_boundary,
                                          u_v_ground_obstacle_boundary,
                                          disparity_map);


    //apply_horizontal_smoothing(disparity_map, 3); // this is non-sensical, but it works

    return;
}


/// the likelihood map should have values in the range [0, 1]
/// < 0.5 indicates "bad stixel depth", > 0.5 indicates "good stixel depth"
void compute_disparity_likelihood_map(const DisparityCostVolume &cost_volume,
                                      const int minimum_disparity_margin,
                                      const int maximum_disparity_margin,
                                      const std::vector<int> &u_disparity_ground_obstacle_boundary,
                                      const std::vector<int> &u_v_ground_obstacle_boundary,
                                      Eigen::MatrixXf &disparity_likelihood_map)
{
    const int num_rows = cost_volume.rows();
    const int num_columns = cost_volume.columns();
    const int num_disparities = cost_volume.disparities();

    // FIXME hardcoded value
    //const float max_delta_cost = 50;
    //const float max_delta_cost = 250;
    const float max_delta_cost = 10;

    // FIXME hardcoded value
    //const float negative_factor = 1; // 0.5 // 0.1 // 0
    const float negative_factor = 2;

#pragma omp parallel for
    for(int v=0; v < num_rows; v+=1)
    {
        for(int u=0; u < num_columns; u+=1)
        {
            const int bottom_v = u_v_ground_obstacle_boundary[u];
            if(v > bottom_v)
            {
                // ground area
                //disparity_map(v, u) = disparity_given_v[v];
                disparity_likelihood_map(v, u) = 0;
            }
            else
            { // above the ground area

                int maximum_disparity = num_disparities;
                int minimum_disparity = 0;
                const int stixel_disparity = u_disparity_ground_obstacle_boundary[u];

                // will use_partial_depth_range if
                // maximum_disparity_margin >=0 or minimum_disparity_margin >= 0
                if(maximum_disparity_margin >= 0)
                {
                    //const int maximum_disparity_margin = 5;
                    //const int maximum_disparity = disparity_given_v[v];
                    maximum_disparity = std::min(stixel_disparity + maximum_disparity_margin, num_disparities);
                }

                if(minimum_disparity_margin >= 0)
                {
                    //const int minimum_disparity_margin = 15;
                    minimum_disparity = std::max(0, stixel_disparity - minimum_disparity_margin);
                }


                const int num_disparities_considered = (maximum_disparity - minimum_disparity);

                float disparity_likelihood = 0;
                //const float maximum_possible_mismatch = cost_volume_estimator_p->get_maximum_cost_per_pixel();

                // count how many disparity costs are higher than the one at stixel_disparity
                const float stixel_cost = cost_volume.get_costs_views()[v][u][stixel_disparity];

                for(int d=minimum_disparity; d < maximum_disparity; d+=1)
                {
                    const float &cost = cost_volume.get_costs_views()[v][u][d];

                    if(d != stixel_disparity)
                    { // in other disparities we expect the cost to be high

                        const float delta_cost = (cost - stixel_cost);
                        const float abs_delta_cost = std::min(max_delta_cost, std::abs(delta_cost));
                        const float normalized_abs_delta_cost = abs_delta_cost / max_delta_cost;
                        // normalized_delta_cost is in between [0 and 1]

                        if(cost > stixel_cost)
                        {
                            //disparity_likelihood += 1;
                            disparity_likelihood += normalized_abs_delta_cost;
                        }
                        else
                        {
                            disparity_likelihood -= negative_factor*normalized_abs_delta_cost;
                            //disparity_likelihood -= negative_factor;
                        }
                    }
                    else
                    { // d == stixel_disparity
                        continue; // we skip this case
                    }

                } // end of "for each disparity between infinity and nearest stixel"

                disparity_likelihood /= (num_disparities_considered - 1);
                disparity_likelihood = std::max(0.0f, disparity_likelihood);

                // disparity_likelihood is in [0,1], the higher the likelihood, the better
                disparity_likelihood_map(v, u) = disparity_likelihood;

            } // end of "if above the ground or not"

        } // end of "for each column"
    } // end of "for each row"

    return;
}


void StixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map(const DisparityCostVolume &cost_volume,
                                                                            Eigen::MatrixXf &disparity_likelihood_map)
{
    const int num_rows = cost_volume.rows();
    const int num_columns = cost_volume.columns();
    //const int num_disparities = cost_volume.disparities();

    // Compute the likelihood map for every pixel above the ground (using stixel constraint)
    if(cost_volume_one_p == false)
    {
        cost_volume_one_p.reset(new DisparityCostVolume());
    }

    if(cost_volume_two_p == false)
    {
        cost_volume_two_p.reset(new DisparityCostVolume());
    }

    DisparityCostVolume &cost_volume_one = *cost_volume_one_p, &cost_volume_two = *cost_volume_two_p;

    // lazy allocation
    cost_volume_one.resize(cost_volume);
    cost_volume_two.resize(cost_volume);


    const bool should_do_horizontal_averaging = horizontal_kernel_size > 1;
    const bool should_do_vertical_averaging = vertical_kernel_size > 1;

    const float max_cost_value =  this->max_cost_value; // cost_volume_estimator_p->get_maximum_cost_per_pixel();

    const DisparityCostVolume *input_p = &cost_volume;
    DisparityCostVolume *output_p = &cost_volume_one;

    // (for every pixel above the ground)
    if(should_do_horizontal_averaging)
    {
        // do horizontal averaging
        do_horizontal_averaging(horizontal_kernel_size, *input_p, max_cost_value, *output_p);
        input_p = output_p;
        output_p = &cost_volume_two;
    }

    if(should_do_vertical_averaging)
    {
        // do vertical averaging
        do_vertical_averaging(vertical_kernel_size, *input_p, max_cost_value, *output_p);
        input_p = output_p;
    }

    // find minima above the ground
    disparity_likelihood_map = Eigen::MatrixXf::Zero(num_rows, num_columns);
    doppia::compute_disparity_likelihood_map(*input_p,
                                             min_disparity_margin, max_disparity_margin,
                                             u_disparity_ground_obstacle_boundary,
                                             u_v_ground_obstacle_boundary,
                                             disparity_likelihood_map);
    return;
}


void disparity_map_to_depth_map(Eigen::MatrixXf &the_map,
                                const MetricStereoCamera &stereo_camera)
{

    const int num_rows = the_map.rows();
    const int num_columns = the_map.cols();

#pragma omp parallel for
    for(int row=0; row < num_rows; row += 1)
    {
        for(int col = 0; col < num_columns; col += 1)
        {
            const float disparity = std::max(min_float_disparity, the_map(row, col));
            the_map(row, col) = stereo_camera.disparity_to_depth(disparity);
        }
    }
    return;
}


/// depth_cost is expected to be in the range [-1, 1]
/// 1 indicates "stixel depth is good", -1 indicates "stixel depth is bad"
void depth_cost_to_stixel_height_cost(const Eigen::MatrixXf &depth_cost,
                                      const std::vector<int> &u_v_ground_obstacle_boundary,
                                      Eigen::MatrixXf &stixel_height_cost)
{
    const int num_rows = depth_cost.rows();
    const int num_columns = depth_cost.cols();

    // compute edge costs --
    //stixel_height_cost = Eigen::MatrixXf::Zero(num_rows, num_columns);
    stixel_height_cost = Eigen::MatrixXf::Constant(num_rows, num_columns, num_rows);

#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];

        stixel_height_cost(0, col) = 1; // set upper line to max value
        for(int row=1; row < (bottom_v - 1); row += 1)
        {
            const float upper_part_sum =
                    (depth_cost.col(col).segment(0, row).array() + 1).abs().sum();
            const float upper_cost = upper_part_sum;

            const float lower_part_sum =
                    (depth_cost.col(col).segment(row, bottom_v - row).array() -1).abs().sum();
            const float lower_cost = lower_part_sum;

            const float total_cost = (lower_cost + upper_cost); // /bottom_v;
            assert(total_cost >= 0);
            stixel_height_cost(row, col) = total_cost;
        } // end of "for each row between 0 and bottom v"
    } // end of "for each column"

    return;
}


void disparity_map_to_stixel_height_cost(const Eigen::MatrixXf &disparity_map,
                                         const MetricStereoCamera &stereo_camera,
                                         const GroundPlane &the_ground_plane,
                                         const std::vector<int> &u_disparity_ground_obstacle_boundary,
                                         const std::vector<int> &u_v_ground_obstacle_boundary,
                                         Eigen::MatrixXf &stixel_height_cost)
{
    const int num_rows = disparity_map.rows();
    const int num_columns = disparity_map.cols();

    Eigen::MatrixXf depth_map = disparity_map;
    disparity_map_to_depth_map(depth_map, stereo_camera);

    // FIXME hardcoded number (this should be a parameter)
    //const float distance_tolerance = 0.25; // [meters]
    //const float distance_tolerance = 1.5; // [meters]
    const float distance_tolerance = 1.0; // [meters]

    //stixel_height_cost = depth_map_meters;
    //return;

    // compute the cost of being far from expected depth --
    Eigen::MatrixXf &depth_cost = depth_map;

#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const float stixel_disparity =
                std::max<float>(min_float_disparity, u_disparity_ground_obstacle_boundary[col]);
        const float stixel_distance = stereo_camera.disparity_to_depth(stixel_disparity);

        for(int row=0; row < bottom_v; row += 1)
        {
            const float delta_distance = abs(stixel_distance - depth_map(row, col)) / distance_tolerance;
            depth_cost(row, col) = pow(2, 1 - delta_distance) - 1;
        }

        depth_cost.col(col).segment(bottom_v, num_rows - bottom_v).setConstant(0);

        const bool set_minimum_height = false;
        if(set_minimum_height)
        { // as it is, this is a bad idea.

            // fix the bottom of each stixel to be with high confidence part of the object
            const float minimum_height = 1.0; // [meters] FIXME should get this value from options
            const Eigen::Vector2f uv_point =
                    stereo_camera.get_left_camera().project_ground_plane_point(the_ground_plane,
                                                                               0, stixel_distance, minimum_height);
            int top_v = std::min(bottom_v - 1, static_cast<int>(uv_point[1]));
            top_v = std::min(num_rows - 1, std::max(0, top_v)); // keep top_v inside the image
            depth_cost.col(col).segment(top_v, bottom_v - top_v).setConstant(1);
        }

    } // end of "for each column"

    //stixel_height_cost = depth_cost;
    //return;

    // compute edge costs --
    depth_cost_to_stixel_height_cost(depth_cost, u_v_ground_obstacle_boundary, stixel_height_cost);

    return;
}


void StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost()
{
    if(use_partial_disparity_map)
    {
        compute_stixel_height_cost_using_partial_disparity_map();
    }
    else
    {
        compute_stixel_height_cost_directly();
    }

    return;
}


void StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost_using_partial_disparity_map()
{

    Eigen::MatrixXf &disparity_map = depth_map;
    //compute_disparity_map(*pixels_cost_volume_p, disparity_map);
    compute_disparity_map(*original_pixels_cost_volume_p, disparity_map);


    disparity_map_to_stixel_height_cost(disparity_map,
                                        stereo_camera,
                                        the_ground_plane,
                                        u_disparity_ground_obstacle_boundary,
                                        u_v_ground_obstacle_boundary,
                                        stixel_height_cost);

    // small post-processing
    const int smoothing_iterations = 3; //0; 3;
    apply_horizontal_smoothing(stixel_height_cost, smoothing_iterations);

    return;
}

void StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost_directly()
{

    // compute a depth likelihood score along the stixel ---
    Eigen::MatrixXf &disparity_likelihood_map = depth_map;

    // at each pixel will compute a value proportional to the likelihood
    // of the pixel having a true disparity equal to the stixel disparity map

    //compute_disparity_likelihood_map(*pixels_cost_volume_p, disparity_likelihood_map);
    compute_disparity_likelihood_map(*original_pixels_cost_volume_p, disparity_likelihood_map);

    // compute stixel height cost based on depth likelihood ---

    // disparity_likelihood_map is in range [0,1]
    // depth_cost_to_stixel_height_cost expects input in range [-1,1]

    Eigen::MatrixXf &depth_cost = disparity_likelihood_map;
    disparity_likelihood_map *= 2;
    disparity_likelihood_map.array() -= 1;
    depth_cost_to_stixel_height_cost(depth_cost, u_v_ground_obstacle_boundary, stixel_height_cost);

    // small post-processing
    // FIXME hardcoded value
    const int smoothing_iterations = 3; //0; 3;
    apply_horizontal_smoothing(stixel_height_cost, smoothing_iterations);

    return;
} // end of StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost_directly


/// Compute the heights cost matrix, without using any depth map
void StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost(const DisparityCostVolume &cost_volume)
{
    const int num_rows = cost_volume.rows();
    const int num_columns = cost_volume.columns();
    //const int num_disparities = cost_volume.disparities();

    const float max_cost_value =  this->max_cost_value; // cost_volume_estimator_p->get_maximum_cost_per_pixel();

    // reset and resize the stixel_height_cost
    stixel_height_cost = Eigen::MatrixXf::Constant(num_rows, num_columns, max_cost_value);

    for(int col = 0; col < num_columns; col += 1)
    {
        compute_stixel_height_cost_column(col, cost_volume);

    } // end of "for every column"

    const bool normalize_each_column = true;
    if(normalize_each_column)
    {
        const int top_pixels = 60; // 10, 30, 1

        for(int col = 0; col < num_columns; col += 1)
        {
            const int bottom_v = u_v_ground_obstacle_boundary[col];
            const int a = top_pixels, b = bottom_v - top_pixels;

            const float col_max = stixel_height_cost.col(col).segment(a,b).maxCoeff();
            const float col_min = stixel_height_cost.col(col).segment(a,b).minCoeff();
            stixel_height_cost.col(col).segment(a,b).array() -= col_min;
            if(col_max != col_min)
            {
                stixel_height_cost.col(col).segment(a,b) /= (col_max - col_min);
            }

            stixel_height_cost.col(col).segment(0, top_pixels).setConstant(1);
            stixel_height_cost.col(col).segment(bottom_v, num_rows - bottom_v).setConstant(1);
        }

    } // end of "if normalize_each_column"

    const bool horizontal_smoothing = false;
    if(horizontal_smoothing)
    {
        const int smoothing_iterations = 3;
        for(int i=0; i < smoothing_iterations; i+=1)
        {
            Eigen::MatrixXf t_cost = stixel_height_cost;
            for(int col = 1; col < (num_columns - 1); col += 1)
            {
                const int previous_col = col - 1;
                const int next_col = col+1;
                t_cost.col(col) += stixel_height_cost.col(previous_col);
                t_cost.col(col) += stixel_height_cost.col(next_col);
                t_cost.col(col) /= 3;
            }

            stixel_height_cost = t_cost;
        }
    }

    return;
} // end of StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost


void StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost_column(const int column_index,
                                                                             const DisparityCostVolume &cost_volume)
{

    const int num_rows = cost_volume.rows();
    //const int num_columns = cost_volume.columns();
    const int num_disparities = cost_volume.disparities();


    // pixels_cost_volume_p data is organized as y (rows), x (columns), disparity
    const int stixel_disparity = u_disparity_ground_obstacle_boundary[column_index];
    const int bottom_v = u_v_ground_obstacle_boundary[column_index];

    const float maximum_cost_value =  this->max_cost_value; // cost_volume_estimator_p->get_maximum_cost_per_pixel();

    const_data_2d_view_t rows_disparities_cost_slice =
            cost_volume.rows_disparities_slice(column_index);

    Eigen::MatrixXf rows_disparities_slice(num_rows, num_disparities);

    for(int r=0; r < num_rows; r += 1)
    {
        for(int d=0; d < num_disparities; d+=1)
        {
            rows_disparities_slice(r, d) = rows_disparities_cost_slice[r][d];
        }
    }

    for(int row=1; row < num_rows; row += 1)
    {
        const int top_pixel = 60;
        if(row < bottom_v)
        {
            float background_area_cost = 0;
            {

                const bool use_other_columns = true;
                if(use_other_columns)
                {
                    if (row < top_pixel)
                    {
                        background_area_cost = maximum_cost_value;
                        continue;
                    }

                    const int start_row = 0;
                    const int start_column = stixel_disparity - 1;
                    //const int block_rows = bottom_v - row;
                    const int block_rows = row;
                    const int block_cols = num_disparities - start_column;
                    if(start_column > 1)
                    {
                        int min_index = 0;
                        float min_disparity_cost =
                                rows_disparities_slice.block(start_row, start_column, block_rows, block_cols).colwise().sum().minCoeff(&min_index);
                        min_disparity_cost /= block_rows;

                        background_area_cost = min_disparity_cost;
                    }
                    else
                    {
                        // nothing to do
                    }
                }
                else
                { // use_other_columns == false

                    background_area_cost =
                            (
                                rows_disparities_slice.col(stixel_disparity).segment(0, row).array() - maximum_cost_value
                                ).abs().sum();
                    background_area_cost /= row;
                }

            } // end of "background_area_cost computation"

            float stixel_area_cost = 0;
            {
                stixel_area_cost =
                        rows_disparities_slice.col(stixel_disparity).segment(row, bottom_v - row).sum();
                stixel_area_cost /= (bottom_v - row);
            }


            // the lower the cost, the more likely of being part of the object
            stixel_height_cost(row, column_index) = stixel_area_cost + background_area_cost;
        }
        else
        { // row >= bottom_v
            stixel_height_cost(row, column_index) = 1;
        }

    } // end of "for each row"

    return;
} // end of StixelsEstimatorWithHeightEstimation::compute_stixel_height_cost_column


/// Apply dynamic programming over the stixels heights
void StixelsEstimatorWithHeightEstimation::compute_stixels_heights()
{
    const bool use_simplest_possible_thing = false;
    if(use_simplest_possible_thing)
    {
        compute_stixels_heights_using_local_minima(
                    stixel_height_cost,
                    u_v_ground_obstacle_boundary,
                    the_stixels);
    }
    else
    {
        compute_stixels_heights_using_dynamic_programming(
                    stixel_height_cost,
                    u_v_ground_obstacle_boundary,
                    stereo_camera,
                    the_stixels);
    }

    enforce_reasonable_stixels_heights();
    return;
} // end of StixelsEstimatorWithHeightEstimation::compute_stixels_heights


void compute_stixels_heights_using_local_minima(
    const stixel_height_cost_t &stixel_height_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    stixels_t &the_stixels)
{

    // the simplest thing that could ever work: take the maximum/minimum of each column
    // (more sofisticated version would do dynamic programming to penalize vertical jumps)

    const size_t num_columns = stixel_height_cost.cols();
    // stixel_height_cost.cols() == pixels_cost_volume_p->columns();

    assert(the_stixels.size() == num_columns);
    assert(u_v_ground_obstacle_boundary.size() == num_columns);

    const int top_limit = 10; //50;
#pragma omp parallel for
    for(size_t u = 0; u < num_columns; u += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[u];
        const int a = top_limit, b = bottom_v - top_limit;

        int max_index = 0;
        //stixel_height_cost.col(u).minCoeff(&max_index);
        stixel_height_cost.col(u).segment(a,b).minCoeff(&max_index);
        max_index += a;

        the_stixels[u].top_y = max_index;
        the_stixels[u].default_height_value = false;
    }

    return;
}

void compute_stixels_heights_using_dynamic_programming(
    const stixel_height_cost_t &stixel_height_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    const MetricStereoCamera &stereo_camera,
    stixels_t &the_stixels)
{
    //printf("PING StixelsEstimatorWithHeightEstimation::compute_stixels_heights_using_dynamic_programming\n");

    // This functions follows a logic similar to StixelsEstimator::compute_ground_obstacle_boundary()
    // but this time we use a cost function similar to the one defined in Badino et al. DAGM 2009
    // http://www.lelaps.de/papers/badino_dagm09.pdf

    //const int num_rows = stixel_height_cost.rows();
    const int num_columns = stixel_height_cost.cols();

    Eigen::MatrixXf M_cost = stixel_height_cost;

    // FIXME hardcoded parameters
    const float k1 = 1; // [scaling factor]
    const float max_distance_for_influence = 3; // [meters]

    // do left to right pass (cumulate cost) ---

    // first column is already initialized with the stixel_height_cost value

    for(int col = 1; col < num_columns; col += 1)
    {
        const int previous_col = col - 1;
        const int previous_bottom_v = u_v_ground_obstacle_boundary[previous_col];
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        const Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);

        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));

        for(int row = 10; row < bottom_v; row +=1 )
        {
            // M_cost(r, c) = stixel_height_cost + min_{rr}( M_cost(rr, c-1) + S(r,rr) )
            // S defined similar to Badino 2009, equation 5

            float min_M_plus_S = std::numeric_limits<float>::max();
            for(int rr = 10; rr < previous_bottom_v; rr +=1 )
            {
                const float distance_factor = std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));
                const float rows_factor = k1*std::abs(row - rr);
                const float cost_S = rows_factor*distance_factor;
                const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                min_M_plus_S = std::min(min_M_plus_S, M_plus_S);

            } // end of "for each row in previous stixel"

            M_cost(row, col) += min_M_plus_S;

        } // end of "for each row in current stixel"

    } // end of "for each column"

    //stixel_height_cost = M_cost; // for visualization only

    // do right to left pass  (find optimal boundary) ---

    // we find the first minimum
    int previous_r_star = 0;
    M_cost.col(num_columns -1).minCoeff(&previous_r_star);

    the_stixels[num_columns -1].top_y = previous_r_star;

    for(int col = num_columns -2; col >=0; col -= 1)
    {
        const int previous_col = col + 1;
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);
        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));

        // r_star = argmin_{r}( M(r, col) + S(previous_r_star, r) )
        int r_star = 0;
        float min_M_plus_S = std::numeric_limits<float>::max();
        for(int row = 10; row < bottom_v; row +=1 )
        {
            const float distance_factor = std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));
            const float rows_factor = k1*std::abs(row - previous_r_star);
            const float cost_S = rows_factor*distance_factor;
            const float M_plus_S = M_cost(row, col) + cost_S;

            if(M_plus_S < min_M_plus_S)
            {
                min_M_plus_S = M_plus_S;
                r_star = row;
            }
        } // end of "for each row in current stixel"

        //printf("PING the_stixels[%i].top_y = %i\n", col, r_star);
        current_stixel.top_y = r_star;
        current_stixel.default_height_value = false;
        previous_r_star = r_star;

    } // end of "for each column"

    return;
} // end of StixelsEstimatorWithHeightEstimation::compute_stixels_heights_using_dynamic_programming

void StixelsEstimatorWithHeightEstimation::enforce_reasonable_stixels_heights()
{
    stixels_height_post_processing_p->operator ()(
                expected_v_given_disparity,
                the_ground_plane,
                the_stixels);
    return;
} // end of StixelsEstimatorWithHeightEstimation::enforce_reasonable_stixels_heights



} // end of namespace doppia
