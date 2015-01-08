#include "FastStixelWorldEstimator.hpp"

#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "FastStixelsEstimator.hpp"
#include "FastStixelsEstimatorWithHeightEstimation.hpp"
#include "ImagePlaneStixelsEstimator.hpp"
#include "StixelWorldEstimator.hpp" // for helper methods

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <omp.h>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "FastStixelWorldEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "FastStixelWorldEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "FastStixelWorldEstimator");
}

} // end of anonymous namespace



namespace doppia {

using namespace std;
using namespace boost;

program_options::options_description FastStixelWorldEstimator::get_args_options()
{
    program_options::options_description desc("FastStixelWorldEstimator options");

    desc.add_options()

            /// will use same options at StixelWorldEstimator

            ;


    return desc;

}


FastStixelWorldEstimator::FastStixelWorldEstimator(
        const boost::program_options::variables_map &options,
        const AbstractVideoInput::dimensions_t &input_dimensions_,
        const MetricStereoCamera &camera_,
        const GroundPlane &ground_plane_prior_)
    :
      input_dimensions(input_dimensions_),
      camera(camera_),
      camera_calibration(camera_.get_calibration()),
      ground_plane_prior(ground_plane_prior_),
      expected_object_height(get_option_value<float>(options, "stixel_world.expected_object_height")),
      minimum_object_height_in_pixels(get_option_value<int>(options, "stixel_world.minimum_object_height_in_pixels"))
{

    silent_mode = true;
    if(options.count("silent_mode"))
    {
        silent_mode = get_option_value<bool>(options, "silent_mode");
    }

    ground_plane_estimator_p.reset(new FastGroundPlaneEstimator(
                                       options, camera.get_calibration()));

    {
        // estimate prior ground horizon estimate ---
        const float far_far_away = 1E3; // [meters]
        const float x = 0, height = 0; // [meters]
        const Eigen::Vector2f uv_point =
                camera.get_left_camera().project_ground_plane_point(ground_plane_prior, x, far_far_away, height);
        const int horizon_row = std::min<int>(std::max(0.0f, uv_point[1]), input_dimensions.y - 1);
        const std::vector<int> ground_object_boundary_prior(input_dimensions.x, horizon_row);

        ground_plane_estimator_p->set_ground_area_prior(ground_object_boundary_prior);
    }

    const int stixel_width = get_option_value<int>(options, "stixel_world.stixel_width");
    if(stixel_width < 1)
    {
        throw std::runtime_error("stixel_world.stixels_width must be >= 1 pixels");
    }

    const string method = get_option_value<string>(options, "stixel_world.method");

    if(method.compare("fast") == 0)
    {
        const string height_method =
                get_option_value<string>(options, "stixel_world.height_method");

        if(height_method.empty() or height_method.compare("fixed") == 0)
        {
            stixels_estimator_p.reset(new FastStixelsEstimator(options,
                                                               camera,
                                                               expected_object_height,
                                                               minimum_object_height_in_pixels,
                                                               stixel_width));
        }
        else if(height_method.compare("3d_cost") == 0)
        {
            throw std::invalid_argument("FastStixelWorldEstimator does not support stixel_world.height_method == '3d_cost'");
        }
        else if(height_method.compare("two_steps") == 0)
        {

            stixels_estimator_p.reset(new FastStixelsEstimatorWithHeightEstimation(options,
                                                                                   camera,
                                                                                   expected_object_height,
                                                                                   minimum_object_height_in_pixels,
                                                                                   stixel_width));
        }
        else
        {
            log_error() << "Received unknown stixel_world.height_method value: " << height_method << std::endl;
            throw std::invalid_argument("FastStixelWorldEstimator::FastStixelWorldEstimator received a unknown "
                                        "'stixel_world.height_method' value");
        }

    }
    else if (method.compare("fast_uv") == 0)
    {
        stixels_estimator_p.reset(new ImagePlaneStixelsEstimator(options,
                                                                 camera,
                                                                 expected_object_height,
                                                                 minimum_object_height_in_pixels,
                                                                 stixel_width));
    }
    else
    {
        log_error() << "Received unknown stixel_world.method value: " << method << std::endl;
        throw std::invalid_argument("FastStixelWorldEstimator::FastStixelWorldEstimator received a unknown "
                                    "'stixel_world.method' value");
    }

    return;
}


FastStixelWorldEstimator::~FastStixelWorldEstimator()
{
    // nothing to do here
    return;
}


const GroundPlane &FastStixelWorldEstimator::get_ground_plane() const
{
    return this->ground_plane_estimator_p->get_ground_plane();
}


const stixels_t &FastStixelWorldEstimator::get_stixels() const
{
    return this->stixels_estimator_p->get_stixels();
}


const std::vector< int > &FastStixelWorldEstimator::get_ground_plane_corridor()
{
    const GroundPlane &ground_plane = get_ground_plane();

    const BaseStixelsEstimator *base_stixels_estimator_p = dynamic_cast<BaseStixelsEstimator *>(stixels_estimator_p.get());

    if(base_stixels_estimator_p)
    {
        const std::vector<int> &disparity_given_v = base_stixels_estimator_p->get_disparity_given_v();
        compute_ground_plane_corridor(ground_plane, disparity_given_v, camera,
                                      expected_object_height, minimum_object_height_in_pixels,
                                      ground_plane_corridor);
    }
    else
    {
        throw std::runtime_error("FastStixelWorldEstimator::get_ground_plane_corridor "
                                 "can only be called when a children of BaseStixelsEstimator is being used.");
    }
    return ground_plane_corridor;
}


int FastStixelWorldEstimator::get_stixel_width() const
{
    const FastStixelsEstimator *fast_estimator_p = dynamic_cast<FastStixelsEstimator *>(stixels_estimator_p.get());

    if(fast_estimator_p)
    {
        return fast_estimator_p->get_stixel_width();
    }
    // else

    return 1;
}


void FastStixelWorldEstimator::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();


    // estimate the ground plane ---
    if(num_iterations == 0)
    {
        ground_plane_estimator_p->set_ground_plane_prior(ground_plane_prior);
    }
    else
    {
        //fast_ground_plane_estimator_p->set_ground_plane_prior(current_ground_plane_estimate);
        ground_plane_estimator_p->set_ground_plane_prior(ground_plane_estimator_p->get_ground_plane());
        //ground_plane_estimator_p->set_ground_plane_prior(ground_plane_prior);
    }

    ground_plane_estimator_p->set_rectified_images_pair(input_left_view, input_right_view);

    ground_plane_estimator_p->compute();
    const GroundPlane &current_ground_plane_estimate = ground_plane_estimator_p->get_ground_plane();

    // estimate the stixels ---
    stixels_estimator_p->set_rectified_images_pair(input_left_view, input_right_view);
    stixels_estimator_p->set_ground_plane_estimate(
                current_ground_plane_estimate,
                ground_plane_estimator_p->get_ground_v_disparity_line() );
    stixels_estimator_p->compute();


    // close the loop between stixels estimation and ground plane estimation ---
    //if(use_stixels_for_ground_estimation)
    //{
    //    ground_plane_estimator_p->set_ground_area_prior( stixels_estimator_p->get_u_v_ground_obstacle_boundary() );
    //}


    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;

    if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
    {
        printf("FastStixelWorldEstimator::compute speed \033[36m%.2lf [Hz]\033[0m (average in the last %i iterations)\n",
               num_iterations / cumulated_time, num_iterations );
    }

    return;
}


} // end of namespace doppia
