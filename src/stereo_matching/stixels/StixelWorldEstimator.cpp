#include "StixelWorldEstimator.hpp"

#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"
#include "stereo_matching/cost_volume/AbstractDisparityCostVolumeEstimator.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "StixelsEstimator.hpp"
#include "StixelsEstimatorWith3dCost.hpp"
#include "StixelsEstimatorWithHeightEstimation.hpp"

#include "video_input/preprocessing/AbstractPreprocessor.hpp"
#include "video_input/preprocessing/CpuPreprocessor.hpp"
#include "video_input/preprocessing/FastReverseMapper.hpp"
#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include "Eigen/Geometry"
#include "Eigen/LU"

#include <boost/gil/extension/numeric/sampler.hpp>

#include <omp.h>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelWorldEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelWorldEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelWorldEstimator");
}

} // end of anonymous namespace



namespace doppia {

using namespace std;
using namespace boost;

program_options::options_description StixelWorldEstimator::get_args_options()
{
    program_options::options_description desc("StixelWorldEstimator options");

    desc.add_options()

            ("stixel_world.stixel_width",
             program_options::value<int>()->default_value(1),
             "width of each stixel, in pixels. This defines the spacing between the stixels, "
             "but not how many columns are used as support for the estimation, see stixel_support_width")

            ("stixel_world.stixel_support_width",
             program_options::value<int>()->default_value(2),
             "how many columns of the image are used to estimate the stixels? "
             "This value is indepent of the stixel width. Values between 1 and 5 are reasonable.")

            ("stixel_world.expected_object_height",
             program_options::value<float>()->default_value(1.8),
             "expected height of the objects above the ground in [meters]. "
             "If set to a negative value, all of upper part of the image will be used")

            ("stixel_world.height_method",
             program_options::value<string>()->default_value("fixed"),
             "fixed: use the default height\n"
             "3d_cost: directly estimate the height of the stixels using a 3d cost volume\n"
             "two_steps: use a two steps approach to estimating the height")

            ("stixel_world.num_height_levels",
             program_options::value<int>()->default_value(3),
             "number of height levels used in the 3d_cost method. "
             "If zero or one only the expected objects height is used, "
             "if more than two levels the object height is split in (levels - 1) equal partitions, "
             "and the last level is everything above")

            ("stixel_world.use_stixels_for_ground_estimation",
             program_options::value<bool>()->default_value(true),
             "use a feedback loop between the stixels estimation and ground estimation. "
             "If true, instead of the a priori horizon, only the area below the stixels are used for ground estimation")

            ("stixel_world.minimum_object_height_in_pixels",
             program_options::value<int>()->default_value(30),
             "minimum height of the objects in the image, in [pixels]")
            ;


    return desc;

}


StixelWorldEstimator::StixelWorldEstimator(const boost::program_options::variables_map &options,
                                           const AbstractVideoInput::dimensions_t &input_dimensions_,
                                           const MetricStereoCamera &camera_,
                                           const shared_ptr<AbstractPreprocessor> preprocessor_p_,
                                           const GroundPlane &ground_plane_prior_)
    :
      input_dimensions(input_dimensions_),
      camera(camera_),
      camera_calibration(camera_.get_calibration()),
      ground_plane_prior(ground_plane_prior_)
{

    const int stixel_width = get_option_value<int>(options, "stixel_world.stixel_width");
    if(stixel_width < 1)
    {
        throw std::runtime_error("stixel_world.stixels_width must be >= 1 pixels");
    }

    const bool preprocess_residual = get_option_value<bool>(options, "preprocess.residual");
    should_compute_residual = preprocess_residual == false;

    use_stixels_for_ground_estimation = get_option_value<bool>(options, "stixel_world.use_stixels_for_ground_estimation");

    {
        //const bool cost_volume_is_from_residual_image = preprocess_residual;
        //const bool cost_volume_is_from_residual_image = true; // FIXME hardcoded value
        const bool cost_volume_is_from_residual_image = false; // FIXME hardcoded value
        ground_plane_estimator_p.reset(new GroundPlaneEstimator(
                                           options,
                                           camera_calibration,
                                           cost_volume_is_from_residual_image));

        // estimate prior ground horizon estimate ---
        const float far_far_away = 1E3; // [meters]
        const float x = 0, height = 0; // [meters]
        const Eigen::Vector2f uv_point =
                camera.get_left_camera().project_ground_plane_point(ground_plane_prior, x, far_far_away, height);
        const int horizon_row = std::min<int>(std::max(0.0f, uv_point[1]), input_dimensions.y - 1);
        const std::vector<int> ground_object_boundary_prior(input_dimensions.x, horizon_row);
        ground_plane_estimator_p->set_ground_area_prior(ground_object_boundary_prior);
    }


    preprocessor_p = boost::dynamic_pointer_cast<CpuPreprocessor>(preprocessor_p_);

    if(preprocessor_p == false)
    {
        throw std::invalid_argument("StixelWorldEstimator requires to receive an CpuPreprocessor instance");
    }

    expected_object_height =
            get_option_value<float>(options, "stixel_world.expected_object_height");
    minimum_object_height_in_pixels =
            get_option_value<int>(options, "stixel_world.minimum_object_height_in_pixels");
    const string height_method =
            get_option_value<string>(options, "stixel_world.height_method");
    const int num_height_levels =
            get_option_value<int>(options, "stixel_world.num_height_levels");

    if(height_method.empty() or height_method.compare("fixed") == 0)
    {
        stixels_estimator_p.reset(new StixelsEstimator(options,
                                                       camera,
                                                       expected_object_height,
                                                       minimum_object_height_in_pixels,
                                                       stixel_width));
    }
    else if(height_method.compare("3d_cost") == 0)
    {
        stixels_estimator_p.reset(new StixelsEstimatorWith3dCost(options,
                                                                 camera,
                                                                 expected_object_height,
                                                                 minimum_object_height_in_pixels,
                                                                 num_height_levels,
                                                                 stixel_width));
    }
    else if(height_method.compare("two_steps") == 0)
    {
        stixels_estimator_p.reset(new StixelsEstimatorWithHeightEstimation(options,
                                                                           camera,
                                                                           preprocessor_p,
                                                                           expected_object_height,
                                                                           minimum_object_height_in_pixels,
                                                                           stixel_width));
    }
    else
    {
        log_error() << "Received unknown stixel_world.height_method value: " << height_method << std::endl;
        throw std::invalid_argument("Unknown 'stixel_world.height_method' value");
    }


    pixels_matching_cost_volume_p.reset(new DisparityCostVolume());
    residual_pixels_matching_cost_volume_p.reset(new DisparityCostVolume());

    cost_volume_estimator_p.reset(DisparityCostVolumeEstimatorFactory::new_instance(options));

    return;
}

StixelWorldEstimator::~StixelWorldEstimator()
{
    // nothing to do here
    return;
}

void StixelWorldEstimator::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();

    input_image_const_view_t input_left_residual_view, input_right_residual_view;
    //if(false and should_compute_residual)
    if(false and should_compute_residual)
    {
        assert(preprocessor_p);

        input_left_residual_image.recreate(input_left_view.dimensions());
        input_right_residual_image.recreate(input_right_view.dimensions());

        input_image_view_t left_view = boost::gil::view(input_left_residual_image);
        preprocessor_p->compute_residual(input_left_view, left_view);

        input_image_view_t right_view = boost::gil::view(input_right_residual_image);
        preprocessor_p->compute_residual(input_right_view, right_view);

        input_left_residual_view = boost::gil::const_view(input_left_residual_image);
        input_right_residual_view = boost::gil::const_view(input_right_residual_image);

        cost_volume_estimator_p->compute(input_left_residual_view,
                                         input_right_residual_view,
                                         *residual_pixels_matching_cost_volume_p);
    }
    else
    {
        input_left_residual_view = input_left_view;
        input_right_residual_view = input_right_view;
        residual_pixels_matching_cost_volume_p = pixels_matching_cost_volume_p;
    }

    // compute the disparity cost volume ---
    cost_volume_estimator_p->compute(input_left_view,
                                     input_right_view,
                                     *pixels_matching_cost_volume_p);

    // estimate the ground plane ---
    ground_plane_estimator_p->set_ground_plane_prior(ground_plane_prior);
    ground_plane_estimator_p->set_ground_disparity_cost_volume(
                residual_pixels_matching_cost_volume_p);

    ground_plane_estimator_p->compute();
    const GroundPlane &current_ground_plane_estimate = ground_plane_estimator_p->get_ground_plane();

    // estimate the stixels ---
    stixels_estimator_p->set_disparity_cost_volume(pixels_matching_cost_volume_p, cost_volume_estimator_p->get_maximum_cost_per_pixel());
    stixels_estimator_p->set_rectified_images_pair(input_left_view, input_right_view);
    stixels_estimator_p->set_ground_plane_estimate(
                current_ground_plane_estimate,
                ground_plane_estimator_p->get_ground_v_disparity_line() );
    stixels_estimator_p->compute();


    // close the loop between stixels estimation and ground plane estimation ---
    if(use_stixels_for_ground_estimation)
    {
        ground_plane_estimator_p->set_ground_area_prior( stixels_estimator_p->get_u_v_ground_obstacle_boundary() );
    }


    // timing ---
    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;

    if((num_iterations % num_iterations_for_timing) == 0)
    {
        printf("Average StixelWorldEstimator::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
        cumulated_time = 0;
    }

    return;
}

const GroundPlane &StixelWorldEstimator::get_ground_plane() const
{
    return this->ground_plane_estimator_p->get_ground_plane();
}

const stixels_t &StixelWorldEstimator::get_stixels() const
{
    return this->stixels_estimator_p->get_stixels();
}

int StixelWorldEstimator::get_stixel_width() const
{
    return stixels_estimator_p->get_stixel_width();
}


void compute_ground_plane_corridor(
    const GroundPlane &ground_plane,
    const std::vector<int> &disparity_given_v,
    const MetricStereoCamera&  camera,
    const float expected_object_height,
    const int minimum_object_height_in_pixels,
    AbstractStixelWorldEstimator::ground_plane_corridor_t &ground_plane_corridor)
{

    //assert(disparity_given_v.empty() == false);
    if(disparity_given_v.empty())
    {
        throw std::runtime_error("compute_ground_plane_corridor received an empty disparity_given_v");
    }


    const int num_rows = static_cast<int>(disparity_given_v.size());
    const MetricCamera &left_camera = camera.get_left_camera();

    // we initialize with -1
    ground_plane_corridor.resize(num_rows);

    for(int v=0; v < num_rows; v+=1 )
    {
        const int &disparity = disparity_given_v[v];
        if(disparity <= 0)
        {
            // we do not consider objects very very far away
            ground_plane_corridor[v] = -1;
            continue;
        }
        else
        { // disparity > 0

            const int bottom_y = v;

            const float depth = camera.disparity_to_depth(disparity);
            Eigen::Vector2f xy = left_camera.project_ground_plane_point(ground_plane, 0, depth, expected_object_height);
            const int object_height_in_pixels = bottom_y - xy(1);

            //printf("object_height_in_pixels == %i\n", object_height_in_pixels);
            assert(object_height_in_pixels >= 0);

            const int top_y = std::max(0, bottom_y - std::max(object_height_in_pixels, minimum_object_height_in_pixels));

            assert(top_y < bottom_y);
            ground_plane_corridor[bottom_y] = top_y;
        }

    } // end of "for each row"

    return;
}


const std::vector< int > &StixelWorldEstimator::get_ground_plane_corridor()
{
    const GroundPlane &ground_plane = get_ground_plane();
    const std::vector<int> &disparity_given_v = stixels_estimator_p->get_disparity_given_v();    
    compute_ground_plane_corridor(ground_plane, disparity_given_v, camera,
                                  expected_object_height, minimum_object_height_in_pixels,
                                  ground_plane_corridor);

    return ground_plane_corridor;
}



} // end of namespace doppia
