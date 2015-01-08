
// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include <boost/gil/gil_all.hpp>

#include "GroundEstimationApplication.hpp"
#include "GroundEstimationGui.hpp"
#include "applications/EmptyGui.hpp"

#include "video_input/VideoInputFactory.hpp"
#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"
#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"
#include "image_processing/IrlsLinesDetector.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/any_to_string.hpp"
#include "helpers/for_each.hpp"

#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>

#include <omp.h>

#include <iostream>
#include <fstream>
#include <cstdlib>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "GroundEstimationApplication");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "GroundEstimationApplication");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "GroundEstimationApplication");
}

} // end of anonymous namespace

namespace doppia
{


using logging::log;
using namespace std;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string GroundEstimationApplication::get_application_title()
{
    return  "Ground estimation. Rodrigo Benenson @ KULeuven. 2010-2011.";
}

GroundEstimationApplication::GroundEstimationApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}



GroundEstimationApplication::~GroundEstimationApplication()
{
    // nothing to do here
    return;
}


program_options::options_description GroundEstimationApplication::get_options_description()
{
    program_options::options_description desc("GroundEstimationApplication options");

    const std::string application_name = "stixel_world";
    BaseApplication::add_args_options(desc, application_name);

    // GroundEstimationApplication has no particular options,
    // it only uses GroundEstimationGui and the factories options


    desc.add_options()
            ("estimation_method",
             program_options::value<string>()->default_value("fast_ground_plane"),
             "select the ground estimation method: ground_plane, fast_ground_plane and b_spline")

            ("gui.disabled",
             program_options::value<bool>()->default_value(false),
             "if true, no user interface will be presented")
            ;

    return desc;
}

void GroundEstimationApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(GroundEstimationApplication::get_options_description());
    desc.add(GroundEstimationGui::get_args_options());
    desc.add(VideoInputFactory::get_args_options());


    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(DisparityCostVolumeEstimatorFactory::get_args_options());

    desc.add(BaseGroundPlaneEstimator::get_args_options());
    desc.add(GroundPlaneEstimator::get_args_options());
    desc.add(FastGroundPlaneEstimator::get_args_options());
    desc.add(IrlsLinesDetector::get_args_options());

    return;
}


/// helper method called by setup_problem
void GroundEstimationApplication::setup_logging(std::ofstream &log_file, const program_options::variables_map &options)
{

    if(log_file.is_open())
    {
        // the logging is already setup
        return;
    }

    // set base logging rules --
    BaseApplication::setup_logging(log_file, options);

    // set our own stdout rules --
    logging::LogRuleSet &rules_for_stdout = logging::get_log().console_log().rule_set();
    //rules_for_stdout.add_rule(logging::InfoMessage, "GroundEstimationApplication");
    rules_for_stdout.add_rule(logging::InfoMessage, "*");

    return;
}

void GroundEstimationApplication::setup_problem(const program_options::variables_map &options)
{
    // instanciate the different processing modules ---

    // video input --
    video_input_p.reset(VideoInputFactory::new_instance(options));
    AbstractVideoInput &video_input = *video_input_p;
    const AbstractVideoInput::dimensions_t &input_dimensions = video_input.get_left_image().dimensions();
    const MetricStereoCamera &camera = video_input.get_metric_camera();

    ground_plane_prior.set_from_metric_units(video_input.camera_pitch, video_input.camera_roll, video_input.camera_height);
    current_ground_plane_estimate = ground_plane_prior;

    // estimate prior ground horizon estimate ---
    const float far_far_away = 1E3; // [meters]
    const float x = 0, height = 0; // [meters]
    const Eigen::Vector2f uv_point =
            camera.get_left_camera().project_ground_plane_point(ground_plane_prior, x, far_far_away, height);
    const int horizon_row = uv_point[1];
    const std::vector<int> ground_object_boundary_prior(input_dimensions.x, horizon_row);

    // create ground plane estimator --
    const string estimation_method = get_option_value<string>(options, "estimation_method");

    if(estimation_method.compare("ground_plane") == 0)
    {
        // cost volume estimator --
        pixels_matching_cost_volume_p.reset(new DisparityCostVolume());
        cost_volume_estimator_p.reset(DisparityCostVolumeEstimatorFactory::new_instance(options));

        // ground plane estimator --

        //const bool preprocess_residual = get_option_value<bool>(options, "preprocess.residual");
        //should_compute_residual = preprocess_residual == false;
        //const bool cost_volume_is_from_residual_image = preprocess_residual;
        //const bool cost_volume_is_from_residual_image = true; // FIXME hardcoded value
        const bool cost_volume_is_from_residual_image = false; // FIXME hardcoded value
        ground_plane_estimator_p.reset(new GroundPlaneEstimator(
                                           options, camera.get_calibration(),
                                           cost_volume_is_from_residual_image));

        ground_plane_estimator_p->set_ground_area_prior(ground_object_boundary_prior);
    }
    else if(estimation_method.compare("fast_ground_plane") == 0)
    {
        fast_ground_plane_estimator_p.reset(new FastGroundPlaneEstimator(
                                                options, camera.get_calibration()));

        fast_ground_plane_estimator_p->set_ground_area_prior(ground_object_boundary_prior);
    }
    else if (estimation_method.compare("b_spline") == 0)
    {
        throw std::runtime_error("b_spline method not (yet) implemented");
    }
    else
    {
        throw std::invalid_argument("Unknown 'estimation_method' value");
    }

    // parse the application specific options --
    //this->should_save_stixels = get_option_value<bool>(options, "save_stixels");

    return;
}


AbstractGui* GroundEstimationApplication::create_gui(const program_options::variables_map &options)
{
    const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    AbstractGui *gui_p=NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        gui_p = new GroundEstimationGui(*this, options);
    }

    return gui_p;
}

int GroundEstimationApplication::get_current_frame_number() const
{
    return this->video_input_p->get_current_frame_number();
}

void GroundEstimationApplication::main_loop()
{

    int num_iterations = 0;
    const int num_iterations_for_timing = 10;
    double start_wall_time = omp_get_wtime();
    bool video_input_is_available = video_input_p->next_frame();
    bool end_of_game = false;
    while(video_input_is_available and (not end_of_game))
    {

        // update video input --
        AbstractStereoMatcher::input_image_view_t
                input_left_view(video_input_p->get_left_image()),
                input_right_view(video_input_p->get_right_image());

        // dirty trick to work around the video input BUG
        // cast to color view
        boost::gil::rgb8c_view_t
                left_view(input_left_view._dynamic_cast<boost::gil::rgb8c_view_t>()),
                right_view(input_right_view._dynamic_cast<boost::gil::rgb8c_view_t>());

        //boost::gil::gray8c_view_t
        //        left_gray_view(boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(left_view)),
        //        right_gray_view(boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(right_view));

        // estimate ground plane --
        if(ground_plane_estimator_p)
        {
            // compute the disparity cost volume --
            cost_volume_estimator_p->compute(left_view, right_view,
                                             *pixels_matching_cost_volume_p);

            // estimate the ground plane --
            ground_plane_estimator_p->set_ground_plane_prior(ground_plane_prior);
            ground_plane_estimator_p->set_ground_disparity_cost_volume(pixels_matching_cost_volume_p);


            ground_plane_estimator_p->compute();
            current_ground_plane_estimate = ground_plane_estimator_p->get_ground_plane();
        }
        else if(fast_ground_plane_estimator_p)
        {

            // estimate the ground plane --
            //fast_ground_plane_estimator_p->set_ground_plane_prior(current_ground_plane_estimate);
            fast_ground_plane_estimator_p->set_ground_plane_prior(ground_plane_prior);
            fast_ground_plane_estimator_p->set_rectified_images_pair(left_view, right_view);

            fast_ground_plane_estimator_p->compute();
            current_ground_plane_estimate = fast_ground_plane_estimator_p->get_ground_plane();

        }
        else
        {
            throw std::runtime_error("No ground plane estimator was instanciated");
        }


        // do semantic labeling and objects detections in stixels world --
        // (not yet implemented)

        // update user interface --
        end_of_game = update_gui();

        num_iterations += 1;
        if((num_iterations % num_iterations_for_timing) == 0)
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        // retrieve next rectified input stereo pair
        video_input_is_available = video_input_p->next_frame();
    }

    printf("Processed a total of %i input frames\n", num_iterations);

    return;
}



} // end of namespace doppia

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
