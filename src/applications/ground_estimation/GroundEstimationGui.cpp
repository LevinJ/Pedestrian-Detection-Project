#include "GroundEstimationGui.hpp"

#include "GroundEstimationApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/draw_matrix.hpp"
#include "drawing/gil/draw_the_ground_corridor.hpp"

#include "helpers/ModuleLog.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/for_each.hpp"
#include "helpers/xyz_indices.hpp"

#include <boost/format.hpp>

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include "SDL/SDL.h"

#include <opencv2/core/core.hpp>
#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"


namespace doppia
{

MODULE_LOG_MACRO("GroundEstimationGui")

using boost::array;
using namespace boost::gil;

program_options::options_description GroundEstimationGui::get_args_options()
{
    program_options::options_description desc("GroundEstimationGui options");

    // specific options --
    desc.add_options()

            /*("gui.colorize_disparity",
                                                     program_options::value<bool>()->default_value(true),
                                                     "colorize the disparity map or draw it using grayscale")*/

            ;

    // add base options --
    BaseSdlGui::add_args_options(desc);

    return desc;
}


GroundEstimationGui::GroundEstimationGui(GroundEstimationApplication &_application,
                                         const program_options::variables_map &options)
    :BaseSdlGui(_application, options), application(_application)
{

    if(application.video_input_p == false)
    {
        throw std::runtime_error("GroundEstimationGui constructor expects that video_input is already initialized");
    }

    if((!application.ground_plane_estimator_p) and
            (!application.fast_ground_plane_estimator_p))
    {
        throw std::runtime_error("GroundEstimationGui constructor expects that ground_plane_estimator_p or  and fast_ground_plane_estimator_p is already initialized");
    }


    // retrieve program options --
    max_disparity = get_option_value<int>(options, "max_disparity");

    // create the application window --
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        const int input_width = left_input_view.width();
        const int input_height = left_input_view.height();

        BaseSdlGui::init_gui(application.get_application_title(), input_width, input_height);
    }

    // populate the views map --
    views_map[SDLK_1] = view_t(boost::bind(&GroundEstimationGui::draw_video_input, this), "draw_video_input");
    views_map[SDLK_2] = view_t(boost::bind(&GroundEstimationGui::draw_ground_plane_estimation, this), "draw_ground_plane_estimation");

    // draw the first image --
    draw_video_input();

    // set the initial view mode --
    current_view = views_map[SDLK_2];

    return;
}


GroundEstimationGui::~GroundEstimationGui()
{
    // nothing to do here
    return;
}


void GroundEstimationGui::draw_video_input()
{
    const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();

    copy_and_convert_pixels(left_input_view, screen_left_view);
    copy_and_convert_pixels(right_input_view, screen_right_view);

    return;
}


void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self)
{
    draw_ground_plane_estimator(ground_plane_estimator,
                                video_input.get_right_image(),
                                video_input.get_stereo_calibration(),
                                self.screen_right_view);
    return;
}


void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self)
{
    draw_ground_plane_estimator(ground_plane_estimator,
                                video_input.get_right_image(),
                                video_input.get_stereo_calibration(),
                                self.screen_right_view);
    return;
}


void GroundEstimationGui::draw_ground_plane_estimation()
{

    const BaseGroundPlaneEstimator *ground_plane_estimator_p = NULL;
    if(application.ground_plane_estimator_p)
    {
        ground_plane_estimator_p = application.ground_plane_estimator_p.get();
    }
    else
    {
        ground_plane_estimator_p = application.fast_ground_plane_estimator_p.get();
    }

    // Left screen --
    {
        // copy left screen image ---
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        copy_and_convert_pixels(left_input_view, screen_left_view);

        if(application.fast_ground_plane_estimator_p and
                application.fast_ground_plane_estimator_p->is_computing_residual_image())
        {
            using namespace boost::gil;

            rgb8_view_t screen_subview = subimage_view(screen_left_view,
                                                       0, screen_left_view.height()/2,
                                                       screen_left_view.width(),
                                                       screen_left_view.height()/2);

            const rgb8c_view_t &left_half_view = application.fast_ground_plane_estimator_p->get_left_half_view();
            copy_pixels(left_half_view, screen_subview);
        }

        // add the ground bottom and top corridor ---
        const GroundPlane &ground_plane = ground_plane_estimator_p->get_ground_plane();
        const MetricCamera &camera = application.video_input_p->get_metric_camera().get_left_camera();
        draw_the_ground_corridor(screen_left_view, camera, ground_plane);

        // add our prior on the ground area ---
        const bool draw_prior_on_ground_area = true;
        if(draw_prior_on_ground_area)
        {
            const std::vector<int> &ground_object_boundary_prior =
                    ground_plane_estimator_p->get_ground_area_prior();
            const std::vector<int> &boundary = ground_object_boundary_prior;
            for(std::size_t u=0; u < boundary.size(); u+=1)
            {
                const int &disparity = boundary[u];
                screen_left_view(u, disparity) = rgb8_colors::yellow;
            }
        }

    } // end of left screen -

    // Right screen --
    {
        if(application.ground_plane_estimator_p)
        {
            // will draw on the right screen
            draw_ground_plane_estimator(*(application.ground_plane_estimator_p),
                                        *application.video_input_p, *this);
        }
        else
        {
            // will draw on the right screen
            draw_ground_plane_estimator(*(application.fast_ground_plane_estimator_p),
                                        *application.video_input_p, *this);
        }
    }


    return;
} // end of GroundEstimationGui::draw_ground_plane_estimation


} // end of namespace doppia




