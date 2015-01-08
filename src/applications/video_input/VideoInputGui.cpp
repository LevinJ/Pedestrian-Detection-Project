#include "VideoInputGui.hpp"

#include "SDL/SDL_keysym.h"

#include "VideoInputApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include "helpers/Log.hpp"

#include <boost/bind.hpp>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "VideoInputGui");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "VideoInputGui");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "VideoInputGui");
}

} // end of anonymous namespace


namespace doppia
{


program_options::options_description VideoInputGui::get_args_options()
{
    program_options::options_description desc("VideoInputGui options");

    // no particular input options other than the base ones
    BaseSdlGui::add_args_options(desc);

    return desc;
}


VideoInputGui::VideoInputGui(VideoInputApplication &_application, const program_options::variables_map &options)
    :BaseSdlGui(_application, options), application(_application)
{

    if(application.video_input_p.get() == NULL)
    {

        throw std::runtime_error("VideoInputGui constructor expects that video_input is already initialized");
    }

    // create the application window --
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        const int input_width = left_input_view.width();
        const int input_height = left_input_view.height();

        BaseSdlGui::init_gui(application.get_application_title(), input_width, input_height);

    }

    // populate the views map --
    views_map[SDLK_1] = view_t(boost::bind(&VideoInputGui::draw_video_input, this), "draw_video_input");


    // draw the first image --
    draw_video_input();

    // set the initial view mode --
    current_view = views_map[SDLK_1];

    return;
}

VideoInputGui::~VideoInputGui()
{
    // nothing to do here
    return;
}



void VideoInputGui::draw_video_input()
{
    const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();

    boost::gil::copy_and_convert_pixels(left_input_view, screen_left_view);
    boost::gil::copy_and_convert_pixels(right_input_view, screen_right_view);

    blit_to_screen();

    return;
}


} // end of namespace doppia
