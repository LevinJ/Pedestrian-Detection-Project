
// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>

#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/format.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

#include "VideoInputApplication.hpp"
#include "VideoInputGui.hpp"
#include "applications/EmptyGui.hpp"

#include "video_input/VideoInputFactory.hpp"
#include "stereo_matching/StereoMatcherFactory.hpp"
#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/any_to_string.hpp"
#include "helpers/for_each.hpp"

#include <omp.h>


namespace doppia
{


using logging::log;
using namespace std;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string VideoInputApplication::get_application_title()
{
    return  "Video input. Rodrigo Benenson, Andreas Ess @ KULeuven. 2009-2010.";
}

VideoInputApplication::VideoInputApplication()
    : BaseApplication()
{
    // nothing to do here

    return;
}


VideoInputApplication::~VideoInputApplication()
{
    // nothing to do here

    return;
}


program_options::options_description VideoInputApplication::get_options_description()
{
    program_options::options_description desc("VideoInputApplication options");

    const std::string application_name = "video_input";
    BaseApplication::add_args_options(desc, application_name);

    desc.add_options()
            ("gui.disabled",
             program_options::value<bool>()->default_value(false),
             "if true, no user interface will be presented")
            ;

    return desc;
}

void VideoInputApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(VideoInputApplication::get_options_description());
    desc.add(VideoInputGui::get_args_options());
    desc.add(VideoInputFactory::get_args_options());

    return;
}


/// helper method called by setup_problem
void VideoInputApplication::setup_logging(std::ofstream &log_file, const program_options::variables_map &options)
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
    rules_for_stdout.add_rule(logging::InfoMessage, "VideoInputApplication");

    return;
}

void VideoInputApplication::setup_problem(const program_options::variables_map &options)
{
    // instanciate the different processing modules
    video_input_p.reset(VideoInputFactory::new_instance(options));

    return;
}


AbstractGui* VideoInputApplication::create_gui(const program_options::variables_map &options)
{
    const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    AbstractGui *gui_p=NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        gui_p = new VideoInputGui(*this, options);
    }

    return gui_p;
}

int VideoInputApplication::get_current_frame_number() const
{
    return this->video_input_p->get_current_frame_number();
}

void VideoInputApplication::main_loop()
{

    int num_iterations = 0;
    const int num_iterations_for_timing = 50;
    double start_wall_time = omp_get_wtime();
    bool video_input_is_available = video_input_p->next_frame();
    bool end_of_game = false;
    while(video_input_is_available && !end_of_game)
    {

        // update video input --
        AbstractStereoMatcher::input_image_view_t
        input_left_view(video_input_p->get_left_image()),
                        input_right_view(video_input_p->get_right_image());

        // update user interface
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
