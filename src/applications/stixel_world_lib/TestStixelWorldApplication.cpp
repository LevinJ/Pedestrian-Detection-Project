#include "TestStixelWorldApplication.hpp"

#include "stixel_world_lib.hpp"

#include "video_input/VideoInputFactory.hpp"

#include "helpers/get_option_value.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <omp.h>

namespace stixel_world {

using namespace std;
using namespace boost;
using namespace doppia;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
string TestStixelWorldApplication::get_application_title() const
{
    return "Simple test program for stixel_world_lib. Rodrigo Benenson @ KULeuven. 2012.";
}


TestStixelWorldApplication::TestStixelWorldApplication()
{
    // nothing to do here
    return;
}


TestStixelWorldApplication::~TestStixelWorldApplication()
{
    // nothing to do here
    return;
}


int TestStixelWorldApplication::main(int argc, char *argv[])
{
    cout << get_application_title() << endl;

    // obtain:
    // - left and right rectified images
    program_options::variables_map options = parse_arguments(argc, argv);
    setup_problem(options);

    compute_solution();

    cout << "End of game, have a nice day." << endl;

    return EXIT_SUCCESS;
}


program_options::options_description TestStixelWorldApplication::get_args_options()
{
    program_options::options_description desc("TestStixelWorldApplication options");
    desc.add_options()

            ("configuration_file,c",
             program_options::value<string>()->default_value("test_stixel_world_lib.config.ini"),
             "indicates the path of the configuration .ini file")

            ("save_detections",
             program_options::value<bool>()->default_value(false),
             "save the detected objects in a data sequence file (only available in monocular mode)")

            ;


    return desc;
}


program_options::variables_map TestStixelWorldApplication::parse_arguments(int argc, char *argv[])
{

    program_options::options_description desc("Allowed options");
    desc.add_options()("help", "produces this help message");

    desc.add(TestStixelWorldApplication::get_args_options());

    stixel_world::get_options_description(desc);

    program_options::variables_map options;

    try
    {
        program_options::command_line_parser parser(argc, argv);
        parser.options(desc);

        const program_options::parsed_options the_parsed_options( parser.run() );

        program_options::store(the_parsed_options, options);
        //program_options::store(program_options::parse_command_line(argc, argv, desc), options);
        program_options::notify(options);
    }
    catch (std::exception & e)
    {
        cout << "\033[1;31mError parsing the command line options:\033[0m " << e.what () << endl << endl;
        cout << desc << endl;
        exit(EXIT_FAILURE);
    }


    if (options.count("help"))
    {
        cout << desc << endl;
        exit(EXIT_SUCCESS);
    }

    // parse the configuration file
    {

        string configuration_filename;

        if(options.count("configuration_file") > 0)
        {
            configuration_filename = get_option_value<std::string>(options, "configuration_file");
        }
        else
        {
            cout << "No configuration file provided. Using command line options only." << std::endl;
        }

        if (configuration_filename.empty() == false)
        {
            boost::filesystem::path configuration_file_path(configuration_filename);
            if(boost::filesystem::exists(configuration_file_path) == false)
            {
                cout << "\033[1;31mCould not find the configuration file:\033[0m "
                     << configuration_file_path << endl;
                return options;
            }

            printf("Going to parse the configuration file: %s\n", configuration_filename.c_str());

            try
            {
                fstream configuration_file;
                configuration_file.open(configuration_filename.c_str(), fstream::in);
                program_options::store(program_options::parse_config_file(configuration_file, desc), options);
                configuration_file.close();
            }
            catch (...)
            {
                cout << "\033[1;31mError parsing THE configuration file named:\033[0m "
                     << configuration_filename << endl;
                cout << desc << endl;
                throw;
            }

            cout << "Parsed the configuration file " << configuration_filename << std::endl;
        }
    }

    return options;
}



void TestStixelWorldApplication::setup_problem(const program_options::variables_map &options)
{

    const boost::filesystem::path configuration_filepath = get_option_value<std::string>(options, "configuration_file");

    printf("Application will use stereo input\n");
    video_input_p.reset(VideoInputFactory::new_instance(options));

    stixel_world::init_stixel_world(configuration_filepath);

    if(not video_input_p)
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
                                    "No images to read, nothing to compute.");
    }

    return;
}

void TestStixelWorldApplication::compute_solution()
{
    printf("Processing, please wait...\n");

    const bool should_print = true;
    int num_iterations = 0;
    const int num_iterations_for_timing = 10, num_iterations_for_processing_timing = 50;
    double cumulated_processing_time = 0, cumulated_stixels_compute_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = false;
    video_input_is_available = video_input_p->next_frame();


    while(video_input_is_available)
    {

        // update video input --
        const double start_processing_wall_time = omp_get_wtime();
        stixel_world::input_image_const_view_t
                left_view(video_input_p->get_left_image()),
                right_view(video_input_p->get_right_image());
        stixel_world::set_rectified_stereo_images_pair(left_view, right_view);

        const double start_stixels_compute_wall_time = omp_get_wtime();

        stixel_world::compute();
        cumulated_stixels_compute_time += omp_get_wtime() - start_stixels_compute_wall_time;

        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        const bool print_stixels_info = true;
        if(print_stixels_info)
        {
            const stixel_world::stixels_t stixels = stixel_world::get_stixels();
            bool stixels_are_fine = false;

            if(stixels.empty())
            {
                throw std::runtime_error("Running in stereo mode but stixels are empty, something went terribly wrong");
            }
            else
            {
                int min_bottom_y=stixels.front().bottom_y, max_bottom_y=min_bottom_y;

                for(size_t i=0; i < stixels.size(); i+=1)
                {
                    const int bottom_y = stixels[i].bottom_y;
                    min_bottom_y = std::min(bottom_y, min_bottom_y);
                    max_bottom_y = std::max(bottom_y, max_bottom_y);
                }

                stixels_are_fine = (min_bottom_y != max_bottom_y);

                if(stixels_are_fine)
                {
                    //printf("max_bottom_y - min_bottom_y == %i\n", max_bottom_y - min_bottom_y);
                    //printf("Stixels bottom seems fine\n");
                }
                else
                {
                    printf("Stixels bottom is completelly flat, sounds bad!\n");
                }
            }


        } // end of if print_stixels_info

        num_iterations += 1;

        if(should_print and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        if(should_print and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average total stixel world speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_processing_time , num_iterations );
        }

        if(should_print and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average stixel world compute only speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_stixels_compute_time , num_iterations );
        }

        video_input_is_available = video_input_p->next_frame();
    } // end of "while video input"

    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average stixel world speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    return;
}

} // end of namespace stixel_world
