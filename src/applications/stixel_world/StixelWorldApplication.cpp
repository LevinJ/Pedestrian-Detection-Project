// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include <boost/gil/gil_all.hpp>

#include "StixelWorldApplication.hpp"
#include "StixelWorldGui.hpp"
#include "applications/EmptyGui.hpp"

#include "video_input/VideoInputFactory.hpp"
#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"
#include "stereo_matching/stixels/StixelWorldEstimator.hpp"

#include "stereo_matching/stixels/motion/DummyStixelMotionEstimator.hpp"

#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/any_to_string.hpp"
#include "helpers/for_each.hpp"
#include "helpers/xyz_indices.hpp"

#include "helpers/data/DataSequence.hpp"
#include "stereo_matching/stixels/ground_top_and_bottom.pb.h"
#include "stereo_matching/stixels/stixels.pb.h"

#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>

#include <omp.h>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelWorldApplication");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelWorldApplication");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelWorldApplication");
}

} // end of anonymous namespace

namespace doppia
{


using logging::log;
using namespace std;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string StixelWorldApplication::get_application_title()
{
    return  "Stixel world scene understanding. Rodrigo Benenson @ KULeuven. 2010-2011.";
}

StixelWorldApplication::StixelWorldApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}



StixelWorldApplication::~StixelWorldApplication()
{
    // nothing to do here
    return;
}


program_options::options_description StixelWorldApplication::get_options_description()
{
    program_options::options_description desc("StixelWorldApplication options");

    const std::string application_name = "stixel_world";
    BaseApplication::add_args_options(desc, application_name);

    // StixelWorldApplication has no particular options,
    // it only uses StixelWorldGui and the factories options


    desc.add_options()
            ("save_stixels",
             program_options::value<bool>()->default_value(false),
             "save the estimated stixels in a data sequence file")

            ("save_ground_plane_corridor",
             program_options::value<bool>()->default_value(false),
             "save the estimated expected bottom and top of objects in the data sequence file")

            ("gui.disabled",
             program_options::value<bool>()->default_value(false),
             "if true, no user interface will be presented")

            ("silent_mode",
             program_options::value<bool>()->default_value(false),
             "if true, no status information will be printed at run time (use this for speed benchmarking)")

            ("stixel_world.estimate_motion",
             program_options::value<bool>()->default_value(false),
             "if true the stixels motion will be estimated")

            ;

    return desc;
}

void StixelWorldApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(StixelWorldApplication::get_options_description());
    desc.add(StixelWorldGui::get_args_options());
    desc.add(VideoInputFactory::get_args_options());

    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(DisparityCostVolumeEstimatorFactory::get_args_options());

    desc.add(StixelWorldEstimatorFactory::get_args_options());

    desc.add(DummyStixelMotionEstimator::get_args_options());

    return;
}


/// helper method called by setup_problem
void StixelWorldApplication::setup_logging(std::ofstream &log_file, const program_options::variables_map &options)
{

    if(log_file.is_open())
    {
        // the logging is already setup
        return;
    }

    // set base logging rules --
    BaseApplication::setup_logging(log_file, options);

    const bool silent_mode = get_option_value<bool>(options, "silent_mode");
    if(silent_mode == false)
    {
        // set our own stdout rules --
        logging::LogRuleSet &rules_for_stdout = logging::get_log().console_log().rule_set();
        //rules_for_stdout.add_rule(logging::InfoMessage, "StixelWorldApplication");
        rules_for_stdout.add_rule(logging::InfoMessage, "*");
    }

    return;
}

void StixelWorldApplication::setup_problem(const program_options::variables_map &options)
{
    // instanciate the different processing modules --
    video_input_p.reset(VideoInputFactory::new_instance(options));
    stixel_world_estimator_p.reset(StixelWorldEstimatorFactory::new_instance(options, *video_input_p));

    // parse the application specific options --
    should_save_stixels = get_option_value<bool>(options, "save_stixels");
    should_save_ground_plane_corridor = get_option_value<bool>(options, "save_ground_plane_corridor");
    silent_mode = get_option_value<bool>(options, "silent_mode");

    if(should_save_stixels == false and should_save_ground_plane_corridor == true )
    {
        throw std::invalid_argument("save_ground_plane_corridor requires the save_stixels option to be true");
    }

    const bool estimate_motion = get_option_value<bool>(options, "stixel_world.estimate_motion");

    if(estimate_motion)
    {
        stixel_motion_estimator_p.reset( new DummyStixelMotionEstimator( options, video_input_p->get_metric_camera(), stixel_world_estimator_p->get_stixel_width() ) );
    }

    return;
}


AbstractGui* StixelWorldApplication::create_gui(const program_options::variables_map &options)
{
    const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    AbstractGui *gui_p=NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        assert(video_input_p);
        assert(stixel_world_estimator_p);

        gui_p = new StixelWorldGui(*this,
                                   video_input_p,
                                   stixel_world_estimator_p, stixel_motion_estimator_p,
                                   options);
    }

    return gui_p;
}

int StixelWorldApplication::get_current_frame_number() const
{
    return this->video_input_p->get_current_frame_number();
}

void StixelWorldApplication::main_loop()
{   

    const bool should_print = not silent_mode;
    if(silent_mode)
    {
        printf("The application is running in silent mode. "
               "No information will be printed until all the frames have been processed.\n");
    }

    int num_iterations = 0;
    const int num_iterations_for_timing = 50;
    double cumulated_processing_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = video_input_p->next_frame();
    bool end_of_game = false;

    while(video_input_is_available and (not end_of_game))
    {

        // update video input --
        AbstractStereoMatcher::input_image_view_t
                input_left_view(video_input_p->get_left_image()),
                input_right_view(video_input_p->get_right_image());

       const double start_processing_wall_time = omp_get_wtime();

        // estimate ground plane and stixels --
        {
            // dirty trick to work around the video input BUG
            // cast to color view
            boost::gil::rgb8c_view_t
                    left_view(input_left_view._dynamic_cast<boost::gil::rgb8c_view_t>()),
                    right_view(input_right_view._dynamic_cast<boost::gil::rgb8c_view_t>());

            stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);

            if(stixel_motion_estimator_p)
            {
                stixel_motion_estimator_p->set_new_rectified_image(left_view);
            }
        }

        stixel_world_estimator_p->compute();


        if(stixel_motion_estimator_p)
        {            
            //printf("\nFRAME # : %i\n", get_current_frame_number());
            //const double start_motion_processing_wall_time = omp_get_wtime();

            const stixels_t &the_stixels = stixel_world_estimator_p->get_stixels();
            stixel_motion_estimator_p->set_estimated_stixels(the_stixels);

            if( get_current_frame_number() > 1 ) // If there are a "current" frame and a "previous" frame
            {
                stixel_motion_estimator_p->compute();
            }
            else
            {
                std::cout << "This is the very first frame - No stixels motion estimation !" << std::endl;
            }

        }
        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        if(should_save_stixels)
        {
            record_stixels();
        }

        // do semantic labeling and objects detections in stixels world --
        // (not yet implemented)

        // update user interface --
        end_of_game = update_gui();

        num_iterations += 1;
        if(should_print and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        // retrieve next rectified input stereo pair
        video_input_is_available = video_input_p->next_frame();
    } // end of "while video input and not end of game"

    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average stixel world estimation speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    return;
}

void StixelWorldApplication::record_stixels()
{
    if(stixels_data_sequence_p == false)
    {
        // first invocation, need to create the data_sequence file first
        string filename = (get_recording_path() / "stixels.data_sequence").string();

        StixelsDataSequence::attributes_t attributes;
        attributes.insert(std::make_pair("created_by", "StixelWorldApplication"));

        stixels_data_sequence_p.reset(new StixelsDataSequence(filename, attributes));

        log_info() << "Created recording file " << filename << std::endl;
    }

    assert((bool) stixels_data_sequence_p == true);

    stixels_t the_stixels;
    if(stixel_motion_estimator_p)
    {
        // copy stixels with motion data
        the_stixels = stixel_motion_estimator_p->get_current_stixels();
        //const AbstractStixelMotionEstimator::stixels_motion_t& stixels_motion = stixel_motion_estimator_p->get_stixels_motion();
    }
    else
    {
        // copy stixels, without motion data
        the_stixels = stixel_world_estimator_p->get_stixels(); // current stixels
    }



    StixelsDataSequence::data_type stixels_data;
    const string image_name = boost::str(boost::format("frame_%i") % this->get_current_frame_number());
    stixels_data.set_image_name(image_name);    

    BOOST_FOREACH(const Stixel &stixel, the_stixels)
    {
        doppia_protobuf::Stixel *stixel_data_p = stixels_data.add_stixels();

        stixel_data_p->set_width(stixel.width);
        stixel_data_p->set_x(stixel.x);
        stixel_data_p->set_bottom_y(stixel.bottom_y);
        stixel_data_p->set_top_y(stixel.top_y);
        stixel_data_p->set_disparity(stixel.disparity);

        doppia_protobuf::Stixel::Type stixel_type = doppia_protobuf::Stixel::Unknown;
        switch(stixel.type)
        {
        case Stixel::Occluded:
            stixel_type = doppia_protobuf::Stixel::Occluded;
            break;

        case Stixel::Car:
            stixel_type = doppia_protobuf::Stixel::Car;
            break;

        case Stixel::Pedestrian:
            stixel_type = doppia_protobuf::Stixel::Pedestrian;
            break;

        case Stixel::StaticObject:
            stixel_type = doppia_protobuf::Stixel::StaticObject;
            break;

        case Stixel::Unknown:
            stixel_type = doppia_protobuf::Stixel::Unknown;
            break;

        default:
            throw std::invalid_argument(
                        "StixelWorldApplication::record_stixels received a stixel "
                        "with a type with a no known correspondence in "
                        "the protocol buffer format");
            break;
        }

        stixel_data_p->set_type(stixel_type);

        if(stixel_motion_estimator_p)
        {
            // if stixel contains motion information
            stixel_data_p->set_backward_delta_x( stixel.backward_delta_x );
            stixel_data_p->set_valid_delta_x( stixel.valid_backward_delta_x );            
        }

    } // end of "for each stixel in stixels"    

    add_ground_plane_data(stixels_data);

    if(should_save_ground_plane_corridor)
    {
        add_ground_plane_corridor_data(stixels_data);
    }

    stixels_data_sequence_p->write(stixels_data);

    return;
} // end of StixelWorldApplication::record_stixels


void StixelWorldApplication::add_ground_plane_corridor_data(doppia_protobuf::Stixels &stixels_data)
{

    StixelWorldEstimator *the_stixel_world_estimator_p =
            dynamic_cast<StixelWorldEstimator *>(stixel_world_estimator_p.get());

    if(the_stixel_world_estimator_p)
    {
        StixelWorldEstimator &the_stixel_world_estimator = *the_stixel_world_estimator_p;

        // add the ground corridor data --
        {
            doppia_protobuf::GroundTopAndBottom *ground_corridor_p = stixels_data.mutable_ground_top_and_bottom();

            const StixelWorldEstimator::ground_plane_corridor_t & ground_corridor = \
                    the_stixel_world_estimator.get_ground_plane_corridor();

            for(size_t v=0; v < ground_corridor.size(); v+=1)
            {
                const int bottom_y = v;
                const int top_y = ground_corridor[v];

                if(top_y < 0)
                {
                    // a non valid bottom_y value
                    continue;
                }

                doppia_protobuf::TopAndBottom *top_and_bottom_p = ground_corridor_p->add_top_and_bottom();

                assert(top_y < bottom_y);
                top_and_bottom_p->set_top_y(top_y);
                top_and_bottom_p->set_bottom_y(bottom_y);

            } // end of "for each row bellow the horizon"
        }

        // add the ground plane data --
        {
            const GroundPlane &ground_plane = the_stixel_world_estimator.get_ground_plane();

            doppia_protobuf::Plane3d *ground_plane3d_p = stixels_data.mutable_ground_plane();

            ground_plane3d_p->set_offset(ground_plane.offset());
            ground_plane3d_p->set_normal_x(ground_plane.normal()(i_x));
            ground_plane3d_p->set_normal_y(ground_plane.normal()(i_y));
            ground_plane3d_p->set_normal_z(ground_plane.normal()(i_z));
        }
    }
    else
    { // the_stixel_world_estimator_p == NULL
        throw std::runtime_error(
                    "StixelWorldApplication::add_ground_plane_corridor expected "
                    "AbstractStixelWorldEstimator to be an instance of StixelWorldEstimator. "
                    "Try using a non_fast variant");
    }

    return;
} // end of StixelWorldApplication::add_ground_plane_corridor_data

void StixelWorldApplication::add_ground_plane_data(doppia_protobuf::Stixels &stixels_data)
{

    const GroundPlane &ground_plane = stixel_world_estimator_p->get_ground_plane();

    doppia_protobuf::Plane3d *ground_plane3d_p = stixels_data.mutable_ground_plane();

    ground_plane3d_p->set_offset(ground_plane.offset());
    ground_plane3d_p->set_normal_x(ground_plane.normal()(i_x));
    ground_plane3d_p->set_normal_y(ground_plane.normal()(i_y));
    ground_plane3d_p->set_normal_z(ground_plane.normal()(i_z));

    return;
} // end of StixelWorldApplication::add_ground_plane_data


} // end of namespace doppia

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
