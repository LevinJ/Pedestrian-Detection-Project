
// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include "ObjectsDetectionApplication.hpp"
#include "ObjectsDetectionGui.hpp"

#include "applications/EmptyGui.hpp"

#include "video_input/VideoInputFactory.hpp"
#include "video_input/ImagesFromDirectory.hpp"
#include "video_input/preprocessing/AddBorderFunctor.hpp"

#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"
#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelsEstimator.hpp"

#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"

#include "objects_detection/ObjectsDetectorFactory.hpp"
#include "objects_detection/AbstractObjectsDetector.hpp"

#include "objects_tracking/ObjectsTrackerFactory.hpp"
#include "objects_tracking/AbstractObjectsTracker.hpp"

#include "objects_detection/non_maximal_suppression/StixelsWeightingAndNonMaximalSuppression.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/any_to_string.hpp"
#include "helpers/for_each.hpp"
#include "helpers/Log.hpp"

#include "helpers/data/DataSequence.hpp"
#include "objects_detection/detections.pb.h"

#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>

#include <boost/format.hpp>

#include <omp.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ObjectsDetectionApplication");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ObjectsDetectionApplication");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ObjectsDetectionApplication");
}

} // end of anonymous namespace

namespace doppia
{

using logging::log;
using namespace std;
typedef AbstractStixelWorldEstimator::ground_plane_corridor_t ground_plane_corridor_t;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string ObjectsDetectionApplication::get_application_title()
{
    return "Stereo objects detection on a flat world. "
            "Rodrigo Benenson, Markus Mathias, Mohamed Omran @ KULeuven & MPI-Inf. 2011-2014.";
}


ObjectsDetectionApplication::ObjectsDetectionApplication()
    : BaseApplication(),
      should_save_detections(false),
      use_ground_plane_only(false),
      should_process_folder(false),
      silent_mode(false),
      additional_border(0),
      stixels_computation_period(1)
{
    // nothing to do here
    return;
}


ObjectsDetectionApplication::~ObjectsDetectionApplication()
{
    // nothing to do here
    return;
}


program_options::options_description ObjectsDetectionApplication::get_options_description()
{
    program_options::options_description desc("ObjectsDetectionApplication options");

    const std::string application_name = "objects_detection";
    BaseApplication::add_args_options(desc, application_name);

    desc.add_options()

            ("save_detections",
             program_options::value<bool>()->default_value(false),
             "save the detected objects in a data sequence file")

            ("process_folder",
             program_options::value<string>(),
             "for evaluation purposes, will process images on a folder. Normal video input will be ignored")

            ("additional_border",
             program_options::value<int>()->default_value(0),
             "when using process_folder, will add border to the image to enable detection of cropped pedestrians. "
             "Value is in pixels (e.g. 50 pixels)")

            ("use_ground_plane",
             program_options::value<bool>()->default_value(false),
             "when using stereo input, estimate the ground plane and use it to guide the detections")

            ("use_stixels",
             program_options::value<bool>()->default_value(false),
             "when using stereo input, estimate the stixel world and use it to guide the detections")

            ("stixels_period",
             program_options::value<int>()->default_value(1),
             "when using stereo input, at which frequency should be update the stixel/ground plane estimate ? "
             "Value 1 means every frame, value 5 means one out of five frames, "
             "value 0 means only the first frame and never after")

            ("gui.disabled",
             program_options::value<bool>()->default_value(false),
             "if true, no user interface will be presented")

            ("silent_mode",
             program_options::value<bool>()->default_value(false),
             "if true, no status information will be printed at run time (use this for speed benchmarking)")

            ;

    return desc;
}


void ObjectsDetectionApplication::get_all_options_descriptions(program_options::options_description &desc)
{
    desc.add(VideoInputFactory::get_args_options());

    // Objects detection options --
    desc.add(ObjectsDetectionApplication::get_options_description());
    desc.add(ObjectsDetectionGui::get_args_options());

    desc.add(ObjectsDetectorFactory::get_args_options());
    desc.add(ObjectsTrackerFactory::get_args_options());

    // Stixel world estimation options --
    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(DisparityCostVolumeEstimatorFactory::get_args_options());

    desc.add(StixelWorldEstimatorFactory::get_args_options());

    return;
}


/// helper method called by setup_problem
void ObjectsDetectionApplication::setup_logging(std::ofstream &log_file, const program_options::variables_map &options)
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
        //rules_for_stdout.add_rule(logging::InfoMessage, "ObjectsDetectionApplication");
#if defined(DEBUG)
        rules_for_stdout.add_rule(logging::DebugMessage, "*"); // we are debugging this application
#else
        rules_for_stdout.add_rule(logging::InfoMessage, "*"); // "production mode"
#endif
    }

    return;
}


void ObjectsDetectionApplication::setup_problem(const program_options::variables_map &options)
{
    // parse the application specific options --
    should_save_detections = get_option_value<bool>(options, "save_detections");
    should_process_folder = options.count("process_folder") > 0;
    silent_mode = get_option_value<bool>(options, "silent_mode");

    if(options.count("additional_border") > 0)
    { // this option may not be available when calling this function from a different application
        additional_border = get_option_value<int>(options, "additional_border");
    }
    else
    {
        additional_border = 0;
    }

    // instanciate the different processing modules --
    if(should_process_folder)
    {
        const filesystem::path folder_to_process = get_option_value<string>(options, "process_folder");
        directory_input_p.reset(new ImagesFromDirectory(folder_to_process));
    }
    else
    {
        video_input_p.reset(VideoInputFactory::new_instance(options));
    }

    objects_detector_p.reset(ObjectsDetectorFactory::new_instance(options));

    if(not objects_detector_p)
    {
        throw std::runtime_error("No objects detector selected, this application is pointless without one. Check the value of objects_detector.method");
    }

    use_ground_plane_only = get_option_value<bool>(options, "use_ground_plane");

    const bool
            use_stixels = get_option_value<bool>(options, "use_stixels"),
            // for ground plane estimation, we compute the stixel world,
            // but use only the ground plane information (omit the stixels data)
            should_estimate_stixels = use_stixels or use_ground_plane_only;

    stixels_computation_period = get_option_value<int>(options, "stixels_period");

    if(use_stixels and use_ground_plane_only)
    {
        throw std::invalid_argument("use_ground_plane is incompatible with the program option use_stixels. Choose one.");
    }

    if(should_estimate_stixels and (not video_input_p))
    {
        throw std::invalid_argument("Stixels estimation requires stereo input. "
                                    "use_stixels and process_folder are incompatible options");
    }

    if(should_estimate_stixels)
    {
        stixel_world_estimator_p.reset(StixelWorldEstimatorFactory::new_instance(options, *video_input_p));
    }

    if(video_input_p)
    { // for process folder, objects_tracker_p stays empty
        objects_tracker_p.reset(ObjectsTrackerFactory::new_instance(options));
        //, video_input_p->get_metric_camera()));
    }
    return;
}


AbstractGui* ObjectsDetectionApplication::create_gui(const program_options::variables_map &options)
{
    const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    AbstractGui *gui_p=NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        gui_p = new ObjectsDetectionGui(*this, options);
    }

    return gui_p;
}


int ObjectsDetectionApplication::get_current_frame_number() const
{
    int current_frame_number = 0;
    if(video_input_p)
    {
        current_frame_number = video_input_p->get_current_frame_number();
    }
    else if(directory_input_p)
    {
        current_frame_number = directory_input_p->get_current_frame_number();
    }

    return current_frame_number;
}


namespace { // anonymous namespace

boost::barrier stixel_world_compute_start_barrier(2); // only two threads are involved in this barrier
boost::barrier stixel_world_compute_ended_barrier(2); // only two threads are involved in this barrier


void stixel_world_compute_thread(shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p)
{
    while(stixel_world_estimator_p) // this loop will be stopped via an interruption
    {
        {
            stixel_world_compute_start_barrier.wait(); // waiting for main_loop to launch a new computation
            // when interrupted the thread will exit at this point
            //printf("stixel_world_compute_barrier.wait() ended, starting stixel_world_estimator_p->compute()\n");
        }

        stixel_world_estimator_p->compute();

        stixel_world_compute_ended_barrier.wait();
    }
    return;
}

} // end of anonymous namespace


void ObjectsDetectionApplication::main_loop()
{

    //const bool should_print_speed = not silent_mode;
    const bool should_print_speed = true;
    if(silent_mode)
    {
        printf("The application is running in silent mode. "
               "No information will be printed until all the frames have been processed.\n");
    }

    int num_iterations = 0;
    //const int num_iterations_for_timing = 10, num_iterations_for_processing_timing = 50;
    const int num_iterations_for_timing = 500, num_iterations_for_processing_timing = 250;
    double cumulated_processing_time = 0, cumulated_objects_detector_compute_time = 0;
    double start_wall_time = omp_get_wtime();

    AddBorderFunctor add_border(additional_border);
    bool video_input_is_available = false;
    if(should_process_folder)
    {
        video_input_is_available = directory_input_p->next_frame();
    }
    else
    {
        video_input_is_available = video_input_p->next_frame();
    }

    stixels_t stixels_from_previous_frame;
    ground_plane_corridor_t ground_corridor_from_previous_frame;

    int stixels_period_counter = stixels_computation_period; // start by considering the stixels of the first frame


    boost::thread stixel_world_estimator_thread;

    // initialized stixels_from_previous_frame
    if(stixel_world_estimator_p)
    {
        // set the input for stixels and objects detection
        AbstractVideoInput::input_image_view_t
                left_view(video_input_p->get_left_image()),
                right_view(video_input_p->get_right_image());

        stixel_world_estimator_thread = boost::thread(stixel_world_compute_thread, stixel_world_estimator_p);

        stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);

        stixel_world_compute_start_barrier.wait();
        // we make sure the first frame is processed
        // this is equivalent to calling stixel_world_estimator_p->compute();
    }

    if(objects_tracker_p)
    {
        AbstractVideoInput::input_image_view_t left_view(video_input_p->get_left_image());
        objects_tracker_p->set_image_size(left_view.width(), left_view.height());
    }



    bool end_of_game = false;

    while(video_input_is_available and (not end_of_game))
    {

        // update video input --
        AbstractVideoInput::input_image_view_t input_view;
        if(should_process_folder)
        {
            input_view = directory_input_p->get_image();
            log_info() << "Processing image: " << directory_input_p->get_image_name() << std::endl;
        }
        else
        {
            input_view = video_input_p->get_left_image();
            //input_right_view(video_input_p->get_right_image());
        }

        input_view = add_border(input_view);

        // we start measuring the time before uploading the data to the GPU
        const double start_processing_wall_time = omp_get_wtime();

        objects_detector_p->set_image(input_view);

        if(stixel_world_estimator_p and (stixels_period_counter == stixels_computation_period))
        {
            // make sure previous computation finished --
            //stixel_world_estimator_thread.join();
            stixels_period_counter = 0;

            // set the input for stixels and objects detection --
            AbstractVideoInput::input_image_view_t
                    left_view(video_input_p->get_left_image()),
                    right_view(video_input_p->get_right_image());

            stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);

            // make sure computation finished before retrieving the results
            stixel_world_compute_ended_barrier.wait();

            ground_corridor_from_previous_frame = stixel_world_estimator_p->get_ground_plane_corridor();
            objects_detector_p->set_ground_plane_corridor(ground_corridor_from_previous_frame);

            if(not use_ground_plane_only)
            {
                stixels_from_previous_frame = stixel_world_estimator_p->get_stixels();
                objects_detector_p->set_stixels(stixels_from_previous_frame);
            }


            // launch stixels estimation in a thread, compute the objects detection --
            {
                {
                    // launch the stixel world computation (and wait if last one has not yet finished)
                    stixel_world_compute_start_barrier.wait();
                    // at this point stixel_world_estimator_p->compute() is being called
                    // inside stixel_world_compute_thread
                }

                { // the gpu computation runs in parallel with the stixel world estimation
                    const double start_objects_detector_compute_wall_time = omp_get_wtime();
                    objects_detector_p->compute();
                    cumulated_objects_detector_compute_time += omp_get_wtime() - start_objects_detector_compute_wall_time;
                }
            } // end of "pragma omp sections"

        } // end of "if should launch stixels world estimation"
        else
        {
            const double start_objects_detector_compute_wall_time = omp_get_wtime();
            objects_detector_p->compute();
            cumulated_objects_detector_compute_time += omp_get_wtime() - start_objects_detector_compute_wall_time;
        }


        // tracking --
        if(true and objects_tracker_p)
        {
            objects_tracker_p->set_detections(objects_detector_p->get_detections());
            objects_tracker_p->compute();
        }

        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        if(should_save_detections)
        {
            record_detections();
        }

        // update user interface --
        end_of_game = update_gui();

        num_iterations += 1;
        stixels_period_counter += 1;

        if(should_print_speed and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        if(should_print_speed and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average total objects detection speed per iteration %.2lf [Hz] "
                   "(in the last %i iterations) (not including stixels/ground plane estimation)\n",
                   num_iterations / cumulated_processing_time , num_iterations );
        }

        if(should_print_speed and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average objects detection compute only speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_objects_detector_compute_time , num_iterations );
        }

        // retrieve next rectified input stereo pair
        if(should_process_folder)
        {
            video_input_is_available = directory_input_p->next_frame();
        }
        else
        {
            video_input_is_available = video_input_p->next_frame();
        }
    } // end of "while video input and not end of game"

    stixel_world_estimator_thread.interrupt(); // stop the computation thread

    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average objects detection speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    return;
}


void ObjectsDetectionApplication::record_detections()
{
    typedef AbstractObjectsDetector::detections_t detections_t;
    typedef AbstractObjectsDetector::detection_t detection_t;

    if(detections_data_sequence_p == false)
    {
        // first invocation, need to create the data_sequence file first
        const string filename = (get_recording_path() / "detections.data_sequence").string();

        DetectionsDataSequence::attributes_t attributes;
        attributes.insert(std::make_pair("created_by", "ObjectsDetectionApplication"));

        detections_data_sequence_p.reset(new DetectionsDataSequence(filename, attributes));

        log_info() << "Created recording file " << filename << std::endl;
    }

    assert(static_cast<bool>(detections_data_sequence_p) == true);

    detections_t the_detections;

    if(objects_tracker_p)
    {
        the_detections = objects_tracker_p->get_current_detections();
    }
    else
    {
        the_detections = objects_detector_p->get_detections();
    }

    DetectionsDataSequence::data_type detections_data;

    string image_name;
    if(should_process_folder)
    {
        image_name = directory_input_p->get_image_name();
    }
    else
    {
        image_name = boost::str(boost::format("frame_%i") % this->get_current_frame_number());
    }

    detections_data.set_image_name(image_name);

    BOOST_FOREACH(const detection_t &detection, the_detections)
    {
        doppia_protobuf::Detection *detection_data_p = detections_data.add_detections();

        detection_data_p->set_score(detection.score);

        // Point2d support negative values, these are needed when handling detections on the image border
        // (occluded detections)
        doppia_protobuf::Point2d &max_corner = *(detection_data_p->mutable_bounding_box()->mutable_max_corner());
        max_corner.set_x(detection.bounding_box.max_corner().x() - additional_border);
        max_corner.set_y(detection.bounding_box.max_corner().y() - additional_border);
        doppia_protobuf::Point2d &min_corner = *(detection_data_p->mutable_bounding_box()->mutable_min_corner());
        min_corner.set_x(detection.bounding_box.min_corner().x() - additional_border);
        min_corner.set_y(detection.bounding_box.min_corner().y() - additional_border);

        doppia_protobuf::Detection::ObjectClasses object_class = doppia_protobuf::Detection::Unknown;
        switch(detection.object_class)
        { // Car, Pedestrian, Bike, Motorbike, Bus, Tram, StaticObject, Unknown

        case detection_t::Car:
            object_class = doppia_protobuf::Detection::Car;
            break;

        case detection_t::Pedestrian:
            object_class = doppia_protobuf::Detection::Pedestrian;
            break;

        case detection_t::Bike:
            object_class = doppia_protobuf::Detection::Bike;
            break;

        case detection_t::Motorbike:
            object_class = doppia_protobuf::Detection::Motorbike;
            break;

        case detection_t::Bus:
            object_class = doppia_protobuf::Detection::Bus;
            break;

        case detection_t::Tram:
            object_class = doppia_protobuf::Detection::Tram;
            break;

        case detection_t::StaticObject:
            object_class = doppia_protobuf::Detection::StaticObject;
            break;

        default:
            throw std::invalid_argument(
                        "ObjectsDetectionApplication::record_detections received a detection "
                        "with an object_class with a no known correspondence in "
                        "the protocol buffer format");
            break;
        }

        detection_data_p->set_object_class(object_class);
    } // end of "for each stixel in stixels"


    detections_data_sequence_p->write(detections_data);

    return;
} // end of ObjectsDetectionApplication::record_detections


} // end of namespace doppia

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
