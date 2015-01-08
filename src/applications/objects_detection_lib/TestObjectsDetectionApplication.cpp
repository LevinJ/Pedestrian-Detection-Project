#include "TestObjectsDetectionApplication.hpp"

#include "objects_detection_lib.hpp"

#include "video_input/ImagesFromDirectory.hpp"

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
#include "video_input/VideoInputFactory.hpp"
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not

#include "objects_detection/ObjectsDetectorFactory.hpp"

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

#if defined(USE_GPU)
#include <cuda_runtime_api.h>
#endif

namespace objects_detection {

using namespace std;
using namespace boost;
using namespace doppia;

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
class FakeAbstractVideoInput
{
public:
    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8c_view_t input_image_view_t;
    typedef input_image_view_t::point_t dimensions_t;

    const input_image_view_t get_left_image();
    const input_image_view_t get_right_image();

    /// Advance in stream, @returns true if successful
    bool next_frame();
};

const FakeAbstractVideoInput::input_image_view_t FakeAbstractVideoInput::get_left_image()
{
    return input_image_view_t();
}

const FakeAbstractVideoInput::input_image_view_t FakeAbstractVideoInput::get_right_image()
{
    return input_image_view_t();
}


/// Advance in stream, @returns true if successful
bool FakeAbstractVideoInput::next_frame()
{
    return false;
}
#endif

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
string TestObjectsDetectionApplication::get_application_title() const
{
    return "Simple test program for objects_detection_lib. Rodrigo Benenson @ KULeuven. 2011-2012.";
}


TestObjectsDetectionApplication::TestObjectsDetectionApplication()
{
    // nothing to do here
    return;
}


TestObjectsDetectionApplication::~TestObjectsDetectionApplication()
{
    free_object_detector();
    return;
}


int TestObjectsDetectionApplication::main(int argc, char *argv[])
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


program_options::options_description TestObjectsDetectionApplication::get_args_options()
{
    program_options::options_description desc("TestObjectsDetectionApplication options");
    desc.add_options()

            ("configuration_file,c",
         #if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
             program_options::value<string>()->default_value("test_monocular_objects_detection_lib.config.ini"),
         #else
             program_options::value<string>()->default_value("test_objects_detection_lib.config.ini"),
         #endif
             "indicates the path of the configuration .ini file")

            ("save_detections",
             program_options::value<bool>()->default_value(false),
             "save the detected objects in a data sequence file (only available in monocular mode)")

            // (added directly in the objects_detection::get_options_description, so that it can be included in the config file)
            //("video_input.images_folder,i", program_options::value<string>(),
            // "path to a directory with monocular images. This option will overwrite left/right_filename_mask values")

            ;


    return desc;
}

program_options::variables_map TestObjectsDetectionApplication::parse_arguments(int argc, char *argv[])
{

    program_options::options_description desc("Allowed options");
    desc.add_options()("help", "produces this help message");

    desc.add(TestObjectsDetectionApplication::get_args_options());

    objects_detection::get_options_description(desc);

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


#if defined(USE_GPU)

/// copy paste from CudaSDK shared/shrUtils.h
inline int ConvertSMVer2Cores(const int major, const int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10,  8 },
      { 0x11,  8 },
      { 0x12,  8 },
      { 0x13,  8 },
      { 0x20, 32 },
      { 0x21, 48 },
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}

/// subset of the code from cudaSDK deviceQuery sample program
void print_gpu_information()
{
    cudaDeviceProp device_properties;
    const int device_index = 0;
    cudaGetDeviceProperties(&device_properties, device_index);

    printf("GPU information ----\n");
    printf("Device %d: \"%s\"\n", device_index, device_properties.name);

    printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n",
           device_properties.multiProcessorCount,
           ConvertSMVer2Cores(device_properties.major, device_properties.minor),
           ConvertSMVer2Cores(device_properties.major, device_properties.minor) * device_properties.multiProcessorCount);

    printf("  Total amount of constant memory:               %zi bytes\n",
           device_properties.totalConstMem);
    printf("  Total amount of shared memory per block:       %zi bytes\n",
           device_properties.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           device_properties.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           device_properties.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           device_properties.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           device_properties.maxThreadsDim[0], device_properties.maxThreadsDim[1], device_properties.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           device_properties.maxGridSize[0], device_properties.maxGridSize[1], device_properties.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zi bytes\n",
           device_properties.memPitch);
    printf("  Texture alignment:                             %zi bytes\n",
           device_properties.textureAlignment);

    printf("----\n\n");
    return;
}

#else

void print_gpu_information()
{
    // no gpu more, nothing to print
    return;
}

#endif

void TestObjectsDetectionApplication::setup_problem(const program_options::variables_map &options)
{
    print_gpu_information();

    should_save_detections = get_option_value<bool>(options, "save_detections");

    const boost::filesystem::path configuration_filepath = get_option_value<std::string>(options, "configuration_file");

    bool use_ground_plane = false, use_stixels = false;

    if(options.count("video_input.images_folder") > 0)
    {
        printf("Application will use monocular input\n");
        use_ground_plane = false;
        use_stixels = false;

        const filesystem::path folder_to_process = get_option_value<string>(options, "video_input.images_folder");
        directory_input_p.reset(new ImagesFromDirectory(folder_to_process));
    }
    else
    {
#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
        throw std::invalid_argument("This executable does not support stereo input");
#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
        printf("Application will use stereo input\n");
        // FIXME should enable using ground_plane only
        use_ground_plane = true;
        use_stixels = true;

        video_input_p.reset(VideoInputFactory::new_instance(options));
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not
    }


    if(video_input_p and should_save_detections)
    {
        throw std::runtime_error("save_detections option is only available when using video_input.images_folder");
    }

    objects_detection::init_objects_detection(configuration_filepath, use_ground_plane, use_stixels);


    if((not directory_input_p) and (not video_input_p))
    {
        throw std::invalid_argument("Failed to initialize a video input module. "
                                    "No images to read, nothing to compute.");
    }

    return;
}

void TestObjectsDetectionApplication::compute_solution()
{
    printf("Processing, please wait...\n");

    const bool should_print = true;
    int num_iterations = 0;
    const int num_iterations_for_timing = 10, num_iterations_for_processing_timing = 50;
    double cumulated_processing_time = 0, cumulated_objects_detector_compute_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = false;
    if(directory_input_p)
    {
        video_input_is_available = directory_input_p->next_frame();
    }
    else
    {
        video_input_is_available = video_input_p->next_frame();
    }


    while(video_input_is_available)
    {

        // update video input --
        const double start_processing_wall_time = omp_get_wtime();
        if(directory_input_p)
        {
            objects_detection::input_image_const_view_t input_view = directory_input_p->get_image();
            objects_detection::set_monocular_image(input_view);
        }
        else
        {
            objects_detection::input_image_const_view_t
                    left_view(video_input_p->get_left_image()),
                    right_view(video_input_p->get_right_image());
            objects_detection::set_rectified_stereo_images_pair(left_view, right_view);
        }

        const double start_objects_detector_compute_wall_time = omp_get_wtime();

        const bool use_async_api = false;
        //const bool use_async_api = true;
        if(use_async_api)
        {
            objects_detection::compute_async();
            while(objects_detection::detections_are_ready() == false)
            {
                // wait for the object detection computation have finished
                // (should be doing something usefull in the mean time)
                const int sleep_milliseconds = 50;
                boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_milliseconds));
                //printf("Slept %i milliseconds\n", sleep_milliseconds);
            }
        }
        else
        {
            objects_detection::compute();
        }
        cumulated_objects_detector_compute_time += omp_get_wtime() - start_objects_detector_compute_wall_time;

        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        if(should_save_detections)
        {
            filesystem::path image_path = directory_input_p->get_image_path();
            objects_detection::record_detections(image_path, objects_detection::get_detections());
        }

#if not defined(MONOCULAR_OBJECTS_DETECTION_LIB)
        const bool print_stixels_info = true;
        if(print_stixels_info)
        {
            const objects_detection::stixels_t stixels = objects_detection::get_stixels();
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
#endif

        num_iterations += 1;

        if(should_print and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        if(should_print and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average total objects detection speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_processing_time , num_iterations );
        }

        if(should_print and ((num_iterations % num_iterations_for_processing_timing) == 0))
        {
            printf("Average objects detection compute only speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_objects_detector_compute_time , num_iterations );
        }

        // retrieve next rectified input stereo pair
        if(directory_input_p)
        {
            video_input_is_available = directory_input_p->next_frame();
        }
        else
        {
            video_input_is_available = video_input_p->next_frame();
        }
    } // end of "while video input"

    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average objects detection speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    return;
}

} // end of namespace objects_detection
