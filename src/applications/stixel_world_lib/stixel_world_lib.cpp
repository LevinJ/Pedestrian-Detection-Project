#include "stixel_world_lib.hpp"

#include "video_input/VideoInputFactory.hpp"

#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"
#include "video_input/MetricStereoCamera.hpp"

#if defined(STIXEL_WORLD_WITH_UI_LIB)
#include "StixelWorldLibGui.hpp"
#endif

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"
#include "helpers/replace_environment_variables.hpp"

#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>

#include <boost/gil/extension/io/png_io.hpp>

#include <iostream>
#include <fstream>


namespace stixel_world {

using namespace std;
using namespace boost;
using namespace doppia;


#if not defined(STIXEL_WORLD_WITH_UI_LIB)
class FakeStixelWorldLibGui
{
public:
    void set_left_input(input_image_const_view_t);
    void set_right_input(input_image_const_view_t);

    void update();
};


void FakeStixelWorldLibGui::set_left_input(input_image_const_view_t)
{
    return; // we do nothing, at all
}

void FakeStixelWorldLibGui::set_right_input(input_image_const_view_t)
{
    return; // we do nothing, at all
}

void FakeStixelWorldLibGui::update()
{
    return; // we do nothing, at all
}


typedef FakeStixelWorldLibGui StixelWorldLibGui;
#endif

shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p;
scoped_ptr<StixelWorldLibGui> gui_p;

/// options are stored for the delayed StixelWorldEstimator instanciation
boost::program_options::variables_map options;

shared_ptr<StereoCameraCalibration> stereo_calibration_p;
shared_ptr<MetricStereoCamera> stereo_camera_p;
float
ground_plane_prior_pitch = 0, // [radians]
ground_plane_prior_roll = 0, // [radians]
ground_plane_prior_height = 1.0; // [meters]

boost::gil::rgb8_image_t left_image, right_image;

bool first_frame = true;


void get_options_description(boost::program_options::options_description &desc)
{
    desc.add_options()
            ("video_input.additional_border",
             boost::program_options::value<int>()->default_value(0),
             "when using process_folder, will add border to the image to enable detection of cropped pedestrians. "
             "Value is in pixels (e.g. 50 pixels)")

            ;


    // we allow input file options, even if we do not use them
    desc.add(VideoInputFactory::get_args_options());

    // Stixel world estimation options --
    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(DisparityCostVolumeEstimatorFactory::get_args_options());
    desc.add(StixelWorldEstimatorFactory::get_args_options());

    return;
}


boost::program_options::variables_map parse_configuration_file(const boost::filesystem::path &configuration_filepath)
{

    boost::program_options::variables_map options;

    boost::program_options::options_description desc("Allowed options");

    get_options_description(desc);

    const string configuration_filename = configuration_filepath.string();

    if (configuration_filename.empty() == false)
    {
        boost::filesystem::path configuration_file_path(configuration_filename);
        if(boost::filesystem::exists(configuration_file_path) == false)
        {
            cout << "\033[1;31mCould not find the configuration file:\033[0m "
                 << configuration_file_path << endl;
            throw std::runtime_error("stixel_world cannot run without a configuration file");
        }

        try
        {
            fstream configuration_file;
            configuration_file.open(configuration_filename.c_str(), fstream::in);
            boost::program_options::store(
                        boost::program_options::parse_config_file(configuration_file, desc), options);
            configuration_file.close();
        }
        catch (...)
        {
            cout << "\033[1;31mError parsing the configuration file named:\033[0m "
                 << configuration_filename << endl;
            cout << desc << endl;
            throw;
        }

        cout << "Parsed the configuration file " << configuration_filename << std::endl;
    }
    else
    {
        std::runtime_error("Configuration file path is emtpy");
    }


    return options;
}



void init_stixel_world(const boost::filesystem::path configuration_filepath)
{
    const boost::program_options::variables_map options = parse_configuration_file(configuration_filepath);

    init_stixel_world(options);
    return;
}


void init_stixel_world(const boost::program_options::variables_map input_options)
{

    boost::shared_ptr<doppia::StereoCameraCalibration> stereo_calibration_p;

    boost::filesystem::path calibration_filename =
            get_option_value<std::string>(input_options, "video_input.calibration_filename");

    calibration_filename = replace_environment_variables(calibration_filename);

    stereo_calibration_p.reset(new StereoCameraCalibration(calibration_filename.string()));

    init_stixel_world(input_options, stereo_calibration_p);
    return;
}


void init_stixel_world(const boost::program_options::variables_map input_options,
                       boost::shared_ptr<doppia::StereoCameraCalibration> input_stereo_calibration_p)
{

    options = input_options; // we store the options for future use

    // setup the logging
    {
        logging::get_log().clear(); // we reset previously existing options

        // set our own stdout rules and set cout as console stream --
        logging::LogRuleSet rules_for_stdout;
        rules_for_stdout.add_rule(logging::ErrorMessage, "*"); // we only print errors

        rules_for_stdout.add_rule(logging::WarningMessage, "*"); // also print warnings

        logging::get_log().set_console_stream(std::cout, rules_for_stdout);

        //logging::log(logging::ErrorMessage, "stixel_world") << "Test error message" << std::endl;
    }

    stereo_calibration_p = input_stereo_calibration_p;
    stereo_camera_p.reset(new MetricStereoCamera(*stereo_calibration_p));

    ground_plane_prior_height = get_option_value<float>(options, "video_input.camera_height");
    ground_plane_prior_pitch = get_option_value<float>(options, "video_input.camera_pitch");
    ground_plane_prior_roll = get_option_value<float>(options, "video_input.camera_roll");

    // StixelWorldEstimatorFactory requires the input image dimensions,
    // stixel_world_estimator_p will only be set at the first call of set_rectified_stereo_images_pair
    return;
}


void set_rectified_stereo_images_pair(input_image_const_view_t &input_left_view,
                                      input_image_const_view_t &input_right_view)
{

    input_image_const_view_t left_view, right_view;

    if(input_left_view.dimensions() != input_right_view.dimensions())
    {
        printf("left width, height == %zi, %zi\n", input_left_view.width(), input_left_view.height());
        printf("right width, height == %zi, %zi\n", input_right_view.width(), input_right_view.height());
        throw std::invalid_argument("Input left and right images do not have the same dimensions");
    }

    const bool copy_input_images = true; // just to be safe
    if(copy_input_images)
    {
        // lazy re-allocation
        left_image.recreate(input_left_view.dimensions());
        right_image.recreate(input_right_view.dimensions());

        boost::gil::copy_pixels(input_left_view, boost::gil::view(left_image));
        boost::gil::copy_pixels(input_right_view, boost::gil::view(right_image));

        left_view = boost::gil::const_view(left_image);
        right_view = boost::gil::const_view(right_image);

        //boost::gil::png_write_view("right_view.png", right_view);
        //throw std::runtime_error("created right_view.png");
    }
    else
    {
        left_view = input_left_view;
        right_view = input_right_view;
    }

    if(first_frame)
    {
        printf("first frame for set_rectified_stereo_images_pair\n");
        if(not stixel_world_estimator_p)
        {
            stixel_world_estimator_p.reset(
                        StixelWorldEstimatorFactory::new_instance(options,
                                                                  left_view.dimensions(),
                                                                  *stereo_camera_p,
                                                                  ground_plane_prior_pitch,
                                                                  ground_plane_prior_roll,
                                                                  ground_plane_prior_height));
        }


        // in the first frame we will not output any detection, that starts in the second frame
        stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);

#if defined(STIXEL_WORLD_WITH_UI_LIB)
        gui_p.reset(new StixelWorldLibGui(left_view.width(), left_view.height(), stereo_camera_p,
                                          stixel_world_estimator_p));
#endif
    }
    else
    {
        // set the input for stixels --
        stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);
    }


    if(gui_p)
    {
        gui_p->set_left_input(left_view);
        gui_p->set_right_input(right_view);
    }


    // first frame is set to false in compute()
    return;
}


/// blocking call to compute the detections
void compute()
{
    // if(thread_launched) raise exception, cannot mix both operation modes

    if(first_frame and stixel_world_estimator_p)
    {
        printf("first frame for compute\n");
        stixel_world_estimator_p->compute();
    }
    else if(stixel_world_estimator_p)
    {
        stixel_world_estimator_p->compute();
    }
    else
    {
        throw std::runtime_error("stixel_world_estimator_p does not exist, did you call init_stixel_world ?");
    }

    if(gui_p)
    {
        gui_p->update();
    }

    first_frame = false;
    return;
}


const ground_plane_t get_ground_plane()
{

    if(stixel_world_estimator_p)
    {

        return stixel_world_estimator_p->get_ground_plane();
    }
    else
    {
        throw std::runtime_error("stixel_world_estimator_p does not exist, did you call init_stixel_world ?");
    }
}


const stixels_t get_stixels()
{

    if(stixel_world_estimator_p)
    {

        return stixel_world_estimator_p->get_stixels();
    }
    else
    {
        throw std::runtime_error("stixel_world_estimator_p does not exist, did you call init_stixel_world ?");
    }
}




} // end of namespace stixel_world
