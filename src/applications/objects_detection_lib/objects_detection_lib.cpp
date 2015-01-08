#include "objects_detection_lib.hpp"

#include "video_input/preprocessing/AddBorderFunctor.hpp"

#include "objects_detection/AbstractObjectsDetector.hpp"
#include "objects_detection/ObjectsDetectorFactory.hpp"

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
#include "video_input/AbstractVideoInput.hpp"

#else // not defined(MONOCULAR_OBJECTS_DETECTION_LIB)

#include "video_input/VideoInputFactory.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/StixelWorldEstimatorFactory.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeEstimatorFactory.hpp"

#endif // defined(MONOCULAR_OBJECTS_DETECTION_LIB)

#include "video_input/MetricCamera.hpp"
#include "video_input/MetricStereoCamera.hpp"


#include "stereo_matching/ground_plane/AbstractGroundPlaneEstimator.hpp"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"


#if defined(OBJECTS_DETECTION_WITH_UI_LIB)

#include "ObjectsDetectionLibGui.hpp"

#else
#include "FakeObjectsDetectionLibGui.hpp"

namespace objects_detection {
// we cannot use a typedef since it would be incompatible with forward declarations,
// we use a child class instead.
//typedef FakeObjectsDetectionLibGui ObjectsDetectionLibGui;
class ObjectsDetectionLibGui: public FakeObjectsDetectionLibGui
{
public:
};
} // end of namespace objects_detection
#endif


#include "helpers/get_option_value.hpp"
#include "helpers/data/DataSequence.hpp"
#include "helpers/Log.hpp"
#include "helpers/replace_environment_variables.hpp"

#include "objects_detection/detections.pb.h"

#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>

#include <boost/gil/extension/io/png_io.hpp>

#include <iostream>
#include <fstream>


namespace objects_detection {

using namespace std;
using namespace boost;
using namespace doppia;

scoped_ptr<AddBorderFunctor> add_border_p;

shared_ptr<AbstractObjectsDetector> objects_detector_p;
shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p;

scoped_ptr<ObjectsDetectionLibGui> gui_p;

typedef DataSequence<doppia_protobuf::Detections> DetectionsDataSequence;
scoped_ptr<DetectionsDataSequence> detections_data_sequence_p;

stixels_t stixels_from_previous_frame;

typedef AbstractStixelWorldEstimator::ground_plane_corridor_t ground_plane_corridor_t;
ground_plane_corridor_t ground_corridor_from_previous_frame;

/// options are stored for the delayed StixelWorldEstimator instanciation
boost::program_options::variables_map options;

boost::gil::rgb8_image_t::point_t input_dimensions;

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
shared_ptr<CameraCalibration> camera_calibration_p;
shared_ptr<MetricCamera> metric_camera_p;
#else  // not monocular
shared_ptr<StereoCameraCalibration> stereo_calibration_p;
shared_ptr<MetricStereoCamera> stereo_camera_p;
boost::gil::rgb8_image_t left_image, right_image;
#endif // defined(MONOCULAR_OBJECTS_DETECTION_LIB)

float
ground_plane_prior_pitch = 0, // [radians]
ground_plane_prior_roll = 0, // [radians]
ground_plane_prior_height = 1.0; // [meters]

bool
first_frame = true,
should_use_ground_plane = false, // used to enable ground plane usage in monocular case
should_use_stixels = false;


void get_options_description(boost::program_options::options_description &desc)
{
    desc.add_options()
            ("video_input.additional_border",
             boost::program_options::value<int>()->default_value(0),
             "when using process_folder, will add border to the image to enable detection of cropped pedestrians. "
             "Value is in pixels (e.g. 50 pixels)")

        #if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
            // we allow input file options, even if we do not use them
            ("video_input.images_folder,i", program_options::value<string>(),
             "path to a directory with monocular images. This option will overwrite left/right_filename_mask values")
        #endif
            ;

    // Objects detection options --
    desc.add(ObjectsDetectorFactory::get_args_options());

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    desc.add(AbstractVideoInput::get_args_options());

#else // not defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    // we allow input file options, even if we do not use them
    desc.add(VideoInputFactory::get_args_options());

    // Stixel world estimation options --
    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(DisparityCostVolumeEstimatorFactory::get_args_options());
    desc.add(StixelWorldEstimatorFactory::get_args_options());

#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not


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
            throw std::runtime_error("objects_detection cannot run without a configuration file");
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



void init_objects_detection(const boost::filesystem::path configuration_filepath,
                            const bool use_ground_plane, const bool use_stixels)
{
    const boost::program_options::variables_map options = parse_configuration_file(configuration_filepath);

    init_objects_detection(options, use_ground_plane, use_stixels);
    return;
}


void init_objects_detection(const boost::program_options::variables_map input_options,
                            const bool use_ground_plane, const bool use_stixels)
{

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)

    if(use_stixels)
    {
        throw std::runtime_error("Stixels computation not available in monocular mode");
    }

    boost::shared_ptr<doppia::CameraCalibration> calibration_p;
    init_objects_detection(input_options, calibration_p, use_ground_plane);

#else // not monocular
    boost::shared_ptr<doppia::StereoCameraCalibration> stereo_calibration_p;

    if(use_ground_plane or use_stixels)
    {

        boost::filesystem::path calibration_filename =
                get_option_value<std::string>(input_options, "video_input.calibration_filename");

        calibration_filename = replace_environment_variables(calibration_filename);

        stereo_calibration_p.reset(new StereoCameraCalibration(calibration_filename.string()));

    }

    init_objects_detection(input_options, stereo_calibration_p, use_ground_plane, use_stixels);

#endif // defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    return;
}


void init_objects_detection(const boost::program_options::variables_map input_options,
                            boost::shared_ptr<doppia::StereoCameraCalibration> input_stereo_calibration_p,
                            const bool use_ground_plane, const bool use_stixels)
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

        //logging::log(logging::ErrorMessage, "objects_detection") << "Test error message" << std::endl;
    }


    const int additional_border = get_option_value<int>(options, "video_input.additional_border");

    if(additional_border > 0)
    {
        add_border_p.reset(new AddBorderFunctor(additional_border));
    }


    objects_detector_p.reset(ObjectsDetectorFactory::new_instance(options));

    if(use_ground_plane or use_stixels)
    {
        should_use_stixels = use_stixels;

#if not defined(MONOCULAR_OBJECTS_DETECTION_LIB)

        if(not input_stereo_calibration_p)
        {
            throw std::invalid_argument("init_objects_detection expects to receive a non-empty stereo calibration shared pointer");
        }

        stereo_calibration_p = input_stereo_calibration_p;
        stereo_camera_p.reset(new MetricStereoCamera(*stereo_calibration_p));

        ground_plane_prior_height = get_option_value<float>(options, "video_input.camera_height");
        ground_plane_prior_pitch = get_option_value<float>(options, "video_input.camera_pitch");
        ground_plane_prior_roll = get_option_value<float>(options, "video_input.camera_roll");

        const bool print_parameters_details = true;

        if(print_parameters_details)
        {
            printf("Camera height == %.4f\n", ground_plane_prior_height);
            printf("Camera pitch == %.4f\n", ground_plane_prior_pitch);

            printf("stereo baseline == %.3f\n", stereo_camera_p->get_calibration().get_baseline() );
            printf("left/right camera focal x,y == (%.3f, %.3f)/(%.3f, %.3f)\n",
                   stereo_camera_p->get_calibration().get_left_camera_calibration().get_focal_length_x(),
                   stereo_camera_p->get_calibration().get_left_camera_calibration().get_focal_length_y(),
                   stereo_camera_p->get_calibration().get_right_camera_calibration().get_focal_length_x(),
                   stereo_camera_p->get_calibration().get_right_camera_calibration().get_focal_length_y() );
        }

        // StixelWorldEstimatorFactory requires the input image dimensions,
        // stixel_world_estimator_p will only be set at the first call of set_rectified_stereo_images_pair
#endif

        // FIXME should have a specialized ground plane only case
    }

    return;
}


#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
void init_objects_detection(const boost::program_options::variables_map input_options,
                            boost::shared_ptr<doppia::CameraCalibration> calibration_p,
                            const bool use_ground_plane)
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

        //logging::log(logging::ErrorMessage, "objects_detection") << "Test error message" << std::endl;
    }


    const int additional_border = get_option_value<int>(options, "video_input.additional_border");

    if(additional_border > 0)
    {
        add_border_p.reset(new AddBorderFunctor(additional_border));
    }


    objects_detector_p.reset(ObjectsDetectorFactory::new_instance(options));

    should_use_stixels = false;

    if(use_ground_plane)
    {

        if(not calibration_p)
        {
            throw std::invalid_argument("When using ground plane, init_objects_detection expects to receive a non-empty calibration shared pointer");
        }


        camera_calibration_p = calibration_p;
        metric_camera_p.reset(new MetricCamera(*camera_calibration_p));

        ground_plane_prior_height = get_option_value<float>(options, "video_input.camera_height");
        ground_plane_prior_pitch = get_option_value<float>(options, "video_input.camera_pitch");
        ground_plane_prior_roll = get_option_value<float>(options, "video_input.camera_roll");

        const bool print_parameters_details = true;

        if(print_parameters_details)
        {
            printf("Camera height == %.4f\n", ground_plane_prior_height);
            printf("Camera pitch == %.4f\n", ground_plane_prior_pitch);

            printf("camera focal x,y == (%.3f, %.3f)\n",
                   metric_camera_p->get_calibration().get_focal_length_x(),
                   metric_camera_p->get_calibration().get_focal_length_y() );
        }

    }

    return;
}
#endif // defined(MONOCULAR_OBJECTS_DETECTION_LIB)



void set_monocular_image(input_image_const_view_t &input_view)
{
    if(first_frame and input_view.size() == 0)
    {
        throw std::runtime_error("objects_detection::set_monocular_image received an image of size 0,0");
    }

    if(stixel_world_estimator_p)
    {
        throw std::runtime_error("Indicated that would stereo features, but provided only a monocular input.");
    }

    if(add_border_p)
    {
        // add the borders (for Markus and Angelos)
        input_image_const_view_t the_input_view = (*add_border_p)(input_view);
        input_dimensions = the_input_view.dimensions();
        objects_detector_p->set_image(the_input_view);
    }
    else
    {
        input_dimensions = input_view.dimensions();
        objects_detector_p->set_image(input_view);
    }

    if(first_frame)
    {
#if defined(OBJECTS_DETECTION_WITH_UI_LIB)
        boost::shared_ptr<doppia::MetricStereoCamera> stereo_camera_p; // left empty for monocular case
        gui_p.reset(new ObjectsDetectionLibGui(input_view.width(), input_view.height(), stereo_camera_p,
                                               objects_detector_p, stixel_world_estimator_p));
#endif
    }

    if(gui_p)
    {
        gui_p->set_monocular_input(input_view);
    }

    // first frame is set to false in compute()
    return;
}


#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
void set_rectified_stereo_images_pair(input_image_const_view_t &left_view, input_image_const_view_t &right_view)
{
    throw std::runtime_error("set_rectified_stereo_images_pair(...) not implemented in "
                             "the monocular objects detection library");
    return;
}

#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined


void set_rectified_stereo_images_pair(input_image_const_view_t &input_left_view,
                                      input_image_const_view_t &input_right_view)
{

    if(false and (not stereo_camera_p)) // we may not use the stereo at all, even if we receive stereo input
    {
        throw std::runtime_error("set_rectified_stereo_images_pair called but no stereo camera has been set, "
                                 "please double-check your objects_detection::init_objects_detection(...) call");
    }


    input_image_const_view_t left_view, right_view;

    if(input_left_view.dimensions() != input_right_view.dimensions())
    {
        printf("left width, height == %zi, %zi\n", input_left_view.width(), input_left_view.height());
        printf("right width, height == %zi, %zi\n", input_right_view.width(), input_right_view.height());
        throw std::invalid_argument("Input left and right images do not have the same dimensions");
    }

    if(first_frame and input_left_view.size() == 0)
    {
        throw std::runtime_error("objects_detection::set_rectified_stereo_images_pair received "
                                 "a left image of size 0,0");
    }

    if(first_frame and input_right_view.size() == 0)
    {
        throw std::runtime_error("objects_detection::set_rectified_stereo_images_pair received "
                                 "a right image of size 0,0");
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

    input_dimensions = left_view.dimensions();

    if(first_frame)
    {
        printf("first frame for set_rectified_stereo_images_pair\n");
        if(not stixel_world_estimator_p and (should_use_stixels or should_use_ground_plane))
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
        if(stixel_world_estimator_p)
        {
            stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);
        }

        // must set the image in the objects detector too, so it knows right away about the image size
        objects_detector_p->set_image(left_view);

#if defined(OBJECTS_DETECTION_WITH_UI_LIB)
        gui_p.reset(new ObjectsDetectionLibGui(left_view.width(), left_view.height(), stereo_camera_p,
                                               objects_detector_p, stixel_world_estimator_p));
#endif

    }
    else
    {
        // set the input for stixels and objects detection --
        if(stixel_world_estimator_p)
        {
            stixel_world_estimator_p->set_rectified_images_pair(left_view, right_view);
            ground_corridor_from_previous_frame = stixel_world_estimator_p->get_ground_plane_corridor();
            objects_detector_p->set_ground_plane_corridor(ground_corridor_from_previous_frame);

            if(should_use_stixels)
            {
                stixels_from_previous_frame = stixel_world_estimator_p->get_stixels();
                objects_detector_p->set_stixels(stixels_from_previous_frame);
            }
        }

        objects_detector_p->set_image(left_view);
    }


    if(gui_p)
    {
        gui_p->set_left_input(left_view);
        gui_p->set_right_input(right_view);
    }


    // first frame is set to false in compute()
    return;
}
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not



/// pseudo distance is related to distance, but not in meters (more something like 1/(meters*focal_length))
void set_pseudo_distance_given_v(const int num_rows,
                                 const int num_disparities,
                                 const AbstractGroundPlaneEstimator::line_t &v_pseudo_distance_ground_line,
                                 std::vector<int> &pseudo_distance_given_v)
{
    // cost volume data is organized as y (rows), x (columns), disparity
    const int max_disparity = num_disparities - 1;

    pseudo_distance_given_v.resize(num_rows);

    const float v_origin = v_pseudo_distance_ground_line.origin()(0);
    const float direction = v_pseudo_distance_ground_line.direction()(0);
    const float direction_inverse = 1 / direction;

    const bool print_mapping = false;
    static bool first_print = true;

    for(int v=0; v < num_rows; v += 1)
    {
        const float d = (v - v_origin) * direction_inverse;
        pseudo_distance_given_v[v] =
                std::max(0, std::min(max_disparity, static_cast<int>(d)));

        if(print_mapping and first_print)
        {
            printf("pseudo_distance value at v %i == %i\n", v,  pseudo_distance_given_v[v]);
            first_print = false;
        }
    } // end of "for each row"

    return;
} // end of set_disparity_given_v(...)

/*
AbstractGroundPlaneEstimator::line_t ground_plane_to_v_pseudo_distance_line(
        const GroundPlane &ground_plane,
        const float camera_v0, const float camera_focal_length)
{
    const float theta = -ground_plane.get_pitch();
    const float heigth = ground_plane.get_height();


    AbstractGroundPlaneEstimator::line_t line;

    // based on equations 10 and 11 from V-disparity paper of Labayrade, Aubert and Tarel 2002.
    const float v_origin = camera_v0 - camera_focal_length*std::tan(theta);
    const float c_r = cos(theta) / heigth; // c_r would normally be multiplied by the stereo_baseline, to become disparities

    line.origin()(0) = v_origin;
    line.direction()(0) = 1/c_r;

    if(false)
    {
        log_debug() << "ground_plane_to_v_pseudo_distance_line ground_plane theta == "
                    << theta << " [radians] == " << (180/M_PI)*theta << " [degrees]" << std::endl;
        log_debug() << "ground_plane_to_v_pseudo_distance_line ground_plane height == " << heigth << std::endl;
        log_debug() << "ground_plane_to_v_pseudo_distance_line line direction == " << c_r << std::endl;
    }

    return line;
}


/// given a ground plane and the camera calibration, will estimate the ground plane corridor,
/// which is used to constraint the search space during monocular objects detection
void ground_plane_to_ground_plane_corridor(const ground_plane_t &ground_plane,
                                           AbstractStixelWorldEstimator::ground_plane_corridor_t &ground_plane_corridor)
{
    if(input_left_view.size() == 0)
    {
        throw std::runtime_error("Sorry, you need to call set_rectified_images_pair before set_ground_plane_estimate");
    }

    const ground_plane_t &the_ground_plane = ground_plane;
    const AbstractGroundPlaneEstimator::line_t the_v_pseudo_distance_line = ground_plane_to_v_pseudo_distance_line(ground_plane, camera_v0, camera_focal_lenght);

    const int num_rows = input_left_view.height();


    std::vector<int> pseudo_distance_given_v;
    set_pseudo_distance_given_v(num_rows, num_disparities, v_pseudo_distance_ground_line, pseudo_distance_given_v);

    compute_ground_plane_corridor(ground_plane, disparity_given_v, camera,
                                  expected_object_height, minimum_object_height_in_pixels,
                                  ground_plane_corridor);


    std::runtime_error("Not yet implemented");

    return;
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

            //const float depth = camera.disparity_to_depth(disparity);
            const float depth = baseline_times_focal_length / disparity;

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


const int get_object_top_v_from_object_bottom_v(const int bottom_v, const float object_height,
                                                const float ground_plane_height, const float ground_plane_pitch,
                                                const MetricCamera &camera)
{ // let us play trigonometry !

    const float
            center_v = camera.get_calibration().get_image_center_y(),
            delta_v = center_v - bottom_v,
            focal_length = camera.get_calibration().get_focal_length_y(),
            alpha_prime = atan2(delta_v, focal_length),
            alpha = M_PI/2 - alpha_prime,
            beta = M_PI/2 + ground_plane_pitch; // we expect pitch to be negative

    if((alpha + beta) >= M_PI)
    {
        // alpha and beta angles do not build up a triangle
        throw std::runtime_error("get_object_top_v_from_object_bottom_v received an unhandled geometric situation");
    }

    const float
            cos_alpha = cos(alpha),
            cos_beta = cos(beta),
            sin_beta = sin(beta),
            b = ground_plane_height/(cos_beta + (sin_beta/tan_alpha)),
            a = b*sin_beta/sin_alpha,
            y_bottom = -a*cos_alpha,
            z_bottom = b*sin_beta, // == a*sin_alpha
            z_top = z_bottom, // we assume ground plane pitch is very small
            y_top = y_bottom + object_height,  // we assume ground plane pitch is very small
            x_top = 0;

    Eigen::Vector3f top_point3d;
    top_point3d << x_top, y_top, z_top;

    const int top_v = camera.project_3d_point(top_point3d)[1];

    return top_v;
}


/// @param height should be in meters
/// @param pitch should be in radians (usually small negative value)
void set_ground_plane_estimate(const float ground_plane_height, const float ground_plane_pitch)
{
    GroundPlane ground_plane;
    const float roll = 0;
    ground_plane.set_from_metric_units(ground_plane_pitch, roll, ground_plane_height);
    set_ground_plane_estimate(ground_plane);
    return;
}


void set_ground_plane_estimate(const GroundPlane &ground_plane)
{
    if(add_border_p)
    {
        throw std::runtime_error("set_ground_plane_estimate is not yet compatible with the use of additional border (to be implemented).");
    }

    shared_ptr<const MetricCamera> camera_p;

    const float object_height = 1.8; // meters
    const float minimum_object_height_in_pixels = 30; // pixels

    const float far_side = 0, far_front = 1E3, height = 0; // [meters]
    const int horizon_v = camera_p->project_ground_plane_point(ground_plane, -far_side, far_front, height)[1];

    ground_plane_corridor_t ground_plane_corridor;
    ground_plane_corridor.resize(input_dimensions.y());

    // for each row in the image
    for(size_t v=0; v < ground_plane_corridor.size(); v+=1)
    {
        if(v <= horizon_v)
        {
            ground_plane_corridor[v] = -1; // indicates rows above the horizon
        }
        else
        {
            // we need to compute the ground plane corridor
            const int
                    top_v = get_object_top_v_from_object_bottom_v(bottom_v, object_height,
                                                                  ground_plane.get_height(), ground_plane.get_pitch(),
                                                                  *camera_p),
                    small_object_top_v = std::max(bottom_v - minimum_object_height_in_pixels, 0),
                    final_top_v = std::min(top_v, small_object_top_v);

            ground_plane_corridor[v] = final_top_v;
        }

    } // end of "for each row in the image"

    objects_detector_p->set_ground_plane_corridor(ground_plane_corridor);
    return;
}
*/

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
        // launch stixels estimation in a thread, compute the objects detection --
#pragma omp sections
        {
#pragma omp section
            {
                stixel_world_estimator_p->compute();
            }
#pragma omp section
            { // the gpu computation runs in parallel with the stixel world estimation
                //const double start_objects_detector_compute_wall_time = omp_get_wtime();
                objects_detector_p->compute();
                //cumulated_objects_detector_compute_time += omp_get_wtime() - start_objects_detector_compute_wall_time;
            }
        } // end of "pragma omp sections"
    }
    else
    {
        //const double start_objects_detector_compute_wall_time = omp_get_wtime();
        objects_detector_p->compute();
        //cumulated_objects_detector_compute_time += omp_get_wtime() - start_objects_detector_compute_wall_time;
    }

    if(gui_p)
    {
        gui_p->update();
    }

    first_frame = false;
    return;
}


boost::barrier compute_barrier(2); // only two threads are involved in this barrier

boost::promise<detections_t> detections_promise;
boost::unique_future<detections_t> compute_async_detections;
ground_plane_t async_ground_plane;
stixels_t async_stixels;
boost::thread compute_async_thread;


void compute_async_thread_function()
{
    //printf("Entering into compute_async_thread_function infinite loop\n");
    while(true) // FIXME is this a good idea ?
    {
        {
            compute_barrier.wait(); // waiting for compute_async to launch a new computation
            //printf("Computation async sent 'true' to the compute condition\n");
        }

        // call the global method
        compute();

        // we keep ground plane and stixels synchronized with the detections
        if(stixel_world_estimator_p)
        {
#if not defined(MONOCULAR_OBJECTS_DETECTION_LIB)
            async_ground_plane = stixel_world_estimator_p->get_ground_plane();
            async_stixels = stixel_world_estimator_p->get_stixels();
#endif
        }
        detections_promise.set_value(objects_detector_p->get_detections());
    }

    return;
}


/// non-blocking call to launch the detections, pool detections_are_ready to check for new results
void compute_async()
{
    if(compute_async_thread.joinable() == false)
    {
        // thread is not running
        printf("Launching async computation thread\n");

        // launch thread
        compute_async_thread = boost::thread(compute_async_thread_function);

    } // end of "if compute async thread is not yet running"

    // new promise, new future
    {
        detections_promise = boost::promise<detections_t>();
        compute_async_detections = detections_promise.get_future();
    }
    compute_barrier.wait(); // at this point, the computation is launched

    //printf("Exiting compute_async\n");
    return;
}


/// returns true if the detection task launched with compute_async has finished
bool detections_are_ready()
{
    if(compute_async_thread.joinable() == false)
    {

        throw std::runtime_error("compute_async() should be called at least once before calling detections_are_ready()");
    }

    return compute_async_detections.is_ready();
}


void shift_detections(detections_t &detections)
{
    const int additional_border = add_border_p->additional_border;

    BOOST_FOREACH(detection_t &detection, detections)
    {
        detection_t::rectangle_t &bb = detection.bounding_box;
        bb.max_corner().x(bb.max_corner().x() - additional_border);
        bb.max_corner().y(bb.max_corner().y() - additional_border);
        bb.min_corner().x(bb.min_corner().x() - additional_border);
        bb.min_corner().y(bb.min_corner().y() - additional_border);
    }
    return;
}

/// returns a copy of the current detections, should not be called if detections_are_ready returns false
const detections_t get_detections_implementation()
{
    if(compute_async_thread.joinable())
    { // using compute_async

        if(not detections_are_ready())
        {
            throw std::runtime_error("objects_detection::get_detections() was called, but detections_are_ready() is false. "
                                     "Should check that detections are ready before calling get_detections()");
        }

        return compute_async_detections.get();
    }
    else
    { // using blocking api

        return objects_detector_p->get_detections();
    }
}


/// returns a copy of the current detections, should not be called if detections_are_ready returns false
const detections_t get_detections()
{
    if(add_border_p)
    {
        detections_t the_detections = get_detections_implementation();
        shift_detections(the_detections);
        return the_detections;
    }
    else
    {
        return get_detections_implementation();
    }
}


#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
const ground_plane_t get_ground_plane()
{
    throw std::runtime_error("get_ground_plane() not implemented in "
                             "the monocular objects detection library");
    return ground_plane_t();
}

#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
const ground_plane_t get_ground_plane()
{

    if(stixel_world_estimator_p)
    {
        if(compute_async_thread.joinable())
        { // using compute_async

            if(not detections_are_ready())
            {
                throw std::runtime_error("objects_detection::get_ground_plane() was called, but detections_are_ready() is false. "
                                         "Should check that detections computation is ready before calling get_ground_plane()");
            }

            return async_ground_plane;
        }
        else
        { // using blocking api

            return stixel_world_estimator_p->get_ground_plane();
        }
    }
    else
    {
        throw std::runtime_error("Cannot (yet) call get_ground_plane when using monocular images only");
    }
}
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not


#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
const stixels_t get_stixels()
{
    throw std::runtime_error("get_stixels() not implemented in "
                             "the monocular objects detection library");
    return stixels_t();
}

#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
const stixels_t get_stixels()
{

    if(stixel_world_estimator_p)
    {
        if(compute_async_thread.joinable())
        { // using compute_async

            if(not detections_are_ready())
            {
                throw std::runtime_error("objects_detection::get_ground_plane() was called, but detections_are_ready() is false. "
                                         "Should check that detections computation is ready before calling get_ground_plane()");
            }

            return async_stixels;
        }
        else
        { // using blocking api

            return stixel_world_estimator_p->get_stixels();
        }
    }
    else
    {
        throw std::runtime_error("Cannot call get_stixels when using monocular images only");
    }
}
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not


void record_detections(const boost::filesystem::path &image_path,
                       const detections_t &the_detections,
                       const int additional_border)
{

    typedef AbstractObjectsDetector::detections_t detections_t;
    typedef AbstractObjectsDetector::detection_t detection_t;

    if(detections_data_sequence_p == false)
    {

        using namespace boost::posix_time;
        const ptime current_time(second_clock::local_time());

        const string recording_path = boost::str( boost::format("%i_%02i_%02i_%i_detections.data_sequence")
                                                  % current_time.date().year()
                                                  % current_time.date().month().as_number()
                                                  % current_time.date().day()
                                                  % current_time.time_of_day().total_seconds() );

        if(boost::filesystem::exists(recording_path) == true)
        {
            // this should never happen
            printf("detections recording path == %s\n", recording_path.c_str());
            throw std::runtime_error("Detections recording path already exists. Please wait one second and try again.");
        }

        DetectionsDataSequence::attributes_t attributes;
        attributes.insert(std::make_pair("created_by", "ObjectsDetectionApplication"));

        detections_data_sequence_p.reset(new DetectionsDataSequence(recording_path, attributes));

        printf("\nCreated recording file %s\n\n", recording_path.c_str());
    }

    assert(static_cast<bool>(detections_data_sequence_p) == true);

    DetectionsDataSequence::data_type detections_data;

    detections_data.set_image_name(image_path.string());

    BOOST_FOREACH(const detection_t &detection, the_detections)
    {
        doppia_protobuf::Detection *detection_data_p = detections_data.add_detections();

        detection_data_p->set_score(detection.score);

        // we must do the max(0, value) check since the Point2d only accept positives coordinates,
        // saving a negative value would create an integer overflow
        doppia_protobuf::Point2d &max_corner = *(detection_data_p->mutable_bounding_box()->mutable_max_corner());
        max_corner.set_x(std::max(0, detection.bounding_box.max_corner().x() - additional_border));
        max_corner.set_y(std::max(0, detection.bounding_box.max_corner().y() - additional_border));
        doppia_protobuf::Point2d &min_corner = *(detection_data_p->mutable_bounding_box()->mutable_min_corner());
        min_corner.set_x(std::max(0, detection.bounding_box.min_corner().x() - additional_border));
        min_corner.set_y(std::max(0, detection.bounding_box.min_corner().y() - additional_border));

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
}


/// Helper function to avoid having harmless CUDA de-allocation error at exit time.
void free_object_detector()
{
    objects_detector_p.reset();
    stixel_world_estimator_p.reset();
}


} // end of namespace objects_detection
