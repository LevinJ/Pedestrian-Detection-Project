#ifndef OBJECTS_DETECTION_OBJECTS_DETECTION_LIB_HPP
#define OBJECTS_DETECTION_OBJECTS_DETECTION_LIB_HPP

#include "objects_detection/Detection2d.hpp"

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
#include "video_input/calibration/CameraCalibration.hpp"
#else
#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "stereo_matching/stixels/Stixel.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"
#endif


#include <boost/gil/typedefs.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>
#include <string>

namespace objects_detection {

typedef doppia::Detection2d detection_t;
typedef std::vector<detection_t> detections_t;

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
// dummy definitions
typedef int ground_plane_t;
typedef int stixels_t;
#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
typedef doppia::GroundPlane ground_plane_t;
typedef doppia::stixels_t stixels_t;
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not


typedef boost::gil::rgb8c_view_t input_image_const_view_t;

/// Initialization functions
/// @{
void init_objects_detection(const boost::filesystem::path configuration_filepath,
                            const bool use_ground_plane = false, const bool use_stixels = false);

void init_objects_detection(const boost::program_options::variables_map options,
                            const bool use_ground_plane = false, const bool use_stixels = false);

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
void init_objects_detection(const boost::program_options::variables_map options,
                            boost::shared_ptr<doppia::CameraCalibration> calibration_p,
                            const bool use_ground_plane = false);
#else
/// this initialization function does not use the video_input.calibration_filename option
/// the camera calibration is given directly
void init_objects_detection(const boost::program_options::variables_map options,
                            boost::shared_ptr<doppia::StereoCameraCalibration> stereo_calibration_p,
                            const bool use_ground_plane = false, const bool use_stixels = false);
#endif // defined(MONOCULAR_OBJECTS_DETECTION_LIB)
/// @}


/// Input functions
/// @{

void set_monocular_image(input_image_const_view_t &input_image);

void set_rectified_stereo_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

/// @}


/// Compute functions
/// @{

/// blocking call to compute the detections
void compute();

/// non-blocking call to launch the detections, pool detections_are_ready to check for new results
void compute_async();

/// @}


/// Output functions
/// @{

/// returns true if the detection task launched with compute_async has finished
bool detections_are_ready();

/// returns a copy of the current detections, will raise an exception if detections_are_ready is currently false
const detections_t get_detections();

/// returns a copy of the current ground plane estimate,
/// assumes stereo input is available,
/// when using compute_async should only be called when detections are ready
const ground_plane_t get_ground_plane();

/// returns a copy of the current stixels estimate,
/// assumes stereo input is available,
/// when using compute_async should only be called when detections are ready
const stixels_t get_stixels();

/// @}

/// helper function to record a sequence of detections in a .data_sequence file
void record_detections(const boost::filesystem::path &image_path,
                       const detections_t &detections,
                       const int additional_border = 0);

/// helper function used by the test applications and for applications that want to parse the options by themselves
void get_options_description(boost::program_options::options_description &desc);


/// Helper function to avoid having harmless CUDA de-allocation error at exit time.
void free_object_detector();


} // end namespace objects_detection

#endif // OBJECTS_DETECTION_OBJECTS_DETECTION_LIB_HPP
