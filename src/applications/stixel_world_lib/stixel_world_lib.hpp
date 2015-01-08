#ifndef STIXEL_WORLD_LIB_HPP
#define STIXEL_WORLD_LIB_HPP


#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "stereo_matching/stixels/Stixel.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"

#include <boost/gil/typedefs.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>
#include <string>

namespace stixel_world {


typedef doppia::GroundPlane ground_plane_t;
typedef doppia::stixels_t stixels_t;


typedef boost::gil::rgb8c_view_t input_image_const_view_t;

void init_stixel_world(const boost::filesystem::path configuration_filepath);

void init_stixel_world(const boost::program_options::variables_map options);

/// this initialization function does not use the video_input.calibration_filename option
/// the camera calibration is given directly
void init_stixel_world(const boost::program_options::variables_map options,
                            boost::shared_ptr<doppia::StereoCameraCalibration> stereo_calibration_p);

void set_rectified_stereo_images_pair(input_image_const_view_t &left, input_image_const_view_t &right);

/// blocking call to compute the detections
void compute();

/// returns a copy of the current ground plane estimate,
/// assumes stereo input is available,
/// when using compute_async should only be called when detections are ready
const ground_plane_t get_ground_plane();


/// returns a copy of the current stixels estimate,
/// assumes stereo input is available,
/// when using compute_async should only be called when detections are ready
const stixels_t get_stixels();

/// helper function used by the test applications and for applications that want to parse the options by themselves
void get_options_description(boost::program_options::options_description &desc);

} // end namespace stixel_world

#endif // STIXEL_WORLD_LIB_HPP
