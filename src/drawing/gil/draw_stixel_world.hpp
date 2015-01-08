#ifndef DOPPIA_DRAW_STIXEL_WORLD_HPP
#define DOPPIA_DRAW_STIXEL_WORLD_HPP

#include "video_input/AbstractVideoInput.hpp"

#include "stereo_matching/stixels/Stixel.hpp"

#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <Eigen/Core>

/// This file contains a collection of stixel world drawing methods that do not depend on BaseSdlGui

namespace doppia {

// forward declarations
class MetricCamera;
class StereoCameraCalibration;
class StixelWorldApplication;
class StixelsEstimator;
class ImagePlaneStixelsEstimator;
class AbstractStixelWorldEstimator;
class AbstractStixelMotionEstimator;


void draw_stixels_estimation(const StixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view);

void draw_stixels_estimation(const ImagePlaneStixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view);

void draw_stixel_match_lines(boost::gil::rgb8_view_t& left_screen_view,
                             boost::gil::rgb8_view_t& right_screen_view,
                             const stixels_t& left_screen_stixels,
                             const stixels_t& right_screen_stixels,
                             const std::vector< int >& stixel_matches );

void draw_stixel_world(const stixels_t &the_stixels,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       const AbstractVideoInput::input_image_view_t &right_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view);

void draw_stixel_world(const stixels_t &the_stixels,
                       const Eigen::MatrixXf &depth_map,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view);

void draw_the_stixels(boost::gil::rgb8_view_t &view, const stixels_t &the_stixels);


} // end of namespace doppia

#endif // DOPPIA_DRAW_STIXEL_WORLD_HPP
