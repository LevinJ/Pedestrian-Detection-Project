#ifndef DOPPIA_DRAW_THE_GROUND_CORRIDOR_HPP
#define DOPPIA_DRAW_THE_GROUND_CORRIDOR_HPP

#include "video_input/AbstractVideoInput.hpp"

#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

namespace doppia {

// forward declarations
class MetricCamera;
class GroundPlane;
class GroundPlaneEstimator;
class GroundPlane;
class FastGroundPlaneEstimator;


void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration, boost::gil::rgb8_view_t &screen_view);

void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration, boost::gil::rgb8_view_t &screen_view);


/// Draws the pedestrians bottom and top planes
void draw_the_ground_corridor(boost::gil::rgb8_view_t &view,
                              const MetricCamera& camera,
                              const GroundPlane &ground_plane);

} // end of namespace doppia

#endif // DOPPIA_DRAW_THE_GROUND_CORRIDOR_HPP
