#ifndef DRAW_GROUND_LINE_HPP
#define DRAW_GROUND_LINE_HPP

#include <boost/gil/typedefs.hpp>

namespace doppia {

// forward declarations
class GroundPlane;
class MetricCamera;

/// x,y and height are in [meters]
void draw_ground_line(
    boost::gil::rgb8_view_t &view,
    const MetricCamera& camera,
    const GroundPlane &ground_plane,
    const boost::gil::rgb8c_pixel_t &color,
    const float x1, const float y1,
    const float x2, const float y2,
    const float height = 0);

} // end of namespace doppia

#endif // DRAW_GROUND_LINE_HPP
