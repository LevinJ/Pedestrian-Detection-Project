#ifndef DRAW_HORIZON_LINE_HPP
#define DRAW_HORIZON_LINE_HPP

#include <boost/gil/typedefs.hpp>

namespace doppia {

// forward declarations
class GroundPlane;
class MetricCamera;

void draw_horizon_line(
    boost::gil::rgb8_view_t &view,
    const MetricCamera& camera,
    const GroundPlane &ground_plane,
    const boost::gil::rgb8c_pixel_t &color);

} // end of namespace doppia

#endif // DRAW_HORIZON_LINE_HPP
