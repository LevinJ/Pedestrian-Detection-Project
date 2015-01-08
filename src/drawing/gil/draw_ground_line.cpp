#include "draw_ground_line.hpp"

#include "line.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"

#include "video_input/MetricCamera.hpp"

#include "colors.hpp"

#include <boost/gil/image_view.hpp>
#include <Eigen/Core>

namespace doppia {

// index for u component, index for v component
enum { i_u=0, i_v=1 };

/// x,y and height are in [meters]
void draw_ground_line(
    boost::gil::rgb8_view_t &view,
    const MetricCamera& camera,
    const GroundPlane &ground_plane,
    const boost::gil::rgb8c_pixel_t &color,
    const float x1, const float y1,
    const float x2, const float y2,
    const float height)
{

    const Eigen::Vector2f uv_point1 =
            camera.project_ground_plane_point(ground_plane, x1, y1, height);

    const Eigen::Vector2f uv_point2 =
            camera.project_ground_plane_point(ground_plane, x2, y2, height);

    draw_line(view, color,
              uv_point1(i_u), uv_point1(i_v),
              uv_point2(i_u), uv_point2(i_v));
    return;
} // end of function draw_ground_line


} // end of namespace doppia
