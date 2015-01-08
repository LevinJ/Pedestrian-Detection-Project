#include "draw_the_ground_corridor.hpp"

#include "video_input/MetricCamera.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"

#include "drawing/gil/colors.hpp"
#include "drawing/gil/hsv_to_rgb.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/line.hpp"
#include "drawing/gil/draw_matrix.hpp"

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "helpers/xyz_indices.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>
#include <boost/gil/extension/io/png_io.hpp>


namespace doppia {

MODULE_LOG_MACRO("draw_the_ground_corridor")

// small helper function
template<typename ViewType, typename PixelType>
void draw_v_disparity_line(ViewType &view, PixelType &color, const GroundPlaneEstimator::line_t &v_disparity_line)
{
    const int x1 = 0, y1 = v_disparity_line.origin()(0);
    const int x2 = view.width(),
            y2 = v_disparity_line.origin()(0) + x2*v_disparity_line.direction()(0);

    draw_line(view, color, x1, y1, x2, y2);

    // printf("draw_v_disparity_line (origin, direction) == (%.2f, %.2f) -> (x2, y2) == (%i, %i)\n",
    //        v_disparity_line.origin()(0), v_disparity_line.direction()(0),
    //        x2, y2);
    return;
}


void draw_v_disparity_lines(const BaseGroundPlaneEstimator &ground_plane_estimator,
                            const StereoCameraCalibration &stereo_calibration,
                            const boost::gil::rgb8_view_t &screen_subview)
{


    // draw image center horizontal line --
    {
        const int image_center_y =
                stereo_calibration.get_left_camera_calibration().get_image_center_y();
        draw_line(screen_subview, rgb8_colors::gray,
                  0, image_center_y, screen_subview.width() -1, image_center_y);
    }


    // draw bounds on expected ground plane --
    {
        // plus line -
        draw_v_disparity_line(screen_subview, rgb8_colors::pink,
                              ground_plane_estimator.get_prior_max_v_disparity_line());

        // minus line -
        draw_v_disparity_line(screen_subview, rgb8_colors::pink,
                              ground_plane_estimator.get_prior_min_v_disparity_line());

        { // prior line -
            const GroundPlane &ground_plane_prior = ground_plane_estimator.get_ground_plane_prior();
            const GroundPlaneEstimator::line_t prior_v_disparity_line =
                    ground_plane_estimator.ground_plane_to_v_disparity_line(ground_plane_prior);
            draw_v_disparity_line(screen_subview, rgb8_colors::violet, prior_v_disparity_line);
        }

        if(false)
        {
            const GroundPlane &ground_plane_prior = ground_plane_estimator.get_ground_plane_prior();
            log.debug() << "draw_ground_plane_estimator ground_plane_prior height == "
                        << ground_plane_prior.offset() << " [meters]" << std::endl;
            const float theta = -std::asin(ground_plane_prior.normal()(i_z));
            log.debug() << "draw_ground_plane_estimator ground_plane_prior theta == "
                        << theta << " [radians] == " << (180/M_PI)*theta << " [degrees]" << std::endl;
        }
    }

    // draw estimated ground --
    {
        const GroundPlane &ground_plane_estimate = ground_plane_estimator.get_ground_plane();

        const GroundPlaneEstimator::line_t v_disparity_line =
                ground_plane_estimator.ground_plane_to_v_disparity_line(ground_plane_estimate);

        draw_v_disparity_line(screen_subview, rgb8_colors::yellow, v_disparity_line);

        if(false)
        {
            log.debug() << "ground plane estimate v_disparity_line has " <<
                           "origin == " << v_disparity_line.origin()(0) << " [pixels]"
                           " and direction == " << v_disparity_line.direction()(0) << " [-]" <<
                           std::endl;
        }

    }

    return;
}




void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration,
                                 boost::gil::rgb8_view_t &screen_view)
{

    // copy right screen image ---
    copy_and_convert_pixels(input_view, screen_view);

    // draw v-disparity image in the right screen image --
    GroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_const_view =
            ground_plane_estimator.get_raw_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_view,
                                                                       0,0,
                                                                       raw_v_disparity_const_view.width(),
                                                                       raw_v_disparity_const_view.height());
    copy_and_convert_pixels(raw_v_disparity_const_view, screen_subview);

    // copy v disparity into the blue channel -
    GroundPlaneEstimator::v_disparity_const_view_t v_disparity_const_view =
            ground_plane_estimator.get_v_disparity_view();
    copy_pixels(v_disparity_const_view, boost::gil::kth_channel_view<0>(screen_subview));


    draw_v_disparity_lines(ground_plane_estimator,
                           stereo_calibration,
                           screen_subview);

    return;
} // end of draw_ground_plane_estimator(...)



void v_disparity_data_to_matrix(const FastGroundPlaneEstimator::v_disparity_data_t &image_data,
                                Eigen::MatrixXf &image_matrix)
{
    typedef FastGroundPlaneEstimator::v_disparity_data_t::const_reference row_slice_t;

    const int rows = image_data.shape()[0], cols = image_data.shape()[1];
    image_matrix.setZero(rows, cols);
    for(int row=0; row < rows; row +=1)
    {
        row_slice_t row_slice = image_data[row];
        row_slice_t::const_iterator data_it = row_slice.begin();
        for(int col=0; col < cols; ++data_it, col+=1)
        {
            //printf("%.f\n", *data_it);
            image_matrix(row, col) = *data_it;
        } // end "for each column"

    } // end of "for each row"

    return;
}


void normalize_each_row(Eigen::MatrixXf &matrix)
{
    //const float row_max_value = 255.0f;
    const float row_max_value = 1.0f;

    for(int row=0; row < matrix.rows(); row += 1)
    {
        const float t_min = matrix.row(row).minCoeff();
        const float t_max = matrix.row(row).maxCoeff();
        matrix.row(row).array() -= t_min;
        matrix.row(row) *= row_max_value/ (t_max - t_min);
    } // end of "for each row"

    return;
}


void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration,
                                 boost::gil::rgb8_view_t &screen_view)
{

    // copy right screen image ---
    copy_and_convert_pixels(input_view, screen_view);

    // draw v-disparity image in the right screen image --
    FastGroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_view =
            ground_plane_estimator.get_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_view,
                                                                       0,0,
                                                                       raw_v_disparity_view.width(),
                                                                       screen_view.height());
    fill_pixels(screen_subview, boost::gil::rgb8_pixel_t());

    boost::gil::rgb8_view_t screen_subsubview = boost::gil::subimage_view(screen_subview,
                                                                          0, raw_v_disparity_view.height(),
                                                                          raw_v_disparity_view.width(),
                                                                          raw_v_disparity_view.height());
    Eigen::MatrixXf v_disparity_data;
    v_disparity_data_to_matrix(ground_plane_estimator.get_v_disparity(),
                               v_disparity_data);
    normalize_each_row(v_disparity_data);
    draw_matrix(v_disparity_data, screen_subsubview);

    if(false)
    {
        log.debug() << "(over)Writing ground_v_disparity_data.png" << std::endl;
        boost::gil::png_write_view("ground_v_disparity_data.png", screen_subsubview);
    }

    const bool draw_lines_on_top = true;
    if(draw_lines_on_top)
    {
        draw_v_disparity_lines(ground_plane_estimator,
                               stereo_calibration,
                               screen_subview);

        // draw the points used to estimate the objects
        typedef std::pair<int, int> point_t;
        const FastGroundPlaneEstimator::points_t &points = ground_plane_estimator.get_points();
        BOOST_FOREACH(const point_t &point, points)
        {
            *screen_subsubview.at(point.first, point.second) = rgb8_colors::orange;
        }
    }
    return;
} // end of StixelWorldGui::draw_ground_plane_estimator(const FastGroundPlaneEstimator &)


/// Draws the pedestrians bottom and top planes
void draw_the_ground_corridor(boost::gil::rgb8_view_t &view,
                              const MetricCamera& camera,
                              const GroundPlane &ground_plane)
{

    //const float min_z = 10, max_z = 50, z_step = 10;
    const float min_z = 2, max_z = 20, z_step = 1;
    const float far_left = -5.5; // -2.5 // [meters]
    const float far_right= 5.5; // 5.5  // [meters]

    const float average_person_height = 1.8; // [meters]

    for (float z = min_z; z <= max_z; z+= z_step)
    {
        draw_ground_line(view, camera, ground_plane, rgb8_colors::blue,
                         far_left, z, far_right, z, 0.0);
        draw_ground_line(view, camera, ground_plane, rgb8_colors::red,
                         far_left, z, far_right, z, average_person_height);
    }

    for (float x = far_left; x <= far_right; x += 1) {
        draw_ground_line(view, camera, ground_plane, rgb8_colors::blue,
                         x, 2, x, 10, 0.0);
        draw_ground_line(view, camera, ground_plane, rgb8_colors::red,
                         x, 2, x, 10, average_person_height);
    }

    draw_ground_line(view, camera, ground_plane, rgb8_colors::dark_blue,
                     far_left, 5, far_right, 5, 0.0);
    draw_ground_line(view, camera, ground_plane, rgb8_colors::dark_red,
                     far_left, 5, far_right, 5, average_person_height);

    // draw horizon line --
    draw_horizon_line(view, camera, ground_plane, rgb8_colors::dark_green);

    return;
}


} // end of namespace doppia
