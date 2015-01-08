#include "draw_stixel_world.hpp"

#include "drawing/gil/colors.hpp"
#include "drawing/gil/hsv_to_rgb.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/line.hpp"
#include "drawing/gil/draw_matrix.hpp"
#include "drawing/gil/draw_the_ground_corridor.hpp"

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/ImagePlaneStixelsEstimator.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "helpers/xyz_indices.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>
#include <boost/gil/extension/io/png_io.hpp>


namespace doppia {

MODULE_LOG_MACRO("draw_stixel_world")

using namespace std;

void draw_the_stixels(boost::gil::rgb8_view_t &view, const stixels_t &the_stixels)
{

    const bool do_no_draw_stixels = false;
    //const bool do_no_draw_stixels = true; // only used for explanatory video creation
    if(do_no_draw_stixels)
    {
        return;
    }

    //const bool color_encoded_depth = true;
    const bool color_encoded_depth = true;

    const float saturation  = 0.7, value = 0.9;

    // Burnt orange (204, 85, 0)
    // according to http://en.wikipedia.org/wiki/Orange_(colour)
    const boost::gil::rgb8c_pixel_t burnt_orange =
            boost::gil::rgb8_view_t::value_type(204, 85, 0);


    const int line_tickness = 8;

    const int height = view.height();
    BOOST_FOREACH(const Stixel &t_stixel, the_stixels)
    {

        float hue = t_stixel.disparity / 128.0f;
        hue = 1 - (hue/2 + 0.5);
        const boost::gil::rgb8c_pixel_t depth_color = hsv_to_rgb(hue, saturation, value);

        const int &x = t_stixel.x;
        if(t_stixel.type == Stixel::Occluded)
        {
            view(x, t_stixel.top_y) = rgb8_colors::dark_violet;
            view(x, t_stixel.bottom_y) = rgb8_colors::dark_violet;
        }
        else
        {
            const bool draw_top = true;
            //const bool draw_top = false; // only used for explanatory video creation
            if(draw_top)
            {
                if(t_stixel.default_height_value == false)
                {
                    for(int y=max<int>(0, t_stixel.top_y-line_tickness); y < t_stixel.top_y; y+=1)
                    {
                        boost::gil::rgb8_pixel_t t_color = rgb8_colors::orange;
                        if(color_encoded_depth)
                        {
                            t_color = depth_color;
                        }

                        view(x, y) = t_color;
                    }
                    //view(x, t_stixel.top_y) = rgb8_colors::orange;
                }
                else
                {
                    boost::gil::rgb8_pixel_t t_color = burnt_orange;
                    if(color_encoded_depth)
                    {
                        t_color = depth_color;
                    }

                    for(int y=max<int>(0, t_stixel.top_y-line_tickness); y < t_stixel.top_y; y+=1)
                    {
                        view(x, y) = t_color;
                    }
                    //view(x, t_stixel.top_y) = t_color;
                }
            } // end of "if draw top"

            for(int y= t_stixel.bottom_y; y < min<int>(height, t_stixel.bottom_y+line_tickness); y+=1)
            {

                boost::gil::rgb8_pixel_t t_color = rgb8_colors::yellow;

                if(color_encoded_depth)
                {
                    t_color = depth_color;
                }

                view(x, y) = t_color;
            }
            //view(x, t_stixel.bottom_y) = rgb8_colors::yellow;
        }
    } // end of "for each stixel in stixels"

    return;
} // end of draw_the_stixels(...)


void draw_stixels_estimation(const StixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view)
{

    // draw left screen ---
    {
        copy_and_convert_pixels(left_input_view, screen_left_view);

        // cost on top -
        StixelsEstimator::u_disparity_cost_t u_disparity_cost =
                stixels_estimator.get_u_disparity_cost();

        // use M cost instead of u_disparity_cost
        //StixelsEstimator::u_disparity_cost_t u_disparity_cost =
        //        stixels_estimator.get_M_cost();

        const bool dirty_normalization_test = false; // FIXME just for debugging
        if(dirty_normalization_test)
        {
            // column wise normalization --
            for(int c=0; c < u_disparity_cost.cols(); c+=1)
            {
                const float col_max_value = u_disparity_cost.col(c).maxCoeff();
                const float col_min_value = u_disparity_cost.col(c).minCoeff();
                u_disparity_cost.col(c).array() -= col_min_value;

                if(col_max_value > col_min_value)
                {
                    const float scaling = 1.0f / (col_max_value - col_min_value);
                    u_disparity_cost.col(c) *= scaling;
                }
            }

            // log scaling --
            u_disparity_cost =
                    (u_disparity_cost.array() + 1).log();
        }


        boost::gil::rgb8_view_t left_top_sub_view =
                boost::gil::subimage_view(screen_left_view,
                                          0, 0,
                                          u_disparity_cost.cols(), u_disparity_cost.rows());

        draw_matrix(u_disparity_cost, left_top_sub_view);


        // draw estimated boundary -
        {
            const std::vector<int> &boundary =
                    stixels_estimator.get_u_disparity_ground_obstacle_boundary();
            for(std::size_t u=0; u < boundary.size(); u+=1)
            {
                const int &disparity = boundary[u];
                left_top_sub_view(u, disparity) = rgb8_colors::pink; // rgb8_colors::yellow;
            }
        }

        // draw the stixels -
        {
            draw_the_stixels(screen_left_view,
                             stixels_estimator.get_stixels());
        }
    } // end of draw left screen -

    // draw right screen ---
    {
        // since all matrices are computed to the left image,
        // it is more useful to see them compared to the left image
        //copy_and_convert_pixels(right_input_view, screen_right_view);
        copy_and_convert_pixels(left_input_view, screen_right_view);


        // objects cost on top --
        const StixelsEstimator::u_disparity_cost_t &objects_cost =
                stixels_estimator.get_object_u_disparity_cost();

        boost::gil::rgb8_view_t right_top_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          0, 0,
                                          objects_cost.cols(), objects_cost.rows());

        draw_matrix(objects_cost, right_top_sub_view);


        // ground cost on the bottom --
        const StixelsEstimator::u_disparity_cost_t &ground_cost =
                stixels_estimator.get_ground_u_disparity_cost();

        const int x_min = 0, y_min = std::max<int>(0, screen_left_view.height() - ground_cost.rows() );

        boost::gil::rgb8_view_t right_bottom_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          x_min, y_min,
                                          ground_cost.cols(), ground_cost.rows());

        draw_matrix(ground_cost, right_bottom_sub_view);
    }
    return;
}


void fill_cost_to_draw(const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost,
                       ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost_to_draw,
                       const int desired_height, const int stixel_width)
{

    typedef ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t cost_t;

    cost_t cost_without_max_value = cost;

    // we fix the lower left corner visualization problem
    {
        const float max_value = std::numeric_limits<float>::max();
        for(int row=0; row < cost.rows(); row +=1)
        {
            for(int column=0; column < cost.cols(); column +=1)
            {
                if(cost(row, column) == max_value)
                {
                    cost_without_max_value(row, column) = 0; // the lowest expected cost
                }
            }
        }

        const float true_max_value = cost_without_max_value.maxCoeff();
        for(int row=0; row < cost.rows(); row +=1)
        {
            for(int column=0; column < cost.cols(); column +=1)
            {
                if(cost(row, column) == max_value)
                {
                    cost_without_max_value(row, column) = true_max_value;
                }
            }
        }
    }

    const int pixels_per_row = desired_height / cost.cols();

    cost_to_draw = Eigen::MatrixXf::Zero(cost.cols()*pixels_per_row, cost.rows()*stixel_width);

    for(int row=0; row < cost_to_draw.rows(); row +=1)
    {
        for(int column=0; column < cost_to_draw.cols(); column +=1)
        {
            const int
                    row_step_index = row / pixels_per_row,
                    stixel_index = column / stixel_width;
            cost_to_draw(row, column) = cost_without_max_value(stixel_index, row_step_index);

        } // end of "for each row"
    } // end of "for each column"

    return;
}


void draw_stixels_estimation(const ImagePlaneStixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view)
{

    const bool should_draw_the_stixels = true;
    //const bool should_draw_the_stixels = false;

    const bool should_draw_cost_matrix = true;
    //const bool should_draw_cost_matrix = false;

    const int desired_cost_height = (left_input_view.height()*0.4);

    // draw left screen ---
    {
        copy_and_convert_pixels(left_input_view, screen_left_view);

        // draw the stixels --
        if(should_draw_the_stixels)
        {
            draw_the_stixels(screen_left_view,
                             stixels_estimator.get_stixels());
        }

        // draw bottom candidates --
        {
            const ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t &candidate_bottom = \
                    stixels_estimator.bottom_v_given_stixel_and_row_step;
            for(size_t stixel_index=0; stixel_index < candidate_bottom.shape()[0]; stixel_index +=1)
            {

                const boost::uint16_t column = stixel_index * stixels_estimator.get_stixel_width();
                for(size_t row_step_index=0; row_step_index < candidate_bottom.shape()[1]; row_step_index +=1)
                {
                    const boost::uint16_t row = candidate_bottom[stixel_index][row_step_index];
                    const boost::uint16_t line_width = 2;
                    for(int t_row = row; (t_row < (row + line_width)) and (t_row < screen_left_view.height()); t_row +=1)
                    {
                        screen_left_view(column, t_row) = rgb8_colors::white;
                    }

                } // end of "for each row step"

            } // end of "for each stixel"
        }


        // cost on top --
        if(should_draw_cost_matrix)
        {
            const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost =
                    stixels_estimator.get_cost_per_stixel_and_row_step();

            ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t cost_to_draw;

            fill_cost_to_draw(cost, cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);

            boost::gil::rgb8_view_t left_top_sub_view =
                    boost::gil::subimage_view(screen_left_view,
                                              0, 0,
                                              cost_to_draw.cols(), cost_to_draw.rows());

            draw_matrix(cost_to_draw, left_top_sub_view);


            // draw estimated boundary -
            {
                const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost =
                        stixels_estimator.get_cost_per_stixel_and_row_step();
                const int pixels_per_row = desired_cost_height / cost.cols();

                const std::vector<int> &boundary = stixels_estimator.get_stixel_and_row_step_ground_obstacle_boundary();

                for(std::size_t stixel_index=0; stixel_index < boundary.size(); stixel_index+=1)
                {
                    const int &row_step = boundary[stixel_index];
                    const int v =  pixels_per_row*row_step + (pixels_per_row/2);
                    const int begin_u = stixel_index*stixels_estimator.stixel_width,
                            end_u = std::min<int>((stixel_index+1)*stixels_estimator.stixel_width, left_top_sub_view.width());
                    for(int u=begin_u; u < end_u; u+=1)
                    {
                        left_top_sub_view(u, v) = rgb8_colors::pink;
                    }
                }
            }
        }

    } // end of draw left screen -

    // draw right screen ---
    {
        // since all matrices are computed to the left image,
        // it is more useful to see them compared to the left image
        //copy_and_convert_pixels(right_input_view, screen_right_view);
        copy_and_convert_pixels(left_input_view, screen_right_view);

        if(should_draw_cost_matrix)
        {

            // objects cost on top --
            const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &object_cost =
                    stixels_estimator.get_object_cost_per_stixel_and_row_step();

            ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t object_cost_to_draw;
            fill_cost_to_draw(object_cost, object_cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);

            boost::gil::rgb8_view_t right_top_sub_view =
                    boost::gil::subimage_view(screen_right_view,
                                              0, 0,
                                              object_cost_to_draw.cols(), object_cost_to_draw.rows());

            draw_matrix(object_cost_to_draw, right_top_sub_view);


            // ground cost on the bottom --
            const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ground_cost =
                    stixels_estimator.get_ground_cost_per_stixel_and_row_step();

            ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t ground_cost_to_draw;
            fill_cost_to_draw(ground_cost, ground_cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);


            const int x_min = 0, y_min = std::max<int>(0, screen_left_view.height() - ground_cost_to_draw.rows() );

            boost::gil::rgb8_view_t right_bottom_sub_view =
                    boost::gil::subimage_view(screen_right_view,
                                              x_min, y_min,
                                              ground_cost_to_draw.cols(), ground_cost_to_draw.rows());

            draw_matrix(ground_cost_to_draw, right_bottom_sub_view);

        } // end of "should draw cost matrix"

    } // end of draw right screen -

    return;
}



void draw_stixel_world(const stixels_t &the_stixels,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       const AbstractVideoInput::input_image_view_t &right_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view)
{

    // draw left input as background --
    copy_and_convert_pixels(left_input_view, screen_left_view);

    // draw the ground plane estimate ---
    //if(false)
    //{
    //    const GroundPlane &ground_plane = the_stixel_world_estimator_p->get_ground_plane();
    //    const MetricCamera &camera = video_input_p->get_metric_camera().get_left_camera();
    //    draw_the_ground_corridor(screen_left_view, camera, ground_plane);
    //}

    // draw the stixels ----
    {
        draw_the_stixels(screen_left_view, the_stixels);
    }

    // draw the right screen  ---
    {
        // copy right screen image ---
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }

    return;
}


void draw_stixel_world(const stixels_t &the_stixels,
                       const Eigen::MatrixXf &depth_map,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view)
{
    // draw left input as background --
    copy_and_convert_pixels(left_input_view, screen_left_view);

    // draw the ground plane estimate ---
    //if(false)
    //{
    //    const GroundPlane &ground_plane = the_stixel_world_estimator_p->get_ground_plane();
    //    const MetricCamera &camera = video_input_p->get_metric_camera().get_left_camera();
    //    draw_the_ground_corridor(screen_left_view, camera, ground_plane);
    //}

    // draw the stixels ----
    {
        draw_the_stixels(screen_left_view, the_stixels);
    }

    // draw the right screen  ---
    {
        // draw stixel_height_cost
        draw_matrix(depth_map, screen_right_view);
    }

    return;
}




void draw_stixel_match_lines( boost::gil::rgb8_view_t& left_screen_view,
                              boost::gil::rgb8_view_t& right_screen_view,
                              const stixels_t& left_screen_stixels,
                              const stixels_t& right_screen_stixels,
                              const std::vector< int >& stixel_matches )
{
    using rgb8_colors::jet_color_map;

    // Coefficients are in RBG order
    // The last entries are offset values
    const float  y_coeffs[ 4 ] = {  0.299,  0.587,   0.114,   0 };
    const float cb_coeffs[ 4 ] = { -0.169, -0.332,   0.500, 128 };
    const float cr_coeffs[ 4 ] = {  0.500, -0.419, -0.0813, 128 };

    // The coefficients are in YCbCr order
    // The last 3 elements in each array are the offset values in YCbCr order
    const double b_coeffs[ 6 ] = { 1.0,  1.7790,     0.0, 0.0,	-128.0,	    0.0 };
    const double g_coeffs[ 6 ] = { 1.0, -0.3455, -0.7169, 0.0,	-128.0,	 -128.0 };
    const double r_coeffs[ 6 ] = { 1.0,     0.0,  1.4075, 0.0,	   0.0,	 -128.0 };

    const size_t number_of_stixels = stixel_matches.size();

    const unsigned int stixel_sampling_width = 20; // Take the first of each 20
    const unsigned int color_stixel_sampling_width_ratio = 5;

    for(size_t i = 1; i < number_of_stixels; i+=1 )
    {
        if( (i % stixel_sampling_width) == 0 )
        {
            const int stixel_match = stixel_matches[ i ];

            if( stixel_match >= 0 )
            {
                const Stixel& right_stixel = right_screen_stixels[ i ];
                const Stixel& left_stixel = left_screen_stixels[ stixel_match ];

                if( right_stixel.type != Stixel::Occluded and left_stixel.type != Stixel::Occluded )
                {
                    const unsigned int color_index = ( i * color_stixel_sampling_width_ratio ) % number_of_stixels;

                    const boost::gil::rgb8c_pixel_t t_color = boost::gil::rgb8_view_t::value_type( jet_color_map[ color_index ][ 0 ],
                                                                                                   jet_color_map[ color_index ][ 1 ],
                                                                                                   jet_color_map[ color_index ][ 2 ] );

                    // Colorize stixels
                    const float color_cb = cb_coeffs[ 0 ] * t_color[ 0 ] +
                                           cb_coeffs[ 1 ] * t_color[ 1 ] +
                                           cb_coeffs[ 2 ] * t_color[ 2 ] +
                                           cb_coeffs[ 3 ];

                    const float color_cr = cr_coeffs[ 0 ] * t_color[ 0 ] +
                                           cr_coeffs[ 1 ] * t_color[ 1 ] +
                                           cr_coeffs[ 2 ] * t_color[ 2 ] +
                                           cr_coeffs[ 3 ];

                    // Colorize left stixel
                    for(int y = left_stixel.top_y; y < left_stixel.bottom_y; y+=1 )
                    {
                        const boost::gil::rgb8c_pixel_t t_pixel = left_screen_view( left_stixel.x, y );

                        const float color_y = y_coeffs[ 0 ] * t_pixel[ 0 ] +
                                              y_coeffs[ 1 ] * t_pixel[ 1 ] +
                                              y_coeffs[ 2 ] * t_pixel[ 2 ] +
                                              y_coeffs[ 3 ];

                        float color_r = r_coeffs[ 0 ] * ( color_y + r_coeffs[ 3 ] ) + r_coeffs[ 1 ] * ( color_cb + r_coeffs[ 4 ] ) + r_coeffs[ 2 ] * ( color_cr + r_coeffs[ 5 ] );
                        float color_g = g_coeffs[ 0 ] * ( color_y + g_coeffs[ 3 ] ) + g_coeffs[ 1 ] * ( color_cb + g_coeffs[ 4 ] ) + g_coeffs[ 2 ] * ( color_cr + g_coeffs[ 5 ] );
                        float color_b = b_coeffs[ 0 ] * ( color_y + b_coeffs[ 3 ] ) + b_coeffs[ 1 ] * ( color_cb + b_coeffs[ 4 ] ) + b_coeffs[ 2 ] * ( color_cr + b_coeffs[ 5 ] );

                        color_r = std::max< float >( 0, color_r ) ;
                        color_g = std::max< float >( 0, color_g ) ;
                        color_b = std::max< float >( 0, color_b ) ;

                        color_r = std::min< float >( 255, color_r ) ;
                        color_g = std::min< float >( 255, color_g ) ;
                        color_b = std::min< float >( 255, color_b ) ;

                        const int int_color_r = int( color_r + 0.5 );
                        const int int_color_g = int( color_g + 0.5 );
                        const int int_color_b = int( color_b + 0.5 );

                        const boost::gil::rgb8c_pixel_t t_new_color = boost::gil::rgb8_view_t::value_type( int_color_r, int_color_g, int_color_b );

                        left_screen_view( left_stixel.x, y ) = t_new_color;

                    }

                    // Colorize right stixel
                    for(int y = right_stixel.top_y; y < right_stixel.bottom_y; y+=1 )
                    {
                        const boost::gil::rgb8c_pixel_t t_pixel = right_screen_view( right_stixel.x, y );

                        const float color_y = y_coeffs[ 0 ] * t_pixel[ 0 ] +
                                              y_coeffs[ 1 ] * t_pixel[ 1 ] +
                                              y_coeffs[ 2 ] * t_pixel[ 2 ] +
                                              y_coeffs[ 3 ];

                        float color_r = r_coeffs[ 0 ] * ( color_y + r_coeffs[ 3 ] ) + r_coeffs[ 1 ] * ( color_cb + r_coeffs[ 4 ] ) + r_coeffs[ 2 ] * ( color_cr + r_coeffs[ 5 ] );
                        float color_g = g_coeffs[ 0 ] * ( color_y + g_coeffs[ 3 ] ) + g_coeffs[ 1 ] * ( color_cb + g_coeffs[ 4 ] ) + g_coeffs[ 2 ] * ( color_cr + g_coeffs[ 5 ] );
                        float color_b = b_coeffs[ 0 ] * ( color_y + b_coeffs[ 3 ] ) + b_coeffs[ 1 ] * ( color_cb + b_coeffs[ 4 ] ) + b_coeffs[ 2 ] * ( color_cr + b_coeffs[ 5 ] );

                        color_r = std::max< float >( 0, color_r ) ;
                        color_g = std::max< float >( 0, color_g ) ;
                        color_b = std::max< float >( 0, color_b ) ;

                        color_r = std::min< float >( 255, color_r ) ;
                        color_g = std::min< float >( 255, color_g ) ;
                        color_b = std::min< float >( 255, color_b ) ;

                        const int int_color_r = int( color_r + 0.5 );
                        const int int_color_g = int( color_g + 0.5 );
                        const int int_color_b = int( color_b + 0.5 );

                        const boost::gil::rgb8c_pixel_t t_new_color = boost::gil::rgb8_view_t::value_type( int_color_r, int_color_g, int_color_b );

                        right_screen_view( right_stixel.x, y ) = t_new_color;
                    }

                    // Draw top and bottom lines between stixels
                    int delta_top_y = right_stixel.top_y - left_stixel.top_y;
                    int delta_bottom_y = right_stixel.bottom_y - left_stixel.bottom_y;
                    int delta_x = left_screen_view.width() + right_stixel.x - left_stixel.x;

                    float slope_top_line = ( float( delta_top_y ) ) / delta_x;
                    float slope_bottom_line = ( float( delta_bottom_y ) ) / delta_x;

                    for( int x = left_stixel.x; x < left_screen_view.width(); ++x )
                    {
                        //                    left_screen_view( x, comon_bottom_y ) = t_color;
                        //                    left_screen_view( x, comon_top_y ) = t_color;

                        float y_top = slope_top_line * ( x - left_stixel.x ) + left_stixel.top_y;
                        float y_bottom = slope_bottom_line * ( x - left_stixel.x ) + left_stixel.bottom_y;

                        left_screen_view( x, int( y_top + 0.5 ) ) = t_color;
                        left_screen_view( x, int( y_bottom + 0.5 ) ) = t_color;
                    }

                    for( int x = right_stixel.x; x >= 0; --x )
                    {
                        //                    right_screen_view( x, comon_bottom_y ) = t_color;
                        //                    right_screen_view( x, comon_top_y ) = t_color;

                        float y_top = slope_top_line * ( x + left_screen_view.width() - left_stixel.x ) + left_stixel.top_y;
                        float y_bottom = slope_bottom_line * ( x + left_screen_view.width() - left_stixel.x ) + left_stixel.bottom_y;

                        right_screen_view( x, int( y_top + 0.5 ) ) = t_color;
                        right_screen_view( x, int( y_bottom + 0.5 ) ) = t_color;
                    }

                } // End of if( right_stixel is NOT Occluded and left_stixel is NOT Occluded )

            } // End of if( stixel_match >= 0 )
        }

    } // End of "for each stixel"

    return;
}





} // end of namespace doppia
