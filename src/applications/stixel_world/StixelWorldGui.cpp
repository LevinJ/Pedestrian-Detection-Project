#include "StixelWorldGui.hpp"

#include "drawing/gil/draw_the_ground_corridor.hpp"
#include "drawing/gil/draw_stixel_world.hpp"

#include "StixelWorldApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/ImagePlaneStixelsEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/FastStixelsEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/FastStixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/motion/DummyStixelMotionEstimator.hpp"

#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolumeFromDepthMap.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/for_each.hpp"
#include "helpers/xyz_indices.hpp"

#include <boost/format.hpp>

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/draw_matrix.hpp"
#include "drawing/gil/hsv_to_rgb.hpp"

#include <SDL/SDL.h>

#include <stdexcept>

//const float pi = 3.14159265358979323846264338327950288419716939937510;
const float pi = 3.14159265;

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelWorldGui");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelWorldGui");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelWorldGui");
}

} // end of anonymous namespace


namespace doppia
{

using boost::array;
using namespace boost::gil;

program_options::options_description StixelWorldGui::get_args_options(const bool add_base_gui_options)
{
    program_options::options_description desc("StixelWorldGui options");

    // specific options --
    desc.add_options()

            ("gui.colorize_disparity",
             program_options::value<bool>()->default_value(true),
             "colorize the disparity map or draw it using grayscale")

            ;

    if(add_base_gui_options)
    {
        // add base options --
        BaseSdlGui::add_args_options(desc);
    }

    return desc;
}

StixelWorldGui::StixelWorldGui(
        BaseApplication &application_,
        shared_ptr<AbstractVideoInput> video_input_p_,
        shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p_,
        shared_ptr<AbstractStixelMotionEstimator> stixel_motion_estimator_p_,
        const program_options::variables_map &options,
        const bool init_gui)
    :BaseSdlGui(application_, options),
      video_input_p(video_input_p_),
      stixel_world_estimator_p(stixel_world_estimator_p_),
      stixel_motion_estimator_p(stixel_motion_estimator_p_)
{

    // stixel_world_estimator_p may be empty (specially when used from ObjectsDetectionGui)
    // video_input_p may also be empty (when used from ObjectsDetectionGui, with process_folder option)
    if(init_gui and not video_input_p)
    {
        throw std::runtime_error("StixelWorldGui constructor expects that video_input is already initialized");
    }

    // retrieve program options --
    colorize_disparity = get_option_value<bool>(options, "gui.colorize_disparity");
    max_disparity = get_option_value<int>(options, "max_disparity");

    // create the application window --
    if(init_gui)
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = video_input_p->get_left_image();
        const int input_width = left_input_view.width();
        const int input_height = left_input_view.height();

        //const int window_width_and_height = max(input_width, input_height);
        //        BaseSdlGui::init_gui(base_application.get_application_title(),
        //                             window_width_and_height, window_width_and_height);

        BaseSdlGui::init_gui(base_application.get_application_title(),
                             input_width, input_height);

        // draw the first image --
        draw_video_input();
    }

    // populate the views map --
    views_map[SDLK_1] = view_t(boost::bind(&StixelWorldGui::draw_video_input, this), "draw_video_input");
    views_map[SDLK_2] = view_t(boost::bind(&StixelWorldGui::draw_ground_plane_estimation, this), "draw_ground_plane_estimation");
    views_map[SDLK_3] = view_t(boost::bind(&StixelWorldGui::draw_stixels_estimation, this), "draw_stixels_estimation");
    views_map[SDLK_4] = view_t(boost::bind(&StixelWorldGui::draw_stixels_height_estimation, this), "draw_stixels_height_estimation");
    views_map[SDLK_5] = view_t(boost::bind(&StixelWorldGui::draw_stixel_world, this), "draw_stixel_world");
    views_map[SDLK_6] = view_t(boost::bind(&StixelWorldGui::draw_disparity_map, this), "draw_disparity_map");
    //views_map[SDLK_3] = view_t(boost::bind(&StixelWorldGui::draw_features_tracks, this), "draw_features_tracks");
    //views_map[SDLK_4] = view_t(boost::bind(&StixelWorldGui::draw_optical_flow, this), "draw_optical_flow");
    views_map[SDLK_7] = view_t(boost::bind(&StixelWorldGui::draw_stixel_motion_tracks, this), "draw_stixel_motion_tracks");
    views_map[SDLK_8] = view_t(boost::bind(&StixelWorldGui::draw_stixel_motion_matrix, this), "draw_stixel_motion_matrix");
    views_map[SDLK_9] = view_t(boost::bind(&StixelWorldGui::draw_stixel_motion, this), "draw_stixel_motion" );


    // set the initial view mode --
    current_view = views_map[SDLK_1]; // video input
    //current_view = views_map[SDLK_2]; // ground plane estimation
    //current_view = views_map[SDLK_3]; // stixels estimation
    //current_view = views_map[SDLK_4]; // stixels height estimation
    if(stixel_world_estimator_p)
    {
        current_view = views_map[SDLK_5]; // stixel world
    }


    if (stixel_motion_estimator_p)
    {
        //    current_view = views_map[SDLK_8]; // stixel motion matrix
        current_view = views_map[SDLK_9]; // color coded stixel motion
    }

    return;
}


StixelWorldGui::~StixelWorldGui()
{
    // nothing to do here
    return;
}

bool StixelWorldGui::process_inputs()
{
    const bool end_of_game = BaseSdlGui::process_inputs();

    Uint8 *keys = SDL_GetKeyState(NULL);

    if( stixel_motion_estimator_p and keys[SDLK_r] )
    {
        stixel_motion_estimator_p->reset_stixel_tracks_image();
    }

    return end_of_game;
}


void StixelWorldGui::draw_video_input()
{
    const AbstractVideoInput::input_image_view_t &left_input_view = video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = video_input_p->get_right_image();

    copy_and_convert_pixels(left_input_view, screen_left_view);
    copy_and_convert_pixels(right_input_view, screen_right_view);

    return;
}



void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self)
{
    draw_ground_plane_estimator(ground_plane_estimator,
                                video_input.get_right_image(),
                                video_input.get_stereo_calibration(),
                                self.screen_right_view);
    return;
}


void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self)
{
    draw_ground_plane_estimator(ground_plane_estimator,
                                video_input.get_right_image(),
                                video_input.get_stereo_calibration(),
                                self.screen_right_view);
    return;
}


void draw_stixels_estimation(const StixelsEstimator &stixels_estimator,
                             AbstractVideoInput &video_input,
                             BaseSdlGui &self)
{
    draw_stixels_estimation(stixels_estimator,
                            video_input.get_left_image(),
                            self.screen_left_view, self.screen_right_view );
    return;
}

void draw_stixels_estimation(const ImagePlaneStixelsEstimator &stixels_estimator,
                             AbstractVideoInput &video_input,
                             BaseSdlGui &self)
{
    draw_stixels_estimation(stixels_estimator,
                            video_input.get_left_image(),
                            self.screen_left_view, self.screen_right_view );
    return;
}

void draw_stixel_motion_tracks( const AbstractStixelMotionEstimator& motion_estimator,
                                AbstractVideoInput &video_input,
                                BaseSdlGui &self )
{
    std::cout << "draw_stixel_motion();" << std::endl;

    //const AbstractVideoInput::input_image_view_t &left_input_view = video_input.get_left_image();
    //const AbstractVideoInput::input_image_view_t &right_input_view = motion_estimator.get_previous_image_view();

    //const AbstractVideoInput::input_image_view_t &previous_input_view = motion_estimator.get_previous_image_view();
    const AbstractVideoInput::input_image_view_t &current_input_view = motion_estimator.get_current_image_view();

    //const AbstractStixelMotionEstimator::motion_cost_matrix_t& motion_cost_matrix = motion_estimator.get_motion_cost_matrix();
    //const AbstractStixelMotionEstimator::motion_cost_matrix_t& visual_motion_cost_matrix = motion_estimator.get_visual_motion_cost_matrix();

    //const Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 >& visual_motion_cost_column_maxima_indices = motion_estimator.get_visual_motion_cost_column_maxima_indices();
    //const Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >& motion_cost_column_maxima_indices = motion_estimator.get_motion_cost_column_maxima_indices();

    //const AbstractStixelMotionEstimator::motion_cost_matrix_t& matching_cost_matrix = motion_estimator.get_matching_cost_matrix();

    const stixels_t current_stixels = motion_estimator.get_current_stixels();
    //const stixels_t previous_stixels = motion_estimator.get_previous_stixels();

    //const std::vector< int > stixels_motion = motion_estimator.get_stixels_motion();

    const AbstractStixelMotionEstimator::input_image_view_t stixel_track_image_view = motion_estimator.get_stixel_tracks_image_view();


    // Draw on the left screen the current frame with current stixels
    {
        copy_and_convert_pixels( current_input_view, self.screen_left_view );
        draw_the_stixels( self.screen_left_view, current_stixels );
    }


    // Draw on the right screen the stixel tracks with current stixels
    {
        boost::gil::copy_pixels( stixel_track_image_view, self.screen_right_view );

        // Draw the stixels on the topmost layer
        draw_the_stixels( self.screen_right_view, current_stixels );
    }

    return;
}

void draw_stixel_motion_matrix( const AbstractStixelMotionEstimator& motion_estimator,
                                AbstractVideoInput &video_input,
                                BaseSdlGui &self )
{
    const AbstractVideoInput::input_image_view_t& current_input_view = motion_estimator.get_current_image_view();
    const AbstractVideoInput::input_image_view_t& previous_input_view = motion_estimator.get_previous_image_view();

    const stixels_t current_stixels = motion_estimator.get_current_stixels();
    const stixels_t previous_stixels = motion_estimator.get_previous_stixels();

    //    const AbstractStixelMotionEstimator::motion_cost_matrix_t& matching_cost_matrix = motion_estimator.get_matching_cost_matrix();

    const AbstractStixelMotionEstimator::motion_cost_matrix_t& motion_cost_matrix = motion_estimator.get_motion_cost_matrix();
    const AbstractStixelMotionEstimator::dynamic_matrix_boolean_t& motion_cost_assignment_matrix = motion_estimator.get_motion_cost_assignment_matrix();
    //const AbstractStixelMotionEstimator::motion_cost_matrix_t& dp_cost_matrix = motion_estimator.get_dynamic_programming_cost_matrix();

    //const AbstractStixelMotionEstimator::motion_cost_matrix_t& visual_motion_cost_matrix = motion_estimator.get_visual_motion_cost_matrix();

    //const Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 >& visual_motion_cost_column_maxima_indices = motion_estimator.get_visual_motion_cost_column_maxima_indices();
    const Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >& motion_cost_column_maxima_indices = motion_estimator.get_motion_cost_column_maxima_indices();

    const Eigen::MatrixXi& stixels_one_to_many_assignment_map = motion_estimator.get_stixels_one_to_many_assignment_map();

    const AbstractStixelMotionEstimator::stixels_motion_t& stixels_motion = motion_estimator.get_stixels_motion();

    AbstractStixelMotionEstimator::motion_cost_matrix_t motion_cost_matrix_copy = Eigen::MatrixXf( motion_cost_matrix.rows(), motion_cost_matrix.cols() );

    for( int i = 0, number_of_rows = motion_cost_matrix_copy.rows(); i < number_of_rows; ++i )
    {
        for( int j = 0, number_of_cols = motion_cost_matrix_copy.cols(); j < number_of_cols; ++j )
        {
            if( motion_cost_assignment_matrix( i, j ) == true )
            {
                motion_cost_matrix_copy( i, j ) = motion_cost_matrix( i, j );
            }
            else
            {
                motion_cost_matrix_copy( i, j ) = 0.0f;
            }

        } // End of for(j)
    } // End of for(i)

    const float precision = 0.01;

    for( int iteration = 0; iteration < 10; ++iteration )
    {
        float maximum_motion_cost = motion_cost_matrix_copy.maxCoeff();

        for( int i = 0, number_of_rows = motion_cost_matrix_copy.rows(); i < number_of_rows; ++i )
        {
            for( int j = 0, number_of_cols = motion_cost_matrix_copy.cols(); j < number_of_cols; ++j )
            {
                if( fabs( maximum_motion_cost - motion_cost_matrix_copy( i, j ) ) < precision )
                {
                    motion_cost_matrix_copy( i, j ) = 0.0f;
                }

            } // End of for(j)
        } // End of for(i)

    } // End of for(iteration)

    /*
     * Draw the left screen
     *
     * Draw the previous frame with previous stixels
     */

    {
        // At the very first frame, there is no previous frame
        if( previous_input_view.dimensions() == current_input_view.dimensions() )
        {
            copy_and_convert_pixels( previous_input_view, self.screen_left_view );
            draw_the_stixels( self.screen_left_view, previous_stixels );
        }
        else
        {
            copy_and_convert_pixels( current_input_view, self.screen_left_view );
            draw_the_stixels( self.screen_left_view, current_stixels );
        }

    }

    /*
     * Draw the right screen
     *
     * Draw the current frame with current stixels and motion cost matrix (on top)
     */

    {
        copy_and_convert_pixels( current_input_view, self.screen_right_view );

        boost::gil::rgb8_view_t right_top_sub_view1 = boost::gil::subimage_view( self.screen_right_view,
                                                                                 0, 0,
                                                                                 motion_cost_matrix_copy.cols(), motion_cost_matrix_copy.rows() );

        draw_matrix( motion_cost_matrix_copy, right_top_sub_view1 );

        ///////////////

        if( current_stixels.size() != 0 and motion_cost_column_maxima_indices.rows() != 0 )
        {
            assert(motion_cost_column_maxima_indices.rows() == current_stixels.size());
            assert(stixels_one_to_many_assignment_map.cols() == current_stixels.size());

            for( unsigned int s = 0, number_of_stixels = current_stixels.size(); s < number_of_stixels; ++s )
            {
                const Stixel& current_stixel = current_stixels[ s ];

                boost::gil::rgb8_pixel_t drawing_color;

                if( current_stixel.type == Stixel::Occluded )
                {
                    drawing_color = rgb8_colors::red;
                }
                else if( stixels_motion[ s ] == -1 )
                {
                    drawing_color = rgb8_colors::green;
                }
                else
                {
                    drawing_color = rgb8_colors::pink;
                }

                for( int y = motion_cost_column_maxima_indices( s ) - stixels_one_to_many_assignment_map( motion_cost_column_maxima_indices( s ), s ),
                     upper_limit_y = motion_cost_column_maxima_indices( s ) + stixels_one_to_many_assignment_map( motion_cost_column_maxima_indices( s ), s );
                     y <= upper_limit_y;
                     ++y )
                {
                    if( y >= 0 and y < right_top_sub_view1.height() )
                    {
                        for( int x = current_stixel.x - current_stixel.width / 2, upper_limit_x = current_stixel.x + current_stixel.width / 2; x <= upper_limit_x; ++x )
                        {
                            if( x >= 0 and x < right_top_sub_view1.width() )
                            {
                                right_top_sub_view1( x, y ) = drawing_color;
                            }

                        } // End of for( x )
                    }

                } // End of for( y )

            } // End of for( s )

            draw_the_stixels( self.screen_right_view, current_stixels );
        }
    }

    return;
}


void draw_color_wheel(const boost::gil::rgb8c_view_t &input_view, boost::gil::rgb8_view_t &view)
{
    assert(input_view.dimensions() == view.dimensions());

    const float
            alpha = 0.9,
            saturation = 0.80,
            value = 0.99,
            center_u = view.width() / 2.0f,
            center_v = view.height() / 2.0f;

    const int
            border = 10,
            width = view.width(),
            height = view.height();

    for( int v = 0; v < height; ++v )
    {
        for( int u = 0; u < width; ++u )
        {
            if( (u < border)
                or (u > width - border)
                or (v < border)
                or (v > height - border) )
            {
                const float
                        delta_u = u - center_u,
                        delta_v = v - center_v,
                        hue = 360 * ( 0.5 + ( atan2( delta_v, delta_u ) / ( 2 * pi ) ) );

                float R, G, B;
                hsv_to_rgb( hue, saturation, value, R, G, B );

                const rgb8c_pixel_t
                        original_color = input_view( u, v ),
                        t_color = rgb8c_pixel_t( alpha * R + ( 1 - alpha ) * original_color[ 0 ],
                                                 alpha * G + ( 1 - alpha ) * original_color[ 1 ],
                                                 alpha * B + ( 1 - alpha ) * original_color[ 2 ] );
                view( u, v ) = t_color;
            }

        } // end of "for each column"
    } // end of "for each row"


    return;
}


void draw_stixel_motion( const AbstractStixelMotionEstimator& motion_estimator,
                         AbstractVideoInput& /*video_input*/,
                         BaseSdlGui& self)
{
    using namespace boost::gil;

    const AbstractVideoInput::input_image_view_t& current_input_view = motion_estimator.get_current_image_view();

    typedef color_converted_view_type< AbstractVideoInput::input_image_view_t, rgba8_pixel_t >::type input_image_view_rgba_t;
    input_image_view_rgba_t current_input_view_rgba = color_converted_view< rgb8_pixel_t >( current_input_view );

    if( current_input_view_rgba.dimensions() != current_input_view.dimensions() )
    {
        throw std::runtime_error( "draw_stixel_motion(...) Image view dimensions must agree" );
    }

    const stixels_t current_stixels = motion_estimator.get_current_stixels();
    const stixels_t previous_stixels = motion_estimator.get_previous_stixels();

    // Draw the left screen
    // Draw the current frame with current stixels
    {
        copy_and_convert_pixels( current_input_view, self.screen_left_view );
        draw_the_stixels( self.screen_left_view, current_stixels );
    }


    // Draw the right screen
    // Draw the current frame with color coded current stixels motion
    {
        copy_and_convert_pixels( current_input_view, self.screen_right_view );
        draw_the_stixels( self.screen_right_view, current_stixels );

        const bool should_draw_color_wheel = false;
        if(should_draw_color_wheel)
        {
            draw_color_wheel(current_input_view, self.screen_right_view);
        }


        // Draw the stixels --

        float alpha = 0.75;

        const float saturation = 0.80;
        float value = 0.99;

        for( size_t stixel_index = 0; stixel_index < current_stixels.size(); ++stixel_index)
        {
            const Stixel& current_stixel = current_stixels[ stixel_index ];

            if( current_stixel.valid_backward_delta_x == false )
            {
                continue; // we simply skip this stixel
            }


            const int
                    u = current_stixel.x,
                    v = current_stixel.bottom_y;

            int previous_u = u + current_stixel.backward_delta_x;
            int previous_v = v;

            int min_delta_u = std::numeric_limits< int >::max();

            for( size_t previous_stixel_index = 0;
                 previous_stixel_index < previous_stixels.size();
                 ++previous_stixel_index )
            {
                const Stixel& previous_stixel = previous_stixels[ previous_stixel_index ];

                int delta_u = abs(previous_stixel.x - previous_u);

                if( delta_u < min_delta_u )
                {
                    min_delta_u = delta_u;
                    previous_v = previous_stixel.bottom_y;
                    if ( delta_u == 0 )
                    {
                        break;
                    }
                }
            } // End of for( s_prev )

            float R, G, B;

            int delta_u = u - previous_u;

            const bool draw_angle = false;

            if(draw_angle)
            {
                int delta_v = ( v - previous_v ) / 10;

                const float hue = 360 * ( 0.5 + ( atan2( delta_v, delta_u ) / ( 2 * pi ) ) );

                hsv_to_rgb( hue, saturation, value, R, G, B );
                alpha = 0.75;
            }
            else
            {
                // FIXME hardcoded parameter (is read from option, just too lazy to retrieve the value)
                const int maximum_possible_motion_in_pixels = 66;

                //value = std::min< float >( 1, sqrt( delta_u * delta_u + delta_v * delta_v ) / 10.f );
                //const float hue = 360 * ( 0.5 + ( atan2( delta_v, delta_u ) / ( 2 * pi ) ) );
                //const float hue = 360 * ( float( delta_u ) / (2 * 66) + 0.5 );

                //const float saturated_delta_u = std::max( std::min(delta_u, 30), -30);

                const float delta_u_ratio = float( delta_u ) / maximum_possible_motion_in_pixels;

                //const float mapped_delta_u_ratio = ( 2 / ( 1 + std::exp( -4 * delta_u_ratio ) ) - 1 ); // between 0 and 1.
                const float mapped_delta_u_ratio = ( 2 / ( 1 + std::exp( -6 * delta_u_ratio ) ) - 1 ); // between 0 and 1.
                const float mapped_delta_u = maximum_possible_motion_in_pixels * mapped_delta_u_ratio;

                //std::cout << "delta_u --> delta_u_ratio : " << delta_u << " -- " << delta_u_ratio << std::endl;

                const float hue = 360 * ( float( mapped_delta_u ) / (2 * maximum_possible_motion_in_pixels) + 0.5 );
                //const float hue = 180 * ( float( mapped_delta_u ) / (2 * maximum_possible_motion_in_pixels) + 0.5 ) + 180;
                //const float hue = 360 * ( float( saturated_delta_u ) / (2 * 30) + 0.5 );

                //const float value = 0.7 + 0.3 * std::abs( mapped_delta_u_ratio );
                //const float saturation = 0.8 + 0.2 * std::abs( mapped_delta_u_ratio );

                hsv_to_rgb( hue, saturation, value, R, G, B );

                const float mapped_delta_u_ratio_bis = ( 2 / ( 1 + std::exp( -15 * delta_u_ratio ) ) - 1 ); // between 0 and 1.
                alpha = std::min(0.85f, std::max(0.2f, std::abs(mapped_delta_u_ratio_bis)));
            }


            for( int stixel_v = current_stixel.top_y; stixel_v < current_stixel.bottom_y; ++stixel_v )
            {
                const rgb8c_pixel_t
                        original_color = current_input_view( u, stixel_v ),
                        t_color = rgb8c_pixel_t( alpha * R + ( 1 - alpha ) * original_color[ 0 ],
                                                 alpha * G + ( 1 - alpha ) * original_color[ 1 ],
                                                 alpha * B + ( 1 - alpha ) * original_color[ 2 ] );

                self.screen_right_view( u, stixel_v ) = t_color;

            } // end of "for the stixel height"


        } // end of "for each stixel"

    } // end of "drawing right screen view"

    return;
} // end of draw_stixel_motion(...)


void StixelWorldGui::draw_ground_plane_estimation()
{
    // draw the ground plane and the stixels

    if(stixel_world_estimator_p and video_input_p)
    {

        AbstractStixelWorldEstimator &stixel_world_estimator = *(stixel_world_estimator_p);
        BaseGroundPlaneEstimator *base_ground_plane_estimator_p = NULL;

        StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(&stixel_world_estimator);
        FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(&stixel_world_estimator);

        if(the_stixel_world_estimator_p != NULL)
        {
            base_ground_plane_estimator_p = the_stixel_world_estimator_p->ground_plane_estimator_p.get();
        }
        else if(the_fast_stixel_world_estimator_p != NULL)
        {
            base_ground_plane_estimator_p = the_fast_stixel_world_estimator_p->ground_plane_estimator_p.get();
        }

        // Left screen --
        {
            // copy left screen image ---
            const AbstractVideoInput::input_image_view_t &left_input_view = video_input_p->get_left_image();
            copy_and_convert_pixels(left_input_view, screen_left_view);

            // add the ground bottom and top corridor ---
            const GroundPlane &ground_plane = stixel_world_estimator.get_ground_plane();
            const MetricCamera &camera = video_input_p->get_metric_camera().get_left_camera();
            draw_the_ground_corridor(screen_left_view, camera, ground_plane);

            if(false)
            {
                // draw the stixels ---
                draw_the_stixels(screen_left_view,
                                 stixel_world_estimator.get_stixels());
            }

            // this variable is useful for icp_tracker.py
            const bool debug_ground_plane_projections = false;
            if(debug_ground_plane_projections)
            {
                const stixels_t &stixels = stixel_world_estimator.get_stixels();
                const Stixel &middle_stixel = stixels[stixels.size() / 2];

                const MetricStereoCamera &stereo_camera = video_input_p->get_metric_camera();
                const float depth = stereo_camera.disparity_to_depth(middle_stixel.disparity);

                Eigen::Vector2f point_uv(2);
                point_uv[0] = middle_stixel.x;
                point_uv[1] = middle_stixel.bottom_y;

                Eigen::Vector3f point_xyz = camera.back_project_2d_point_to_3d(point_uv, depth);
                Eigen::Vector3f ground_point_xyz = ground_plane.projection(point_xyz);

                Eigen::Vector2f ground_point_uv = camera.project_3d_point(ground_point_xyz);

                for(int c=0; c < 10; c+=1)
                {
                    screen_left_view(ground_point_uv[0]+c, ground_point_uv[1]) = rgb8_colors::magenta;
                }
            }

            // add our prior on the ground area ---
            const bool draw_prior_on_ground_area = true;
            if(draw_prior_on_ground_area and base_ground_plane_estimator_p != NULL)
            {
                const std::vector<int> &ground_object_boundary_prior =
                        base_ground_plane_estimator_p->get_ground_area_prior();
                const std::vector<int> &boundary = ground_object_boundary_prior;
                for(std::size_t u=0; u < boundary.size(); u+=1)
                {
                    const int &disparity = boundary[u];
                    screen_left_view(u, disparity) = rgb8_colors::cyan;
                }
            }

        } // end of left screen -

        // Right screen --
        {
            // will draw on the right screen
            if(the_stixel_world_estimator_p != NULL)
            {
                draw_ground_plane_estimator(*(the_stixel_world_estimator_p->ground_plane_estimator_p),
                                            *video_input_p, *this);
            }
            else  if(the_fast_stixel_world_estimator_p != NULL)
            {
                draw_ground_plane_estimator(*(the_fast_stixel_world_estimator_p->ground_plane_estimator_p),
                                            *video_input_p, *this);
            }
        }

    } // end of "the_stixel_world_estimator_p != NULL"
    else
    {
        // we simply freeze the screen
    }

    return;
} // end of StixelWorldGui::draw_ground_plane_estimation


void StixelWorldGui::draw_stixels_estimation()
{
    // draw the ground plane and the stixels

    StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p.get());

    if(the_stixel_world_estimator_p != NULL)
    {
        doppia::draw_stixels_estimation(*(the_stixel_world_estimator_p->stixels_estimator_p),
                                        *video_input_p, *this);
    }
    else if(the_fast_stixel_world_estimator_p != NULL)
    {
        const FastStixelsEstimator *fast_estimator_p = \
                dynamic_cast<FastStixelsEstimator *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());

        const ImagePlaneStixelsEstimator *uv_estimator_p = \
                dynamic_cast<ImagePlaneStixelsEstimator *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());

        if(fast_estimator_p)
        {
            doppia::draw_stixels_estimation(*fast_estimator_p, *video_input_p, *this);
        }
        else if(uv_estimator_p)
        {
            doppia::draw_stixels_estimation(*uv_estimator_p, *video_input_p, *this);
        }
        else
        {
            // simply freeze the screen
        }
    }
    else
    { // the_stixel_world_estimator_p == NULL
        // simply freeze the screen
    }

    return;
} // end of StixelWorldGui::draw_stixels_estimation


void draw_stixels_height_estimation(const stixels_t &the_stixels,
                                    const Eigen::MatrixXf &height_cost,
                                    const std::vector<int> &u_v_ground_obstacle_boundary,
                                    AbstractVideoInput &video_input,
                                    BaseSdlGui &self)
{
    const AbstractVideoInput::input_image_view_t &left_input_view = video_input.get_left_image();
    //const AbstractVideoInput::input_image_view_t &right_input_view = video_input.get_right_image();

    // draw left input as background --
    copy_and_convert_pixels(left_input_view, self.screen_left_view);

    // draw the stixels ----
    draw_the_stixels(self.screen_left_view, the_stixels);


    // draw stixel_height_cost
    {
        Eigen::MatrixXf normalized_height_cost = height_cost;
        vertical_normalization(normalized_height_cost,
                               u_v_ground_obstacle_boundary);

        draw_matrix(normalized_height_cost, self.screen_right_view);
    }

    // draw the stixels on top
    //draw_the_stixels(screen_right_view,
    //                 the_stixel_world_estimator_p->get_stixels());


    return;

}

void StixelWorldGui::draw_stixels_height_estimation()
{
    // draw the ground plane and the stixels

    StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p.get());
    //const AbstractVideoInput::input_image_view_t &left_input_view = video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = video_input_p->get_right_image();

    bool drawed_height_estimation = false;

    if(the_stixel_world_estimator_p != NULL)
    {

        StixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< StixelsEstimatorWithHeightEstimation *>(the_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p != NULL)
        {
            doppia::draw_stixels_height_estimation(the_stixel_world_estimator_p->get_stixels(),
                                                   the_stixels_estimator_p->get_stixel_height_cost(),
                                                   the_stixels_estimator_p->get_u_v_ground_obstacle_boundary(),
                                                   *video_input_p, *this);
            drawed_height_estimation = true;
        }


    }
    else if(the_fast_stixel_world_estimator_p != NULL)
    {
        FastStixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< FastStixelsEstimatorWithHeightEstimation *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p != NULL)
        {
            doppia::draw_stixels_height_estimation(the_fast_stixel_world_estimator_p->get_stixels(),
                                                   the_stixels_estimator_p->get_stixel_height_cost(),
                                                   the_stixels_estimator_p->get_u_v_ground_obstacle_boundary(),
                                                   *video_input_p, *this);
            drawed_height_estimation = true;
        }

    }

    if(drawed_height_estimation == false)
    {
        // we simply copy the right input to the right screen

        // copy right screen image ---
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }



    return;
} // end of StixelWorldGui::draw_stixels_height_estimation


void draw_stixel_world(const stixels_t &the_stixels,
                       AbstractVideoInput &video_input,
                       BaseSdlGui &self)
{
    draw_stixel_world(the_stixels,
                      video_input.get_left_image(), video_input.get_right_image(),
                      self.screen_left_view, self.screen_right_view);
    return;
}


void draw_stixel_world(const stixels_t &the_stixels,
                       const Eigen::MatrixXf &depth_map,
                       AbstractVideoInput &video_input,
                       BaseSdlGui &self)
{
    draw_stixel_world(the_stixels, depth_map, video_input.get_left_image(),
                      self.screen_left_view, self.screen_right_view);
    return;
}



void StixelWorldGui::draw_stixel_world()
{
    // draw the ground plane and the stixels

    StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p.get());
    //const AbstractVideoInput::input_image_view_t &left_input_view = video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = video_input_p->get_right_image();

    if(the_stixel_world_estimator_p != NULL)
    {

        StixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< StixelsEstimatorWithHeightEstimation *>(the_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p)
        {
            doppia::draw_stixel_world(the_stixel_world_estimator_p->get_stixels(),
                                      the_stixels_estimator_p->get_depth_map(),
                                      *video_input_p, *this);

        }
        else
        {
            doppia::draw_stixel_world(the_stixel_world_estimator_p->get_stixels(),
                                      *video_input_p, *this);
        }

    }
    else if(the_fast_stixel_world_estimator_p != NULL)
    {
        FastStixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< FastStixelsEstimatorWithHeightEstimation *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p != NULL)
        {
            doppia::draw_stixel_world(the_fast_stixel_world_estimator_p->get_stixels(),
                                      the_stixels_estimator_p->get_disparity_likelihood_map(),
                                      *video_input_p, *this);
        }
        else
        {
            doppia::draw_stixel_world(the_fast_stixel_world_estimator_p->get_stixels(),
                                      *video_input_p, *this);
        }
    }
    else
    {
        // we simply freeze the left screen
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }

    return;
} // end of StixelWorldGui::draw_stixel_world


void StixelWorldGui::draw_stixel_motion_tracks()
{
    if( stixel_motion_estimator_p )
    {
        doppia::draw_stixel_motion_tracks( *stixel_motion_estimator_p, *video_input_p, *this );
    }

    return;

}  // end of StixelWorldGui::draw_stixel_motion

void StixelWorldGui::draw_stixel_motion_matrix()
{
    if( stixel_motion_estimator_p )
    {
        doppia::draw_stixel_motion_matrix( *stixel_motion_estimator_p, *video_input_p, *this );
    }

    return;
}

void StixelWorldGui::draw_stixel_motion()
{
    if( stixel_motion_estimator_p )
    {
        doppia::draw_stixel_motion( *stixel_motion_estimator_p, *video_input_p, *this );
    }

    return;
}

// FIXME should move this into a draw_disparity_map
class pixel_disparity_colorizer
{

    float max_disparity;
public:

    pixel_disparity_colorizer(const int max_disparity_);

    template<typename PixelType>
    void operator()(PixelType &value) const;

};

pixel_disparity_colorizer::pixel_disparity_colorizer(const int max_disparity_)
    : max_disparity(max_disparity_)
{
    // nothing to do here
    return;
}

template<typename PixelType>
void pixel_disparity_colorizer::operator()(PixelType &pixel_value) const
{
    using namespace boost::gil;
    const int max_possible_channel_value = channel_traits< typename channel_type< PixelType>::type >::max_value();

    const float d = pixel_value[0];


    // non valid disparities are set to max_possible_channel_value
    const bool is_valid_disparity = (d >= 0) and (d < max_possible_channel_value);

    float saturation, value, hue;

    // first draw unmatched pixels in grey
    if (is_valid_disparity)
    {
        //const float normalized_disparity = 2.0* (d / max_disparity); // why 2.0* ?
        const float normalized_disparity = (d/ max_disparity);
        //normalized_disparity *= normalized_disparity * normalized_disparity;

        // 360 is red, 240 is blue, 180 is cyan, 120 is green, 60 yellow, 0 is red
        //hue = 240*normalized_disparity + 120;
        //hue = (180 - 120*normalized_disparity);
        //hue = 240*(1-normalized_disparity); // red is far, green is near
        //hue = 360 - 240*(1-normalized_disparity); // red is far, blue is near
        //hue = 240*(1-normalized_disparity) + 120; // blue near, green far


        hue = 130*(1-normalized_disparity) + 10;

        saturation = 0.6f;
        //value = 0.9f;
        //value = 0.9f*(0.5f*normalized_disparity + 0.5f);
        value = 0.4f*normalized_disparity + 0.6f;
    }
    else
    {
        hue = 0;
        saturation = 0.0f;
        value = 0.0f;
    }

    // HSV to RGB
    {
        float
                H = hue,
                S = saturation,
                V = value,
                R = 0, G = 0, B = 0;
        if (H==0 and S==0)
        {
            R = G = B = V;
        }
        else
        {
            H/=60;
            const int i = (int) std::floor(H);
            const float
                    f = (i&1)?(H-i):(1-H+i),
                    m = V*(1-S),
                    n = V*(1-S*f);
            switch (i)
            {
            case 6 :
            case 0 :
                R = V;
                G = n;
                B = m;
                break;
            case 1 :
                R = n;
                G = V;
                B = m;
                break;
            case 2 :
                R = m;
                G = V;
                B = n;
                break;
            case 3 :
                R = m;
                G = n;
                B = V;
                break;
            case 4 :
                R = n;
                G = m;
                B = V;
                break;
            case 5 :
                R = V;
                G = m;
                B = n;
                break;
            }
        }
        R*=255;
        G*=255;
        B*=255;
        get_color(pixel_value, red_t())   = (R<0?0:(R>255?255:R));
        get_color(pixel_value, green_t()) = (G<0?0:(G>255?255:G));
        get_color(pixel_value, blue_t())  = (B<0?0:(B>255?255:B));
    }

    return;
}

void StixelWorldGui::colorize_disparity_map(const gil::rgb8_planar_view_t &view)
{
    using namespace boost::gil;

    const bool use_data_max_disparity = false;
    if(use_data_max_disparity)
    {
        const int max_possible_channel_value =
                channel_traits< channel_type< gil::rgb8_planar_view_t::value_type >::type >::max_value();

        int max_value = 0;

        gil::rgb8_planar_view_t::iterator pixels_it;
        for(pixels_it = view.begin(); pixels_it != view.end(); ++pixels_it )
        {
            if((*pixels_it)[0] > max_value and (*pixels_it)[0] != max_possible_channel_value)
            {
                max_value = (*pixels_it)[0] ;
            }
        }

        const pixel_disparity_colorizer colorizer(max_value);
        gil::for_each_pixel(view, colorizer);
    }
    else
    {
        const pixel_disparity_colorizer colorizer(max_disparity);
        gil::for_each_pixel(view, colorizer);
    }

    return;
}


void StixelWorldGui::draw_disparity_map()
{
    using namespace boost::gil;
    typedef DisparityCostVolumeFromDepthMap::disparity_map_view_t disparity_map_view_t;

    const AbstractVideoInput::input_image_view_t &left_input_view =
            video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view =
            video_input_p->get_right_image();

    {
        // draw left input as background --
        copy_and_convert_pixels(left_input_view, screen_left_view);

        if(stixel_world_estimator_p)
        {
            // draw the stixels
            draw_the_stixels(screen_left_view,
                             stixel_world_estimator_p->get_stixels());
        }
    }

    // retrieve the disparity map ---
    disparity_map_view_t disparity_map;
    StixelWorldEstimator *the_stixel_world_estimator_p =
            dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    if(the_stixel_world_estimator_p != NULL)
    {

        DisparityCostVolumeFromDepthMap *cost_volume_from_depth_map_p =
                dynamic_cast< DisparityCostVolumeFromDepthMap *>(the_stixel_world_estimator_p->cost_volume_estimator_p.get());
        if(cost_volume_from_depth_map_p != NULL)
        {
            disparity_map = cost_volume_from_depth_map_p->get_disparity_map();
        }
    }

    //if(not disparity_map.empty())
    if(disparity_map.size() > 0)
    {
        // equalize (using opencv)
        using namespace cv;
        gil::opencv::ipl_image_wrapper ipl_disparity =
                gil::opencv::create_ipl_image(disparity_map);

        Mat mat_disparity(ipl_disparity.get()), mat_equalized_disparity;
        const bool equalize_depth_map = true;

        if(equalize_depth_map)
        {
            cv::equalizeHist(mat_disparity, mat_equalized_disparity);
        }
        else
        {
            mat_equalized_disparity = mat_disparity;
            //cv::normalize(mat_disparity, mat_equalized_disparity, 0, cv::NORM_MINMAX);
        }

        const disparity_map_view_t equalized_disparity =
                gil::interleaved_view(mat_equalized_disparity.cols, mat_equalized_disparity.rows,
                                      reinterpret_cast<disparity_map_view_t::value_type *>(mat_equalized_disparity.data),
                                      static_cast<size_t>(mat_equalized_disparity.step));

        // draw on screen
        copy_and_convert_pixels(equalized_disparity, screen_right_view);
    }
    else
    {
        // no disparity map available
        // we simply draw the inputs

        // copy right screen image ---
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }

    return;
}



void StixelWorldGui::save_screenshot()
{

    BaseSdlGui::save_screenshot(); // first save the traditional screenshot

    const bool save_rectified_images = false;
    const bool save_colored_stixel_motion_images = true;

    if( save_rectified_images or save_colored_stixel_motion_images )
    {
        const boost::filesystem::path &recording_path = base_application.get_recording_path();

        const int frame_number = base_application.get_current_frame_number();

        if(save_rectified_images)
        {
            const  boost::filesystem::path
                    left_screenshot_filename = recording_path / boost::str(boost::format("image_%08i_0.png") % frame_number),
                    right_screenshot_filename = recording_path / boost::str(boost::format("image_%08i_1.png") % frame_number);

            // record the screenshot --
            boost::gil::png_write_view( left_screenshot_filename.string(), video_input_p->get_left_image() );
            boost::gil::png_write_view( right_screenshot_filename.string(), video_input_p->get_right_image() );

            if(save_all_screenshots == false)
            {
                printf("Recorded image %s\n", left_screenshot_filename.string().c_str());
            }
        }

        if( save_colored_stixel_motion_images )
        {
            const  boost::filesystem::path
                    screenshot_filename = recording_path / boost::str(boost::format("image_%08i_0.png") % frame_number);

            // record the screenshot --
            //            boost::gil::png_write_view( screenshot_filename.string(), /*video_input_p->get_left_image()*/ );
        }
    }

    return;
}


} // end of namespace doppia



