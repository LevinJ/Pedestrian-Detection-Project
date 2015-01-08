#include "DummyStixelMotionEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/xyz_indices.hpp"

#include "drawing/gil/colors.hpp"

#include <iostream>

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>

namespace doppia {

// FIXME where should I put this ?
/// min_float_disparity will be used to replace integer disparity == 0
const float min_float_disparity = 0.8f;

boost::program_options::options_description DummyStixelMotionEstimator::get_args_options()
{

    boost::program_options::options_description desc("DummyStixelMotionEstimator options");

    desc.add_options()

            ("stixel_world.motion.maximum_displacement_between_frames",
             boost::program_options::value<float>()->default_value( 0.30 ),
             "maximum lateral displacement we expect to see between two frames. "
             "This value is in meters and will define the search range for the stixels motion")

            ("stixel_world.motion.average_pedestrian_speed",
             boost::program_options::value<float>()->default_value( 1.5 ),
             "average pedestrian speed. "
             "This value is in meters per second and will help the computation of the search range for the stixels motion")

            ("stixel_world.motion.maximum_pedestrian_speed",
             boost::program_options::value<float>()->default_value( 2.5 ),
             "maximum pedestrian speed. "
             "This value is in meters per second and will help the computation of the search range for the stixels motion")

            ("stixel_world.motion.maximum_possible_motion_in_pixels",
             boost::program_options::value< int >()->default_value( 66 ), // Default value is derived from Bahnhofstrasse sequence
             "maximum motion in pixels over the whole sequence. "
             "This value is in pixels and the motion cost matrix will be allocated accordingly.")

            ("stixel_world.motion.maximum_number_of_one_to_many_stixels_matching",
             boost::program_options::value< int >()->default_value( 2 ),
             "maximum number of stixels assigned to the same stixel motion in pixels over the whole sequence. "
             "If this value is n, then one stixel might be assigned to 2n+1 stixels at most.")
            /*
                                                                    ("stixel_world.ground_cost_weight",
                                                                     boost::program_options::value<float>()->default_value(1.0),
                                                                     "relative weight between the ground cost and the objects cost")


                                                                    ("stixel_world.ground_cost_threshold",
                                                                     boost::program_options::value<float>()->default_value(-1),
                                                                     "threshold the ground cost based on a percent of the highest value. "
                                                                     "ground_cost_threshold == 0.3 indicates that only values higher than 0.3*highest_ground_cost_value will be allowed. "
                                                                     "ground_cost_threshold <= 0 indicates that no thresholding should be applied")
                                                        */
            ;

    return desc;
}

DummyStixelMotionEstimator::DummyStixelMotionEstimator(const boost::program_options::variables_map &options,
                                                       const MetricStereoCamera &camera, int stixels_width )
    : stereo_camera( camera ),
      average_pedestrian_speed( get_option_value< float >( options, "stixel_world.motion.average_pedestrian_speed" ) ),
      maximum_pedestrian_speed( get_option_value< float >( options, "stixel_world.motion.maximum_pedestrian_speed" ) ),
      video_frame_rate( get_option_value< int >( options, "video_input.frame_rate" ) ),
      frame_width( get_option_value< int >( options, "video_input.frame_width" ) ),
      frame_height( get_option_value< int >( options, "video_input.frame_height" ) ),
      minimum_object_height_in_pixels( get_option_value< int >( options, "stixel_world.minimum_object_height_in_pixels" ) ),
      stixel_representation_height( minimum_object_height_in_pixels ),
      maximum_possible_motion_in_pixels( get_option_value< int >( options, "stixel_world.motion.maximum_possible_motion_in_pixels" ) ),
      number_of_stixels_per_frame( frame_width / stixels_width ),
      maximum_number_of_one_to_many_stixels_matching( get_option_value< int >( options, "stixel_world.motion.maximum_number_of_one_to_many_stixels_matching" ) )
{
    maximum_displacement_between_frames = maximum_pedestrian_speed / video_frame_rate;

    maximum_pixel_value = 255;
    maximum_depth_value = stereo_camera.disparity_to_depth( min_float_disparity );

    initialize_matrices();

    reset_stixel_tracks_image();

    return;
}


DummyStixelMotionEstimator::~DummyStixelMotionEstimator()
{
    // nothing to do here
    return;
}

void DummyStixelMotionEstimator::initialize_matrices()
{
    motion_cost_matrix = Eigen::MatrixXf::Zero( 2 * maximum_possible_motion_in_pixels + 2, number_of_stixels_per_frame ); // Matrix is initialized with 0.

    pixelwise_sad_matrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.
    real_height_differences_matrix = Eigen::MatrixXf::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() ); // Matrix is initialized with 0.

    motion_cost_assignment_matrix = dynamic_matrix_boolean_t::Constant( motion_cost_matrix.rows(), motion_cost_matrix.cols(), false ); // Matrix is initialized with 'false'.

    real_motion_vectors_matrix.resize( boost::extents[ motion_cost_matrix.cols() ][ motion_cost_matrix.rows() ] ); // Elements are accessed as [stixel_index][motion_index]

    current_stixel_depths = Eigen::VectorXf::Zero( number_of_stixels_per_frame );
    current_stixel_real_heights = Eigen::VectorXf::Zero( number_of_stixels_per_frame );

    stixels_one_to_many_assignment_map = Eigen::MatrixXi::Zero( motion_cost_matrix.rows(), motion_cost_matrix.cols() );

    return;
}


void DummyStixelMotionEstimator::compute()
{
    compute_motion_cost_matrix();
    compute_motion();
    update_stixel_tracks_image();

    return;
}

void DummyStixelMotionEstimator::compute_motion_v1()
{
    compute_motion_dp_v1();

    return;
}

void DummyStixelMotionEstimator::compute_matching_cost_matrix()
{
    //const float maximum_depth_difference = 1.0f; // in meters
    //const float maximum_real_motion = maximum_pedestrian_speed / video_frame_rate;

    const size_t number_of_current_stixels = current_stixels_p->size();
    const size_t number_of_previous_stixels = previous_stixels_p->size();

    matching_cost_matrix = Eigen::MatrixXf::Zero( number_of_current_stixels, number_of_previous_stixels ); // Matrix is initialized with 0.
    pixelwise_sad_matrix = Eigen::MatrixXf::Zero( number_of_current_stixels, number_of_previous_stixels ); // Matrix is initialized with 0.
    real_height_differences_matrix = Eigen::MatrixXf::Zero( number_of_current_stixels, number_of_previous_stixels ); // Matrix is initialized with 0.
    pixelwise_height_differences_matrix = Eigen::MatrixXf::Zero( number_of_current_stixels, number_of_previous_stixels ); // Matrix is initialized with 0.

    Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic  > matching_cost_assignment_matrix =
            Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic  >::Constant( number_of_current_stixels, number_of_previous_stixels, false ); // Matrix is initialized with 'false'.

    // Fill in the matching cost matrix
    for( size_t s_current = 0; s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];

        const int stixel_horizontal_padding = compute_stixel_horizontal_padding( current_stixel );

        if( current_stixel.x - ( current_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
                current_stixel.x + ( current_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() )
        {
            const float current_stixel_real_height = compute_stixel_real_height( current_stixel );
            const int current_stixel_pixelwise_height = abs( current_stixel.top_y - current_stixel.bottom_y );

            const unsigned int maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( current_stixel );

            for( size_t s_prev = 0; s_prev < number_of_previous_stixels; ++s_prev )
            {
                const Stixel& previous_stixel = ( *previous_stixels_p )[ s_prev ];

                if( previous_stixel.x - ( previous_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
                        previous_stixel.x + ( previous_stixel.width - 1 ) / 2 + stixel_horizontal_padding < previous_image_view.width() )
                {
                    const int pixelwise_motion = previous_stixel.x - current_stixel.x; // Motion can be positive or negative

                    //                    if( pixelwise_motion >= -( int( maximum_motion_in_pixels ) ) and
                    //                        pixelwise_motion <= int( maximum_motion_in_pixels ) )
                    if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) and
                            pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ) )
                    {
                        const float previous_stixel_real_height = compute_stixel_real_height( previous_stixel );
                        const int previous_stixel_pixelwise_height = abs( previous_stixel.top_y - previous_stixel.bottom_y );

                        const float pixelwise_sad = compute_pixelwise_sad( current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding );
                        const float real_height_difference =
                                fabs( current_stixel_real_height - previous_stixel_real_height ) / ( current_stixel_real_height + previous_stixel_real_height );
                        const float pixelwise_height_difference =
                                fabs( float( current_stixel_pixelwise_height ) - float( previous_stixel_pixelwise_height ) ) / ( current_stixel_pixelwise_height + previous_stixel_pixelwise_height );


                        pixelwise_sad_matrix( s_current, s_prev ) = pixelwise_sad;
                        real_height_differences_matrix( s_current, s_prev ) = real_height_difference;
                        pixelwise_height_differences_matrix( s_current, s_prev ) = pixelwise_height_difference;

                        matching_cost_assignment_matrix( s_current, s_prev ) = true;
                    }

                } // End of if( previous_stixel stays inside image borders )

            } // End of for( s_prev )

        } // End of if( current_stixel stays inside image borders )

    } // End of for( s_current )

    /// Rescale the real height difference matrix elemants so that it will have the same range with pixelwise_sad
    //const float maximum_real_height_difference = real_height_differences_matrix.maxCoeff();
    //    real_height_differences_matrix = real_height_differences_matrix * ( float ( maximum_pixel_value ) / maximum_real_height_difference );
    real_height_differences_matrix = real_height_differences_matrix * maximum_pixel_value;

    pixelwise_height_differences_matrix = pixelwise_height_differences_matrix * maximum_pixel_value;

    /// Fill in the motion cost matrix
    matching_cost_matrix = pixelwise_sad_matrix + real_height_differences_matrix;

    //const size_t number_of_matrix_elements = matching_cost_matrix.rows() * matching_cost_matrix.cols();

    const float maximum_cost_matrix_element = matching_cost_matrix.maxCoeff(); // Minimum is 0 by definition

    //    insertion_cost_dp = ( matching_cost_matrix * ( ( float( 1.0 ) ) / number_of_matrix_elements ) ).sum();
    insertion_cost_dp = maximum_cost_matrix_element * 0.25;
    deletion_cost_dp = insertion_cost_dp;

    for( motion_cost_matrix_t::Index i = 0, number_of_rows = matching_cost_matrix.rows(); i < number_of_rows; ++i )
    {
        for( motion_cost_matrix_t::Index j = 0, number_of_cols = matching_cost_matrix.cols(); j < number_of_cols; ++j )
        {
            if( matching_cost_assignment_matrix( i, j ) == false )
            {
                matching_cost_matrix( i, j ) = 1.1 * maximum_cost_matrix_element;
                matching_cost_assignment_matrix( i, j ) = true;
            }
        }
    }

    return;
}

void DummyStixelMotionEstimator::compute_motion_cost_matrix()
{
    //const float maximum_depth_difference = 1.0f;

    const float maximum_allowed_real_height_difference = 0.5f;
    const float alpha = 0.3;

    //const float maximum_real_motion = maximum_pedestrian_speed / video_frame_rate;

    const unsigned int number_of_current_stixels = current_stixels_p->size();
    const unsigned int number_of_previous_stixels = previous_stixels_p->size();

    motion_cost_matrix.fill( 0.f );
    pixelwise_sad_matrix.fill( 0.f );
    real_height_differences_matrix.fill( 0.f );
    motion_cost_assignment_matrix.fill( false );

    current_stixel_depths.fill( 0.f );
    current_stixel_real_heights.fill( 0.f );

    // Fill in the motion cost matrix
    for( unsigned int s_current = 0; s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];

        const unsigned int stixel_horizontal_padding = compute_stixel_horizontal_padding( current_stixel );

        /// Do NOT add else conditions since it can affect the computation of matrices
        if( current_stixel.x - ( current_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
                current_stixel.x + ( current_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() /*and
                                current_stixel.type != Stixel::Occluded*/ ) // Horizontal padding for current stixel is suitable
        {
            const float current_stixel_disparity = std::max< float >( min_float_disparity, current_stixel.disparity );
            const float current_stixel_depth = stereo_camera.disparity_to_depth( current_stixel_disparity );

            const float current_stixel_real_height = compute_stixel_real_height( current_stixel );

            // Store for future reference
            current_stixel_depths( s_current ) = current_stixel_depth;
            current_stixel_real_heights( s_current ) = current_stixel_real_height;

            for( unsigned int s_prev = 0; s_prev < number_of_previous_stixels; ++s_prev )
            {
                const Stixel& previous_stixel = ( *previous_stixels_p )[ s_prev ];

                if( previous_stixel.x - ( previous_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
                        previous_stixel.x + ( previous_stixel.width - 1 ) / 2 + stixel_horizontal_padding < previous_image_view.width() /*and
                                        previous_stixel.type != Stixel::Occluded*/ ) // Horizontal padding for previous stixel is suitable
                {
                    const float previous_stixel_disparity = std::max< float >( min_float_disparity, previous_stixel.disparity );
                    const float previous_stixel_depth = stereo_camera.disparity_to_depth( previous_stixel_disparity );

                    Eigen::Vector3f real_motion = compute_real_motion_between_stixels( current_stixel, previous_stixel, current_stixel_depth, previous_stixel_depth );
                    //const float real_motion_magnitude = real_motion.norm();

                    //                    if( fabs( current_stixel_depth - previous_stixel_depth ) < maximum_depth_difference )
                    {
                        const int pixelwise_motion = previous_stixel.x - current_stixel.x; // Motion can be positive or negative

                        const unsigned int maximum_motion_in_pixels_for_current_stixel = compute_maximum_pixelwise_motion_for_stixel( current_stixel );

                        if( pixelwise_motion >= -( int( maximum_motion_in_pixels_for_current_stixel ) ) and
                                pixelwise_motion <= int( maximum_motion_in_pixels_for_current_stixel ) /*and
                                                real_motion_magnitude <= maximum_real_motion*/ )
                        {
                            float pixelwise_sad;
                            float real_height_difference;

                            if( current_stixel.type != Stixel::Occluded and previous_stixel.type != Stixel::Occluded )
                            {
                                pixelwise_sad = compute_pixelwise_sad( current_stixel, previous_stixel, current_image_view, previous_image_view, stixel_horizontal_padding );
                                real_height_difference = fabs( current_stixel_real_height - compute_stixel_real_height( previous_stixel ) );
                            }
                            else
                            {
                                pixelwise_sad = maximum_pixel_value;
                                real_height_difference = maximum_allowed_real_height_difference;
                            }

                            pixelwise_sad_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = pixelwise_sad;
                            real_height_differences_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) =
                                    std::min( 1.0f, real_height_difference / maximum_allowed_real_height_difference );

                            motion_cost_assignment_matrix( pixelwise_motion + maximum_possible_motion_in_pixels, s_current ) = true;
                            real_motion_vectors_matrix[ s_current ][ pixelwise_motion + maximum_possible_motion_in_pixels ] = real_motion;
                        }
                    }
                }

            } // End of for( s_prev )
        }

    } // End of for( s_current )

    /// Rescale the real height difference matrix elemants so that it will have the same range with pixelwise_sad
    //const float maximum_real_height_difference = real_height_differences_matrix.maxCoeff();
    //    real_height_differences_matrix = real_height_differences_matrix * ( float ( maximum_pixel_value ) / maximum_real_height_difference );
    real_height_differences_matrix = real_height_differences_matrix * maximum_pixel_value;

    /// Fill in the motion cost matrix
    motion_cost_matrix = alpha * pixelwise_sad_matrix + ( 1 - alpha ) * real_height_differences_matrix; // [0, 255]

    const float maximum_cost_matrix_element = motion_cost_matrix.maxCoeff(); // Minimum is 0 by definition

    /// Fill in disappearing stixel entries specially
    //    insertion_cost_dp = maximum_cost_matrix_element * 0.75;
    insertion_cost_dp = maximum_pixel_value * 0.6;
    deletion_cost_dp = insertion_cost_dp; // insertion_cost_dp is not used for the moment !!

    for( motion_cost_matrix_t::Index
         j = 0,
         number_of_cols = motion_cost_matrix.cols(),
         largest_row_index = motion_cost_matrix.rows() - 1;
         j < number_of_cols;
         ++j )
    {
        motion_cost_matrix( largest_row_index, j ) = deletion_cost_dp;
        motion_cost_assignment_matrix( largest_row_index, j ) = true;

    } // End of for(j)

    for( motion_cost_matrix_t::Index i = 0, number_of_rows = motion_cost_matrix.rows(); i < number_of_rows; ++i )
    {
        for( motion_cost_matrix_t::Index j = 0, number_of_cols = motion_cost_matrix.cols(); j < number_of_cols; ++j )
        {
            if( motion_cost_assignment_matrix( i, j ) == false )
            {
                motion_cost_matrix( i, j ) = 1.2 * maximum_cost_matrix_element;
                // motion_cost_assignment_matrix(i,j) should NOT be set to true for the entries which are "forced".
            }
        }
    }

    /**
      *
      * Lines below are intended for DEBUG & VISUALIZATION purposes
      *
      **/

    //    fill_in_visualization_motion_cost_matrix();

    return;
}

void DummyStixelMotionEstimator::fill_in_visualization_motion_cost_matrix()
{
    const unsigned int visual_cost_matrix_height = 100;

    visual_motion_cost_matrix = Eigen::MatrixXf::Zero( visual_cost_matrix_height, current_stixels_p->size() ); // Matrix is initialized with 0.

    float motion_cost_matrix_rescaling_factor = ( float( visual_cost_matrix_height ) ) / motion_cost_matrix.rows();

    for( unsigned int i = 0; i < visual_cost_matrix_height; ++i )
    {
        float scaled_i = ( float( i ) ) / motion_cost_matrix_rescaling_factor;

        float scaled_float_upper_i = std::ceil( scaled_i );
        float scaled_float_lower_i = std::floor( scaled_i );

        float coefficient_upper_i = scaled_i - scaled_float_lower_i;
        float coefficient_lower_i = scaled_float_upper_i - scaled_i;

        unsigned int scaled_upper_i = ( unsigned int )( scaled_float_upper_i );
        unsigned int scaled_lower_i = ( unsigned int )( scaled_float_lower_i );

        if( scaled_upper_i == scaled_lower_i )
        {
            coefficient_upper_i = 0.5;
            coefficient_lower_i = 0.5;
        }

        for( unsigned int j = 0, number_of_stixels = current_stixels_p->size(); j < number_of_stixels; ++j )
        {
            visual_motion_cost_matrix( i, j ) = coefficient_upper_i * motion_cost_matrix( scaled_upper_i, j ) +
                    coefficient_lower_i * motion_cost_matrix( scaled_lower_i, j );
        }

    } // End of for( i )

    return;
}

void DummyStixelMotionEstimator::compute_motion()
{
    compute_motion_dp();

    indices_columnwise_minimum_visual_cost =
            Eigen::Matrix< unsigned int, Eigen::Dynamic, 2 >( motion_cost_matrix.cols(), 2 ); // Vector is uninitialized

    // Convert the motion solution to visual format
    {
        const float matrix_scaling_factor = ( float )( visual_motion_cost_matrix.rows() ) / ( float )( motion_cost_matrix.rows() );

        for( unsigned int i = 0, number_of_cols = motion_cost_matrix.cols(); i < number_of_cols; ++i  )
        {
            const float scaled_index = matrix_scaling_factor * indices_columnwise_minimum_cost( i );

            const unsigned int scaled_lower_index = ( unsigned int )( std::floor( scaled_index ) );
            unsigned int scaled_upper_index = ( unsigned int )( std::ceil( scaled_index ) );

            if( scaled_lower_index == scaled_upper_index )
            {
                scaled_upper_index = scaled_upper_index + 1;
            }

            indices_columnwise_minimum_visual_cost( i, 0 ) = scaled_lower_index;
            indices_columnwise_minimum_visual_cost( i, 1 ) = scaled_upper_index;
        }
    }

    return;
}

/// see section III.C of Kubota et al. 2007 paper
void DummyStixelMotionEstimator::compute_motion_dp()
{
    // Copy all m_i(d_i) values
    M_cost_dp = motion_cost_matrix;

    neighbour_differences = Eigen::MatrixXf::Zero( ( *current_stixels_p ).size() - 1, 1 );
    c_cost_matrix.resize(
                boost::extents[motion_cost_matrix.cols() - 1][motion_cost_matrix.rows()][motion_cost_matrix.rows()] );

    // right to left pass
    dp_first_pass();
    //    dp_first_pass_one_to_many(); // buggy version

    // left to right pass
    dp_second_pass();

    return;
}

void DummyStixelMotionEstimator::compute_motion_dp_v1()
{
    const unsigned int number_of_matching_cost_matrix_rows = matching_cost_matrix.rows();
    const unsigned int number_of_matching_cost_matrix_cols = matching_cost_matrix.cols();

    M_cost_dp = Eigen::MatrixXf::Zero( number_of_matching_cost_matrix_rows + 1, number_of_matching_cost_matrix_cols + 1 );

    Eigen::MatrixXi M_cost_source_indicator_matrix = Eigen::MatrixXi::Zero( number_of_matching_cost_matrix_rows, number_of_matching_cost_matrix_cols );

    dp_first_pass_v1( M_cost_source_indicator_matrix );
    dp_second_pass_v1( M_cost_source_indicator_matrix );

    return;
}

void DummyStixelMotionEstimator::dp_first_pass_v1( Eigen::MatrixXi& M_cost_source_indicator_matrix )
{
    const unsigned int number_of_rows = M_cost_dp.rows();
    const unsigned int number_of_cols = M_cost_dp.cols();

    M_cost_source_indicator_matrix = Eigen::MatrixXi::Zero( number_of_rows, number_of_cols );

    for( unsigned int i = 0; i < number_of_rows; ++i )
    {
        M_cost_dp( i, 0 ) = i * deletion_cost_dp;
        M_cost_source_indicator_matrix( i, 0 ) = 2;
    }

    for( unsigned int j = 0; j < number_of_cols; ++j )
    {
        M_cost_dp( 0, j ) = j * insertion_cost_dp;
        M_cost_source_indicator_matrix( 0, j ) = 3;
    }

    for( unsigned int i = 1; i < number_of_rows; ++i )
    {
        unsigned int s_current = i - 1; // Index for the current stixel

        for( unsigned int j = 1; j < number_of_cols; ++j )
        {
            unsigned int s_prev = j - 1; // Index for the previous stixel

            const float cost_matching_option = M_cost_dp( i - 1, j - 1 ) + matching_cost_matrix( s_current, s_prev );
            const float cost_deletion_option = M_cost_dp( i - 1, j ) + deletion_cost_dp;
            const float cost_insertion_option = M_cost_dp( i, j - 1 ) + insertion_cost_dp;

            if( cost_matching_option <= cost_insertion_option and cost_matching_option <= cost_deletion_option ) // cost_matching_option is the minimum
            {
                M_cost_dp( i, j ) = cost_matching_option;
                M_cost_source_indicator_matrix( i, j ) = 1;
            }
            else if( cost_deletion_option <= cost_insertion_option ) // cost_deletion_option is the minimum
            {
                M_cost_dp( i, j ) = cost_deletion_option;
                M_cost_source_indicator_matrix( i, j ) = 2;
            }
            else // cost_insertion_option is the minimum
            {
                M_cost_dp( i, j ) = cost_insertion_option;
                M_cost_source_indicator_matrix( i, j ) = 3;
            }

        } // End of for( j )

    } // End of for( i )

    return;
}

void DummyStixelMotionEstimator::dp_second_pass_v1( const Eigen::MatrixXi& M_cost_source_indicator_matrix )
{
    unsigned int i = M_cost_dp.rows() - 1;
    unsigned int j = M_cost_dp.cols() - 1;

    while( i != 0 and j != 0 )
    {
        unsigned int s_current = i - 1;
        unsigned int s_prev = j - 1;

        switch( M_cost_source_indicator_matrix( i, j ) )
        {
        case ( 1 ) : // Current stixel is matched to previous stixel

            stixels_motion[ s_current ] = s_prev;

            i = i - 1;
            j = j - 1;

            break;

        case ( 2 ) : // Current stixel can not be matched to any previous stixel

            stixels_motion[ s_current ] = -1;
            i = i - 1;
            break;

        case ( 3 ) : // Previous stixel can not be matched to any current stixel

            j = j - 1;
            break;

        default :

            throw std::runtime_error( "DummyStixelMotionEstimator::dp_second_pass()_v1 -- Wrong input in M_cost_source_indicator_matrix !" );
            break;
        }
    }

    return;
}


void DummyStixelMotionEstimator::dp_first_pass()
{
    // FIXME : all these parameters should be program options
    const float
            delta_h = 1, // expected change in height in meters
            delta_z = 3, // expected change in depth in meters
            alpha_z = 0.1,
            alpha_h = 0.5,
            //k_z = 0.5 * maximum_pixel_value * 2,
            k_z = 0.5 * maximum_pixel_value * 2,
            k_h = 0,
            k_abs_diff = 500; // controls overall smoothing

    //    const float lambda_s = 1; // never used

    const int number_of_cols = motion_cost_matrix.cols();
    const int number_of_motions = motion_cost_matrix.rows() - 1; // The number of motions in range
    const int number_of_total_motions = number_of_motions + 1; // The last row is for "disappearing" motion - special case

    const int number_of_current_stixels = current_stixels_p->size();

    Eigen::VectorXf current_neighbour_stixel_real_height_differences = Eigen::VectorXf::Zero( number_of_current_stixels - 1 );
    Eigen::VectorXf current_neighbour_stixel_depth_differences = Eigen::VectorXf::Zero( number_of_current_stixels - 1 );

    for( unsigned int s_cur = 0, number_of_neighbours = number_of_current_stixels - 1; s_cur < number_of_neighbours; ++s_cur )
    {
        current_neighbour_stixel_depth_differences( s_cur ) = fabs( current_stixel_depths( s_cur ) - current_stixel_depths( s_cur + 1 ) );
        current_neighbour_stixel_real_height_differences( s_cur ) = fabs( current_stixel_real_heights( s_cur ) - current_stixel_real_heights( s_cur + 1 ) );
    }

    //    const float maximum_neighbour_stixel_depth_difference = current_neighbour_stixel_depth_differences.maxCoeff();
    //    const float maximum_neighbour_stixel_real_height_difference = current_neighbour_stixel_real_height_differences.maxCoeff();

    // Rescale the neighbour information vectors
    //    current_neighbour_stixel_depth_differences = current_neighbour_stixel_depth_differences * ( float( maximum_pixel_value ) ) / maximum_neighbour_stixel_depth_difference;
    //    current_neighbour_stixel_real_height_differences = current_neighbour_stixel_real_height_differences * ( float( maximum_pixel_value ) ) / maximum_neighbour_stixel_real_height_difference;

    // col and next_col should not be unsigned int (causes segmentation fault)
    for( int col = number_of_cols - 2; col >=0;  --col )
    {
        int next_col = col + 1;

        //        const Stixel& current_stixel = ( *current_stixels_p )[ col ];
        const Stixel& next_stixel = ( *current_stixels_p )[ next_col ];

        //        const unsigned int stixel_horizontal_padding = compute_stixel_horizontal_padding( current_stixel );

        //        const bool stixels_safe = ( current_stixel.x - ( current_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
        //                                    current_stixel.x + ( current_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() and
        //                                    next_stixel.x - ( next_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
        //                                    next_stixel.x + ( next_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() /*and
        //                                    current_stixel.type != Stixel::Occluded and
        //                                    next_stixel.type != Stixel::Occluded*/ );

        float neighbour_difference =
                k_z * std::max( alpha_z, 1 - current_neighbour_stixel_depth_differences( col ) / delta_z ) +
                k_h * std::max( alpha_h, 1 - current_neighbour_stixel_real_height_differences( col ) / delta_h );

        neighbour_differences( col ) = neighbour_difference;

        for( int m = 0; m < number_of_total_motions; ++m )
        {
            float min_M_plus_c = std::numeric_limits< float >::max();

            for( int e = 0; e < number_of_total_motions; ++e )
            {
                // ( m - max_motion_in_pixels ) - ( e - max_motion_in_pixels )
                float motion_abs_difference = 0.f;
                //float real_motion_vectors_difference = 0.f;

                float c_cost = 0;

                if( (m < number_of_motions) and (e < number_of_motions) )
                {
                    if( (motion_cost_assignment_matrix( m, col ) == true )
                            and (motion_cost_assignment_matrix( e, next_col ) == true) )
                    {
                        motion_abs_difference = float( abs( m - e ) ) / ( number_of_motions - 1 );

                        //                        real_motion_vectors_difference =
                        //                                ( ( real_motion_vectors_matrix[ col ][ m ] - real_motion_vectors_matrix[ next_col ][ e ] ).norm() ) / maximum_displacement_between_frames;
                    }
                    else
                    {
                        motion_abs_difference = 1.0f;
                        //real_motion_vectors_difference = 1.0f;
                    }

                }
                else if( m == number_of_motions and e == number_of_motions )
                {
                    motion_abs_difference = 0;
                    //real_motion_vectors_difference = 0;
                }
                else
                {
                    motion_abs_difference = 1.0f;
                    //real_motion_vectors_difference = 1.0f;
                }


                //                c_cost = real_motion_vectors_difference * neighbour_difference;
                c_cost = motion_abs_difference * (k_abs_diff +  neighbour_difference);
                //                c_cost = real_motion_vectors_difference;
                //                c_cost = 0;

                if( next_stixel.type == Stixel::Occluded )
                {
                    c_cost = 0;
                }



                // Store for future use
                c_cost_matrix[ col ][ m ][ e ] = c_cost;

                const float t_cost = M_cost_dp( e, next_col ) + c_cost;

                min_M_plus_c = std::min( t_cost, min_M_plus_c );

            } // End of for( e )

            M_cost_dp( m, col ) += min_M_plus_c;
            stixels_one_to_many_assignment_map( m, col ) = 0; // Just to be in accordance with the one-to-many matching formulation

        } // End of for( m )


    } // End of for( col )

    return;
}

void DummyStixelMotionEstimator::dp_first_pass_one_to_many()
{
    /// FIXME : Should be carried outside !
    const float delta_h = 1; // in meters
    const float delta_z = 3; // in meters

    const float alpha_z = 0.1;
    const float alpha_h = 0.5;

    const float k_z = 0.5 * maximum_pixel_value * 2;
    const float k_h = 0;

    //const float lambda_s = 1.0f;

    typedef motion_cost_matrix_t::Index index_t;
    const index_t number_of_cols = motion_cost_matrix.cols();
    const int number_of_motions = motion_cost_matrix.rows() - 1; // The number of motions in range
    const int number_of_total_motions = number_of_motions + 1; // The last row is for "disappearing" motion - special case

    const unsigned int number_of_current_stixels = current_stixels_p->size();

    Eigen::VectorXf current_neighbour_stixel_real_height_differences = Eigen::VectorXf::Zero( number_of_current_stixels - 1 );
    Eigen::VectorXf current_neighbour_stixel_depth_differences = Eigen::VectorXf::Zero( number_of_current_stixels - 1 );

    for( unsigned int s_cur = 0, number_of_neighbours = number_of_current_stixels - 1; s_cur < number_of_neighbours; ++s_cur )
    {
        current_neighbour_stixel_depth_differences( s_cur ) = fabs( current_stixel_depths( s_cur ) - current_stixel_depths( s_cur + 1 ) );
        current_neighbour_stixel_real_height_differences( s_cur ) = fabs( current_stixel_real_heights( s_cur ) - current_stixel_real_heights( s_cur + 1 ) );
    }

    //    const float maximum_neighbour_stixel_depth_difference = current_neighbour_stixel_depth_differences.maxCoeff();
    //    const float maximum_neighbour_stixel_real_height_difference = current_neighbour_stixel_real_height_differences.maxCoeff();

    // Rescale the neighbour information vectors
    //    current_neighbour_stixel_depth_differences = current_neighbour_stixel_depth_differences * ( float( maximum_pixel_value ) ) / maximum_neighbour_stixel_depth_difference;
    //    current_neighbour_stixel_real_height_differences = current_neighbour_stixel_real_height_differences * ( float( maximum_pixel_value ) ) / maximum_neighbour_stixel_real_height_difference;

    // col and next_col should not be unsigned int (causes segmentation fault)
    for( int col = number_of_cols - 2; col >=0;  --col )
    {
        int next_col = col + 1;

        //const Stixel& current_stixel = ( *current_stixels_p )[ col ];
        const Stixel& next_stixel = ( *current_stixels_p )[ next_col ];

        //const int stixel_horizontal_padding = compute_stixel_horizontal_padding( current_stixel );

        //        const bool stixels_safe = ( current_stixel.x - ( current_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
        //                                    current_stixel.x + ( current_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width() and
        //                                    next_stixel.x - ( next_stixel.width - 1 ) / 2 - stixel_horizontal_padding >= 0 and
        //                                    next_stixel.x + ( next_stixel.width - 1 ) / 2 + stixel_horizontal_padding < current_image_view.width()
        //        /*and current_stixel.type != Stixel::Occluded and next_stixel.type != Stixel::Occluded*/
        //);

        float neighbour_difference =  k_z * std::max( alpha_z, 1 - current_neighbour_stixel_depth_differences( col ) / delta_z ) +
                k_h * std::max( alpha_h, 1 - current_neighbour_stixel_real_height_differences( col ) / delta_h );

        neighbour_differences( col ) = neighbour_difference;

        for(int m = 0; m < number_of_motions; ++m )
        {

            Eigen::MatrixXf group_cost_values =
                    Eigen::MatrixXf::Zero( maximum_number_of_one_to_many_stixels_matching + 1,
                                           number_of_total_motions );

            float min_group_dp_cost = std::numeric_limits< float >::max();

            int winning_index_k = -1;

            for( int k = 0; k <= maximum_number_of_one_to_many_stixels_matching; ++k )
            {
                if( ((m - k) >= 0) and ((m + k) < number_of_motions)  )
                {
                    float min_M_plus_c = std::numeric_limits< float >::max();

                    const int number_of_contributing_motions = 2 * k + 1;

                    float group_motion_cost = 0;

                    // Compute the matching cost representing the multiple motion group
                    for( int l = m - k; l <= (m + k); ++l )
                    {
                        group_motion_cost += motion_cost_matrix( l, col );

                    } // End of for(l)

                    group_motion_cost /= number_of_contributing_motions;

                    for( int e = 0; e < number_of_total_motions; ++e )
                    {
                        float motion_abs_difference = 0;
                        //                        float real_motion_vectors_difference = 0;
                        float c_cost = 0;

                        for( int l = m - k; l <= m + k; ++l )
                        {
                            float abs_diff;
                            //                            float real_motion_vec_diff;

                            if( l < number_of_motions and e < number_of_motions )
                            {
                                if( motion_cost_assignment_matrix( l, col ) == true and motion_cost_assignment_matrix( e, next_col ) == true )
                                {
                                    abs_diff = float( abs( l - e ) ) / ( number_of_motions - 1 );

                                    //                                    real_motion_vec_diff =
                                    //                                            ( ( real_motion_vectors_matrix[ col ][ l ] - real_motion_vectors_matrix[ next_col ][ e ] ).norm() ) / maximum_displacement_between_frames;
                                }
                                else
                                {
                                    abs_diff = 1.0f;
                                    //                                    real_motion_vec_diff = 1.0f;
                                }

                            }
                            else if( l == number_of_motions and e == number_of_motions ) // This else if is unnecessary for the moment !
                            {
                                abs_diff = 0;
                                //                                real_motion_vec_diff = 0;
                            }
                            else
                            {
                                abs_diff = 1.0f;
                                //                                real_motion_vec_diff = 1.0f;
                            }

                            motion_abs_difference += abs_diff;
                            //                            real_motion_vectors_difference += real_motion_vec_diff;

                        } // End of for( l )

                        motion_abs_difference /= number_of_contributing_motions;
                        //                        real_motion_vectors_difference /= number_of_contributing_motions;

                        c_cost = motion_abs_difference * neighbour_difference;

                        if( next_stixel.type == Stixel::Occluded )
                        {
                            c_cost = 0;
                        }

                        group_cost_values( k, e ) = c_cost;

                        const float t_cost = M_cost_dp( e, next_col ) + c_cost;

                        min_M_plus_c = std::min( t_cost, min_M_plus_c );

                    } // End of for( e )

                    if( group_motion_cost + min_M_plus_c < min_group_dp_cost )
                    {
                        min_group_dp_cost = group_motion_cost + min_M_plus_c;
                        winning_index_k = k;
                    }

                }
                else
                {
                    break;
                }

            } // End of for( k )

            M_cost_dp( m, col ) = min_group_dp_cost;

            for( int e = 0; e < number_of_total_motions; ++e )
            {
                c_cost_matrix[ col ][ m ][ e ] = group_cost_values( winning_index_k, e );
            }

            stixels_one_to_many_assignment_map( m, col ) = winning_index_k;

        } // End of for( m )

        { // Process m == number_of_motions specially !!!

            const int m = number_of_motions;

            float min_M_plus_c = std::numeric_limits< float >::max();

            for( int e = 0; e < number_of_total_motions; ++e )
            {
                // ( m - max_motion_in_pixels ) - ( e - max_motion_in_pixels )
                float motion_abs_difference;
                //float real_motion_vectors_difference;

                if( e == number_of_motions )
                {
                    motion_abs_difference = 0;
                    //real_motion_vectors_difference = 0;
                }
                else
                {
                    motion_abs_difference = 1.0f;
                    //real_motion_vectors_difference = 1.0f;
                }

                float c_cost = motion_abs_difference * neighbour_difference;

                if( next_stixel.type == Stixel::Occluded )
                {
                    c_cost = 0;
                }

                // Store for future use
                c_cost_matrix[ col ][ m ][ e ] = c_cost;

                const float t_cost = M_cost_dp( e, next_col ) + c_cost;

                min_M_plus_c = std::min( t_cost, min_M_plus_c );

            } // End of for( e )

            M_cost_dp( m, col ) += min_M_plus_c;

            stixels_one_to_many_assignment_map( m, col ) = 0;

        } // End of special processing( m == number_of_motions )


    } // End of for( col )

    return;
}

void DummyStixelMotionEstimator::dp_second_pass()
{
    const int number_of_cols = motion_cost_matrix.cols();

    const unsigned int number_of_motions = motion_cost_matrix.rows() - 1; // The number of motions in range
    //const int number_of_total_motions = number_of_motions + 1; // The last row is for "disappearing" motion - special case

    assert(motion_cost_matrix.rows() >= 1);

    //const size_t number_of_current_stixels = current_stixels_p->size();
    const int number_of_previous_stixels = previous_stixels_p->size();

    indices_columnwise_minimum_cost = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >( motion_cost_matrix.cols(), 1 ); // Vector is uninitialized

    const motion_cost_matrix_t& const_M_cost_dp = M_cost_dp;

    { // The first value is set directly

        Stixel& current_stixel = ( *current_stixels_p )[ 0 ];

        unsigned int &d_star = indices_columnwise_minimum_cost( 0 );

        // minCoeff takes the "lowest index", but
        // we search for the maximum index with minimum value
        //        const_M_cost_dp.col( 0 ).minCoeff( &d_star );

        float d_star_score = const_M_cost_dp( 0, 0 );
        d_star = 0;

        for(unsigned int i = 0; i < number_of_motions; ++i )
        {
            if( const_M_cost_dp( i, 0 ) < d_star_score )
            {
                const int motion = i - maximum_possible_motion_in_pixels;

                if( motion >= 0 and motion < number_of_previous_stixels ) // 0 + motion
                {
                    d_star_score = const_M_cost_dp( i, 0 );
                    d_star = i;
                }
            }

        }

        { // Treat the last entry specially

            if( const_M_cost_dp( number_of_motions, 0 ) < d_star_score )
            {
                d_star_score = const_M_cost_dp( number_of_motions, 0 );
                d_star = number_of_motions;
            }
        }

        if( d_star == number_of_motions ) // Disappearing stixel
        {
            stixels_motion[ 0 ] = -1;

            current_stixel.backward_delta_x = 0;
            current_stixel.valid_backward_delta_x = false;
            current_stixel.backward_width = 0;
        }
        else
        {
            const int motion = d_star - maximum_possible_motion_in_pixels;
            const int previous_stixel_center_x = current_stixel.x + motion;

            /// FIXME : Should be computed intelligently if the stixel width(s) is(are) not 1.
            const int previous_stixel_index = previous_stixel_center_x;

            stixels_motion[ 0 ] = previous_stixel_index;

            const int stixels_one_to_many_assignment = stixels_one_to_many_assignment_map( d_star, 0 );

            current_stixel.backward_width = 2 * stixels_one_to_many_assignment + 1;
            current_stixel.backward_delta_x = motion - stixels_one_to_many_assignment;
            current_stixel.valid_backward_delta_x = true;
        }

    }

    for( int col = 1; col < number_of_cols; ++col )
    {
        const int prev_col = col - 1;

        Stixel& current_stixel = ( *current_stixels_p )[ col ];

        const unsigned int previous_d_star = indices_columnwise_minimum_cost( prev_col );

        unsigned int &d_star = indices_columnwise_minimum_cost( col );

        float min_M_minus_c = std::numeric_limits<float>::max();

        for(unsigned int e = 0; e < number_of_motions; ++e )
        {
            const float c_cost = c_cost_matrix[ prev_col ][ previous_d_star ][ e ];
            const float t_cost = const_M_cost_dp( e, col ) + c_cost;

            if( t_cost <= min_M_minus_c )
            {
                const int motion = e - maximum_possible_motion_in_pixels;

                const int previous_stixel_center_x = current_stixel.x + motion;

                /// FIXME : Should be computed intelligently if the stixel width(s) is(are) not 1.
                const int previous_stixel_index = previous_stixel_center_x;

                if( previous_stixel_index >= 0 and previous_stixel_index < number_of_previous_stixels )
                {
                    d_star = e;
                    min_M_minus_c = t_cost;
                }

            }

        } // End of for( e )

        { // Treat the last entry specially

            const float c_cost = c_cost_matrix[ prev_col ][ previous_d_star ][ number_of_motions ];
            const float t_cost = const_M_cost_dp( number_of_motions, col ) + c_cost;

            if( t_cost <= min_M_minus_c )
            {
                d_star = number_of_motions;
                min_M_minus_c = t_cost;
            }

        }

        if( d_star == number_of_motions ) // Disappearing stixel
        {
            stixels_motion[ col ] = -1;

            current_stixel.backward_delta_x = 0;
            current_stixel.valid_backward_delta_x = false;
            current_stixel.backward_width = 0;
        }
        else
        {
            const int motion = d_star - maximum_possible_motion_in_pixels;
            const int previous_stixel_center_x = current_stixel.x + motion;

            /// FIXME : Should be computed intelligently if the stixel width(s) is(are) not 1.
            const int previous_stixel_index = previous_stixel_center_x;

            stixels_motion[ col ] = previous_stixel_index;

            const int stixels_one_to_many_assignment = stixels_one_to_many_assignment_map( d_star, col );

            current_stixel.backward_width = 2 * stixels_one_to_many_assignment + 1;
            current_stixel.backward_delta_x = motion - stixels_one_to_many_assignment;
            current_stixel.valid_backward_delta_x = true;
        }

    } // End of for( col )

    return;
}


float DummyStixelMotionEstimator::compute_pixelwise_sad( const Stixel& stixel1, const Stixel& stixel2,
                                                         const input_image_const_view_t& image_view1, const input_image_const_view_t& image_view2,
                                                         const unsigned int stixel_horizontal_padding ) const
{
    const unsigned int stixel_representation_width = stixel1.width + 2 * stixel_horizontal_padding;

    const unsigned int number_of_channels = image_view1.num_channels();

    stixel_representation_t stixel_representation1;
    stixel_representation_t stixel_representation2;

    compute_stixel_representation( stixel1, image_view1, stixel_representation1, stixel_horizontal_padding );
    compute_stixel_representation( stixel2, image_view2, stixel_representation2, stixel_horizontal_padding );

    float pixelwise_sad = 0;

    for( unsigned int c = 0; c < number_of_channels; ++c )
    {
        const Eigen::MatrixXf& current_stixel_representation_channel = stixel_representation1[ c ];
        const Eigen::MatrixXf& previous_stixel_representation_channel = stixel_representation2[ c ];

        for( int y = 0; y < stixel_representation_height; ++y )
        {
            for( unsigned int x = 0; x < stixel_representation_width; ++x )
            {
                pixelwise_sad += fabs( current_stixel_representation_channel( y, x ) - previous_stixel_representation_channel( y, x ) );

            } // End of for( x )

        } // End of for( y )

    } // End of for( c )

    pixelwise_sad = pixelwise_sad / number_of_channels;
    pixelwise_sad = pixelwise_sad / ( stixel_representation_height * stixel_representation_width );

    stixel_representation1.clear();
    stixel_representation2.clear();

    return pixelwise_sad;
}


Eigen::Vector3f DummyStixelMotionEstimator::compute_real_motion_between_stixels( const Stixel& reference_stixel, const Stixel& destination_stixel ) const
{
    const float reference_stixel_disparity = std::max< float >( min_float_disparity, reference_stixel.disparity );
    const float destination_stixel_disparity = std::max< float >( min_float_disparity, destination_stixel.disparity );

    const float reference_stixel_depth = stereo_camera.disparity_to_depth( reference_stixel_disparity );
    const float destination_stixel_depth = stereo_camera.disparity_to_depth( destination_stixel_disparity );

    return compute_real_motion_between_stixels( reference_stixel, destination_stixel, reference_stixel_depth, destination_stixel_depth );
}


Eigen::Vector3f DummyStixelMotionEstimator::compute_real_motion_between_stixels( const Stixel& reference_stixel, const Stixel& destination_stixel,
                                                                                 const float reference_stixel_depth, const float destination_stixel_depth ) const
{
    const MetricCamera& left_camera = stereo_camera.get_left_camera();

    const Eigen::Vector2f reference_stixel_bottom_point( reference_stixel.x, reference_stixel.bottom_y );
    const Eigen::Vector2f destination_stixel_bottom_point( destination_stixel.x, destination_stixel.bottom_y );

    Eigen::Vector3f reference_stixel_bottom_point_3d = left_camera.back_project_2d_point_to_3d( reference_stixel_bottom_point, reference_stixel_depth );
    Eigen::Vector3f destination_stixel_bottom_point_3d = left_camera.back_project_2d_point_to_3d( destination_stixel_bottom_point, destination_stixel_depth );

    return ( destination_stixel_bottom_point_3d - reference_stixel_bottom_point_3d );
}


float DummyStixelMotionEstimator::compute_stixel_real_height_difference( const Stixel& stixel1, const Stixel& stixel2 ) const
{
    const float stixel_height1 = compute_stixel_real_height( stixel1 );
    const float stixel_height2 = compute_stixel_real_height( stixel2 );

    const float stixel_height_difference = fabs( stixel_height1 - stixel_height2 );

    return stixel_height_difference;

}


float DummyStixelMotionEstimator::compute_stixel_real_height( const Stixel& stixel ) const
{
    const float stixel_disparity = std::max< float >( min_float_disparity, stixel.disparity );
    const float stixel_depth = stereo_camera.disparity_to_depth( stixel_disparity );

    const float stixel_real_height = compute_stixel_real_height( stixel, stixel_depth );

    return stixel_real_height;
}


float DummyStixelMotionEstimator::compute_stixel_real_height( const Stixel& stixel, const float stixel_depth ) const
{
    const MetricCamera& left_camera = stereo_camera.get_left_camera();

    Eigen::Vector2f stixel_top_point( stixel.x, stixel.top_y );
    Eigen::Vector2f stixel_bottom_point( stixel.x, stixel.bottom_y );

    Eigen::Vector3f stixel_top_point_3d = left_camera.back_project_2d_point_to_3d( stixel_top_point, stixel_depth );
    Eigen::Vector3f stixel_bottom_point_3d = left_camera.back_project_2d_point_to_3d( stixel_bottom_point, stixel_depth );

    const float stixel_real_height = fabs( stixel_top_point_3d( i_y ) - stixel_bottom_point_3d( i_y ) );

    return stixel_real_height;
}


void DummyStixelMotionEstimator::compute_stixel_representation( const Stixel& stixel, const input_image_const_view_t& image_view_hosting_the_stixel,
                                                                stixel_representation_t& stixel_representation,
                                                                const unsigned int stixel_horizontal_padding, const unsigned int stixel_representation_width) const
{
    throw std::runtime_error( "DummyStixelMotionEstimator::compute_stixel_representation() -- This version is being used !" );

    if( stixel_representation_width == stixel.width + 2 * stixel_horizontal_padding )
    {
        compute_stixel_representation( stixel, image_view_hosting_the_stixel, stixel_representation, stixel_horizontal_padding );
    }
    else
    {
        const unsigned int stixel_representation_width_initial = stixel.width + 2 * stixel_horizontal_padding;

        if( (stixel.width % 2) != 1 )
        {
            throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The width of stixel should be an odd number !" );
        }

        if( stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding < 0 or
                stixel.x + ( stixel.width - 1 ) / 2 + stixel_horizontal_padding >= image_view_hosting_the_stixel.width() )
        {
            throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The stixel representation should obey the image boundaries !" );
        }

        if( (( stixel_representation_width - stixel.width )) % 2 != 0 )
        {
            throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- Stixel representation should obey padding rules !" );
        }

        //const int stixel_representation_padding = ( stixel_representation_width - stixel.width ) / 2;

        const unsigned int stixel_height = abs( stixel.top_y - stixel.bottom_y );

        const float horizontal_scaling_factor = float( stixel_representation_width ) / float( stixel_representation_width_initial );
        const float vertical_scaling_factor = float( stixel_representation_height ) / float( stixel_height );

        // Sub-image representation for the input stixel
        //        input_image_const_view_t stixel_view = boost::gil::subimage_view( image_view_hosting_the_stixel,
        //                                                                          stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding, stixel.top_y,
        //                                                                          stixel_representation_width_initial, abs( stixel.bottom_y - stixel.top_y ) );

        input_image_const_view_t stixel_view = boost::gil::subimage_view( image_view_hosting_the_stixel,
                                                                          stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding, stixel.top_y,
                                                                          stixel_representation_width_initial, abs( stixel.bottom_y - stixel.top_y ) / 2 );

        const unsigned int number_of_channels = image_view_hosting_the_stixel.num_channels();

        stixel_representation.clear();
        stixel_representation.resize( number_of_channels );

        for( unsigned int c = 0; c < number_of_channels; ++c )
        {
            stixel_representation[ c ].resize( stixel_representation_height, stixel_representation_width );

        } // End of for( c )

        for(int y = 0; y < stixel_representation_height; ++y )
        {
            const float projected_y = float( y ) / vertical_scaling_factor;

            const float projected_upper_y = std::ceil( projected_y );
            const float projected_lower_y = std::floor( projected_y );

            const float delta_y1 = fabs( projected_y - projected_lower_y );
            const float delta_y2 = fabs( projected_y - projected_upper_y );

            const float delta_y1_square = delta_y1 * delta_y1;
            const float delta_y2_square = delta_y2 * delta_y2;

            //input_image_const_view_t::x_iterator src_iter_lower = stixel_view.row_begin( int( projected_lower_y ) );
            //input_image_const_view_t::x_iterator src_iter_upper = stixel_view.row_begin( int( projected_upper_y ) );

            for( unsigned int x = 0; x < stixel_representation_width; ++x )
            {
                const float projected_x = float( x ) / horizontal_scaling_factor;

                const float projected_upper_x = std::ceil( projected_x );
                const float projected_lower_x = std::floor( projected_x );

                const float delta_x1 = fabs( projected_x - projected_lower_x );
                const float delta_x2 = fabs( projected_x - projected_upper_x );

                const float delta_x1_square = delta_x1 * delta_x1;
                const float delta_x2_square = delta_x2 * delta_x2;

                float coeff1 = 1 / sqrt( delta_x1_square + delta_y1_square );
                float coeff2 = 1 / sqrt( delta_x1_square + delta_y2_square );
                float coeff3 = 1 / sqrt( delta_x2_square + delta_y1_square );
                float coeff4 = 1 / sqrt( delta_x2_square + delta_y2_square );

                const float coeff_sum = coeff1 + coeff2 + coeff3 + coeff4;

                coeff1 = coeff1 / coeff_sum;
                coeff2 = coeff2 / coeff_sum;
                coeff3 = coeff3 / coeff_sum;
                coeff4 = coeff4 / coeff_sum;

                for( unsigned int c = 0; c < number_of_channels; ++c )
                {
                } // End of for( c )

            } // End of for( x )

        } // End of for( y )


    }
}

void DummyStixelMotionEstimator::compute_stixel_representation( const Stixel &stixel, const input_image_const_view_t& image_view_hosting_the_stixel,
                                                                stixel_representation_t &stixel_representation, const unsigned int stixel_horizontal_padding ) const
{
    const unsigned int stixel_representation_width = stixel.width + 2 * stixel_horizontal_padding;

    const int stixel_height = abs( stixel.top_y - stixel.bottom_y );
    const int stixel_effective_part_height = stixel_height;

    const float reduction_ratio = float( stixel_representation_height ) / float( stixel_effective_part_height );

    // Image boundary conditions are NOT checked for speed efficiency !
    if( (stixel.width % 2) != 1 )
    {
        printf("stixel.width == %i\n", stixel.width);
        throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The width of stixel should be an odd number !" );
    }

    if( stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding < 0 or
            stixel.x + ( stixel.width - 1 ) / 2 + stixel_horizontal_padding >= image_view_hosting_the_stixel.width() )
    {
        throw std::invalid_argument( "DummyStixelMotionEstimator::compute_stixel_representation() -- The stixel representation should obey the image boundaries !" );
    }

    input_image_const_view_t stixel_view = boost::gil::subimage_view( image_view_hosting_the_stixel,
                                                                      stixel.x - ( stixel.width - 1 ) / 2 - stixel_horizontal_padding, stixel.top_y,
                                                                      stixel_representation_width, stixel_effective_part_height );

    const unsigned int number_of_channels = image_view_hosting_the_stixel.num_channels();

    stixel_representation.clear();
    stixel_representation.resize( number_of_channels );

    for( unsigned int c = 0; c < number_of_channels; ++c )
    {
        stixel_representation[ c ].resize( stixel_representation_height, stixel_representation_width );

    } // End of for( c )

    for( int y = 0; y < stixel_representation_height; ++y )
    {
        const float projected_y = float( y ) / reduction_ratio;

        const float projected_upper_y = std::ceil( projected_y );
        const float projected_lower_y = std::floor( projected_y );

        // The coefficients are in reverse order (sum of coefficients is 1)
        float coefficient_lower_y = projected_upper_y - projected_y;
        float coefficient_upper_y = projected_y - projected_lower_y;

        if( coefficient_lower_y + coefficient_upper_y < 0.05 ) // If the projected pixel falls just on top of an integer coordinate
        {
            coefficient_lower_y = 0.5;
            coefficient_upper_y = 0.5;
        }

        input_image_const_view_t::x_iterator src_iter_lower = stixel_view.row_begin( int( projected_lower_y ) );
        input_image_const_view_t::x_iterator src_iter_upper = stixel_view.row_begin( int( projected_upper_y ) );

        for( unsigned int x = 0; x < stixel_representation_width; ++x )
        {
            for( unsigned int c = 0; c < number_of_channels; ++c )
            {
                ( stixel_representation[ c ] )( y, x ) = coefficient_lower_y * src_iter_lower[ x ][ c ] +
                        coefficient_upper_y * src_iter_upper[ x ][ c ];

            } // End of for( c )

        } // End of for( x )

    } // End of for( y )

    return;
}

unsigned int DummyStixelMotionEstimator::compute_maximum_pixelwise_motion_for_stixel( const Stixel& stixel )
{
    float disparity = std::max< float >( min_float_disparity, stixel.disparity );
    float depth = stereo_camera.disparity_to_depth( disparity );

    Eigen::Vector3f point3d1( -maximum_displacement_between_frames / 2, 0, depth );
    Eigen::Vector3f point3d2( maximum_displacement_between_frames / 2, 0, depth );

    const MetricCamera& left_camera = stereo_camera.get_left_camera();

    Eigen::Vector2f point2d1 = left_camera.project_3d_point( point3d1 );
    Eigen::Vector2f point2d2 = left_camera.project_3d_point( point3d2 );

    const unsigned int maximum_motion_in_pixels = static_cast<unsigned int>( fabs( point2d2[ 0 ] - point2d1[ 0 ] ) );

    return maximum_motion_in_pixels;
}

unsigned int DummyStixelMotionEstimator::compute_maximum_motion_in_pixels()
{
    float disparity = std::max< float >( min_float_disparity, current_stixels_p->at(0).disparity );

    float min_depth = stereo_camera.disparity_to_depth( disparity );

    for( unsigned int i = 1, number_of_stixels = current_stixels_p->size(); i < number_of_stixels; ++i )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ i ];

        if( current_stixel.type != Stixel::Occluded )
        {
            disparity = std::max< float >( min_float_disparity, current_stixel.disparity );

            float depth = stereo_camera.disparity_to_depth( float( disparity ) );

            if( depth < min_depth )
            {
                min_depth = depth;
            }
        }

    } // End of for( i )

    Eigen::Vector3f point3d1( 0, 0, min_depth );
    Eigen::Vector3f point3d2( maximum_displacement_between_frames, 0, min_depth );

    const MetricCamera& left_camera = stereo_camera.get_left_camera();

    Eigen::Vector2f point2d1 = left_camera.project_3d_point( point3d1 );
    Eigen::Vector2f point2d2 = left_camera.project_3d_point( point3d2 );

    const unsigned int maximum_motion_in_pixels = static_cast<unsigned int>(point2d2[0] - point2d1[0]);

    return maximum_motion_in_pixels;
}

/// FIXME: Drawing methods should not be in the logic code
void DummyStixelMotionEstimator::update_stixel_tracks_image()
{
    //    std::cout << "UpdateStixelTracks()" << std::endl;

    const unsigned int rows_per_frame = number_of_rows_per_frame_in_stixel_track_visualization;

    typedef AbstractStixelMotionEstimator::input_image_view_t input_view_t;

    // prepare the frame for new drawings in last row
    {
        // Copy the oldest N - 1 frame tracks
        input_view_t upper_sub_view =
                boost::gil::subimage_view( stixel_tracks_view,
                                           0, 0,
                                           stixel_tracks_view.width(), (number_of_frames_in_history - 1)*rows_per_frame );
        input_view_t lower_sub_view =
                boost::gil::subimage_view( stixel_tracks_view,
                                           0, rows_per_frame,
                                           upper_sub_view.width(), upper_sub_view.height() );
        // src -> dst
        boost::gil::copy_pixels( lower_sub_view, upper_sub_view );

        // Fill in the stixel area with the selected color
        input_view_t last_row_sub_view =
                boost::gil::subimage_view( stixel_tracks_view,
                                           0, ( number_of_frames_in_history - 1 ) * rows_per_frame,
                                           stixel_tracks_view.width(), rows_per_frame );

        boost::gil::fill_pixels( last_row_sub_view, rgb8_colors::black );

    }

    // propagate colors to last row
    for( unsigned int s_current = 0, number_of_current_stixels = current_stixels_p->size();
         s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];

        //if( stixels_motion[ s_current ] >= 0 )
        if(current_stixel.valid_backward_delta_x)
        {   // stixel has a valid match
            // copy colors from previous stixel to current stixel
            const unsigned int s_prev = stixels_motion[ s_current ];

            const Stixel& previous_stixel = ( *previous_stixels_p )[ s_prev ];

            // (previous_stixel.width / 2) == 0 when stixel.width == 1
            input_view_t
                    upper_sub_view =
                    boost::gil::subimage_view( stixel_tracks_view,
                                               previous_stixel.x - (previous_stixel.width / 2),
                                               ( number_of_frames_in_history - 2 ) * rows_per_frame,
                                               previous_stixel.width, rows_per_frame );

            input_view_t lower_sub_view =
                    boost::gil::subimage_view( stixel_tracks_view,
                                               current_stixel.x - (current_stixel.width / 2),
                                               ( number_of_frames_in_history - 1 ) * rows_per_frame,
                                               current_stixel.width, rows_per_frame );

            boost::gil::copy_pixels( upper_sub_view, lower_sub_view );

            current_stixel_color_indices( s_current ) = previous_stixel_color_indices( s_prev );

            //            color_occupancy_vector( current_stixel_color_indices( s_current ) ) = true;
        }
    }

    //unsigned int unoccupied_color_search_start = 0;

    for( unsigned int s_current = 0, number_of_current_stixels = current_stixels_p->size(); s_current < number_of_current_stixels; ++s_current )
    {
        const Stixel& current_stixel = ( *current_stixels_p )[ s_current ];

        if( stixels_motion[ s_current ] < 0 ) // Unmatched stixels
        {
            //            // Search for the next unoccupied color
            //            for( unsigned int color = unoccupied_color_search_start; color < number_of_colors; ++color )
            //            {
            //                if( color_occupancy_vector( color ) == false )
            //                {
            //                    unoccupied_color_search_start = color;
            //                    break;
            //                }
            //            }

            //            // Extract the color
            //            const boost::gil::rgb8c_pixel_t t_color = boost::gil::rgb8_view_t::value_type( jet_color_map[ unoccupied_color_search_start ][ 0 ],
            //                                                                                           jet_color_map[ unoccupied_color_search_start ][ 1 ],
            //                                                                                           jet_color_map[ unoccupied_color_search_start ][ 2 ] );

            // Update the color occupancy map
            //            color_occupancy_vector( unoccupied_color_search_start ) = true;

            // Fill in the stixel area with the selected color
            AbstractStixelMotionEstimator::input_image_view_t stixel_sub_view =
                    boost::gil::subimage_view( stixel_tracks_view,
                                               current_stixel.x - current_stixel.width / 2, ( number_of_frames_in_history - 1 ) * number_of_rows_per_frame_in_stixel_track_visualization,
                                               current_stixel.width, number_of_rows_per_frame_in_stixel_track_visualization );

            //            boost::gil::fill_pixels( stixel_sub_view, t_color );
            boost::gil::fill_pixels( stixel_sub_view, rgb8_colors::black );

        }
    }

    return;
}


void DummyStixelMotionEstimator::reset_stixel_tracks_image()
{
    // FIXME this code should be in the GUI, not in the logic

    using rgb8_colors::jet_color_map;

    number_of_rows_per_frame_in_stixel_track_visualization = 10;
    const int image_width = 640; /// FIXME : Will be obtained automatically
    const int image_height = 480; /// FIXME : Will be obtained automatically
    number_of_frames_in_history = image_height / number_of_rows_per_frame_in_stixel_track_visualization;

    stixel_tracks_image.recreate( image_width, image_height );
    stixel_tracks_view = boost::gil::view( stixel_tracks_image );
    boost::gil::fill_pixels( stixel_tracks_view, rgb8_colors::white );

    current_stixel_color_indices.resize( image_width, 1 );
    previous_stixel_color_indices.resize( image_width, 1 );

    for( int i = 0; i < image_width; ++i )
    {
        AbstractStixelMotionEstimator::input_image_view_t stixel_sub_view =
                boost::gil::subimage_view( stixel_tracks_view,
                                           i, 0,
                                           1, image_height );

        const boost::gil::rgb8c_pixel_t t_color = boost::gil::rgb8_view_t::value_type( jet_color_map[ i ][ 0 ],
                jet_color_map[ i ][ 1 ],
                jet_color_map[ i ][ 2 ] );

        boost::gil::fill_pixels( stixel_sub_view, t_color );

        current_stixel_color_indices( i ) = i;
        previous_stixel_color_indices( i ) = i;

    } // End of for( i )

    return;
}

} // namespace doppia
