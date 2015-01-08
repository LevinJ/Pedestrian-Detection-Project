#ifndef DOPPIA_DUMMYSTIXELMOTIONESTIMATOR_HPP
#define DOPPIA_DUMMYSTIXELMOTIONESTIMATOR_HPP

#include "AbstractStixelMotionEstimator.hpp"

#include <boost/program_options.hpp>

#include <vector>

namespace doppia {

class MetricStereoCamera; // forward declaration

class DummyStixelMotionEstimator : public doppia::AbstractStixelMotionEstimator
{
public:

    typedef std::vector< Eigen::MatrixXf > stixel_representation_t;

    static boost::program_options::options_description get_args_options();

    DummyStixelMotionEstimator( const boost::program_options::variables_map &options,
                                const MetricStereoCamera &camera, int stixels_width );

    ~DummyStixelMotionEstimator();


    void compute();

    void update_stixel_tracks_image();
    void reset_stixel_tracks_image();

protected:

    void compute_motion_cost_matrix();
    void compute_motion();

    void initialize_matrices();

    void compute_matching_cost_matrix();
    void compute_motion_v1();

    unsigned int compute_maximum_motion_in_pixels();
    unsigned int compute_maximum_pixelwise_motion_for_stixel( const Stixel& stixel );

    inline int compute_stixel_horizontal_padding( const Stixel& stixel );

    float compute_pixelwise_sad( const Stixel& stixel1, const Stixel& stixel2,
                                 const input_image_const_view_t& image_view1, const input_image_const_view_t& image_view2,
                                 const unsigned int stixel_horizontal_padding ) const;

//    float compute_pixelwise_sad_v1( const Stixel& stixel1, const Stixel& stixel2,
//                                    const input_image_const_view_t& image_view1, const input_image_const_view_t& image_view2,
//                                    const unsigned int stixel_horizontal_padding ) const;

    float compute_stixel_real_height_difference( const Stixel& stixel1, const Stixel& stixel2 ) const;
    float compute_stixel_real_height( const Stixel& stixel ) const;
    float compute_stixel_real_height( const Stixel& stixel, const float stixel_depth ) const;

    /// Computes the 3D motion vector from reference_stixel to destination_stixel
    Eigen::Vector3f compute_real_motion_between_stixels( const Stixel& reference_stixel, const Stixel& destination_stixel ) const;
    Eigen::Vector3f compute_real_motion_between_stixels( const Stixel& reference_stixel, const Stixel& destination_stixel,
                                                         const float reference_stixel_depth, const float destination_stixel_depth ) const;

    void compute_stixel_representation( const Stixel &stixel, const input_image_const_view_t& image_hosting_the_stixel,
                                        stixel_representation_t &stixel_representation, const unsigned int stixel_horizontal_padding ) const;

    void compute_stixel_representation( const Stixel &stixel, const input_image_const_view_t& image_hosting_the_stixel,
                                        stixel_representation_t &stixel_representation,
                                        const unsigned int stixel_horizontal_padding, const unsigned int stixel_representation_width ) const;

    void compute_motion_dp();
    void compute_motion_dp_v1();

    void dp_first_pass_v1( Eigen::MatrixXi& M_cost_source_indicator_matrix );
    void dp_second_pass_v1( const Eigen::MatrixXi& M_cost_source_indicator_matrix );

    void dp_first_pass();
    void dp_first_pass_one_to_many();
    void dp_second_pass();

    void fill_in_visualization_motion_cost_matrix(); // Helper visualization method

    /**
      * CLASS DATA ATTRIBUTES
      **/

    /*
     * Variables initialized in the constructor initialization list
     *
     * The order here should be the same with the order in the constructor initialization list
     */

    const MetricStereoCamera &stereo_camera;

    const float average_pedestrian_speed;
    const float maximum_pedestrian_speed;
    const int video_frame_rate;
    const int frame_width;
    const int frame_height;
    const int minimum_object_height_in_pixels;
    const int stixel_representation_height;
    const int maximum_possible_motion_in_pixels; // Maximum possible motion in pixels over the whole sequence (not specific to a particular frame)
    const int number_of_stixels_per_frame;
    const int maximum_number_of_one_to_many_stixels_matching; // If this is n, then one stixel might be assigned to 2n+1 stixels at most

    /*
     * Variables initialized after the constructor initialization list
     *
     */

    friend class StixelWorldGui; // used for debugging only

    float maximum_displacement_between_frames;

    // Stored for easy access
    unsigned int maximum_pixel_value;
    float maximum_depth_value;

    /// FIXME :
    /// All those matrices and vectors should be (?) deallocated after the computation for the current frame is over

    motion_cost_matrix_t pixelwise_sad_matrix;
    motion_cost_matrix_t real_height_differences_matrix;
    motion_cost_matrix_t pixelwise_height_differences_matrix;
    motion_cost_matrix_t depth_differences_matrix;

    // Stored for easy access
    Eigen::VectorXf current_stixel_depths;
    Eigen::VectorXf current_stixel_real_heights;

    matrix_of_3d_vectors_t real_motion_vectors_matrix; // Each matrix[i][j] element is a 3D vector

    // Cost values for edit distance dynamic programming
    float insertion_cost_dp;
    float deletion_cost_dp;

    unsigned int number_of_rows_per_frame_in_stixel_track_visualization;
    unsigned int number_of_frames_in_history;

    Eigen::MatrixXi current_stixel_color_indices; // Stores the information of which stixel is visualized with which color from the jet_color_map.
    Eigen::MatrixXi previous_stixel_color_indices; // Stores the information of which stixel is visualized with which color from the jet_color_map.





    //
};

/// FIXME : Should this be inline ?
int DummyStixelMotionEstimator::compute_stixel_horizontal_padding( const Stixel& stixel )
{
    // FIXME hardcoded parameter
    return ( (stixel.width / 2) + 2 );
}

} // namespace doppia

#endif // DOPPIA_DUMMYSTIXELMOTIONESTIMATOR_HPP
