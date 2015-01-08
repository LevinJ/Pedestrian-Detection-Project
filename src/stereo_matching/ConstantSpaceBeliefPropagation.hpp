#ifndef CONSTANTSPACEBELIEFPROPAGATION_HPP
#define CONSTANTSPACEBELIEFPROPAGATION_HPP


#include "AbstractStereoBlockMatcher.hpp"

#include <boost/scoped_array.hpp>

// forward declarations ---
template<typename T> class qx_csbp_rgb;
class qx_disparity_map_bf_refinement;

namespace doppia
{

/**
QingXiong Yang CPU implementation of
Qingxiong Yang, Liang Wang and Narendra Ahuja,
A Constant-Space Belief Propagation Algorithm for Stereo Matching,
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2010.
*/
class ConstantSpaceBeliefPropagation: public AbstractStereoBlockMatcher
{


public:

    static boost::program_options::options_description get_args_options();

    ConstantSpaceBeliefPropagation(const boost::program_options::variables_map &options);
    ~ConstantSpaceBeliefPropagation();

    void set_rectified_images_pair( any_image<input_images_t>::const_view_t &left, any_image<input_images_t>::const_view_t &right);

    void compute_disparity_map(gray8c_view_t &left, gray8c_view_t &right, const bool left_right_are_inverted);

    void compute_disparity_map(rgb8c_view_t  &left, rgb8c_view_t &right, const bool left_right_are_inverted);

private:

    typedef rgb8c_view_t::point_t dimensions_t;

    // options
    bool use_local_minima;
    bool do_post_processing;
    int num_scales;
    int iterations_per_scale;
    int max_nr_plane;
    short discontinuity_cost_single_jump;
    int bilateral_filter_iterations;
    int bilateral_filter_radius;
    //int pixels_matching_method;

    boost::scoped_ptr<qx_csbp_rgb<unsigned char> > csbp_p;
    boost::scoped_ptr<qx_disparity_map_bf_refinement> disparity_refinement_p;

    boost::scoped_array<unsigned char> qx_image_left_mem;
    boost::scoped_array<unsigned char> qx_image_right_mem;
    unsigned char *qx_image_left, *qx_image_right;
};


} // end of namespace doppia


#endif // CONSTANTSPACEBELIEFPROPAGATION_HPP
