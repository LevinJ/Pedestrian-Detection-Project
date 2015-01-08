#ifndef SIMPLEBLOCKMATCHER_HPP
#define SIMPLEBLOCKMATCHER_HPP

#include "AbstractStereoBlockMatcher.hpp"

namespace doppia
{

namespace gil = boost::gil;

class SimpleBlockMatcher : public AbstractStereoBlockMatcher
{

    typedef gil::gray32f_image_t mismatch_map_t;
    mismatch_map_t mismatch_map;
    mismatch_map_t::view_t mismatch_map_view;

    typedef gil::gray32f_image_t lowest_mismatch_map_t;
    lowest_mismatch_map_t lowest_mismatch_map;
    lowest_mismatch_map_t::view_t lowest_mismatch_view;

    disparity_map_t left_to_right_disparity_map, right_to_left_disparity_map;
    disparity_map_t::view_t left_to_right_disparity_map_view, right_to_left_disparity_map_view;

public:

    static boost::program_options::options_description get_args_options();

    SimpleBlockMatcher(const boost::program_options::variables_map &options);
    ~SimpleBlockMatcher();

    void set_rectified_images_pair(gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right);

    void compute_disparity_map(gil::gray8c_view_t &left, gil::gray8c_view_t &right, const bool left_right_are_inverted);

    void compute_disparity_map(gil::rgb8c_view_t  &left, gil::rgb8c_view_t &right, const bool left_right_are_inverted);

private:

    bool do_left_right_consistency_check;
    std::string pixels_matching_method;
    int implementation_version_to_use;

    float threshold_percent;

    /// Generic stereo matching using block matching
    template <typename ImgT> void compute_disparity_map_impl(ImgT &left, ImgT &right, const bool left_right_are_inverted);


    /// ImgT is expected to be bitsetN_view_t, gray8c_view_t or rgb8c_view_t
    template <typename ImgT, typename PixelsCostType> void compute_dispary_map_vN(const ImgT &left, const ImgT &right, const bool left_right_are_inverted, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map);

    template <typename ImgT, typename PixelsCostType> void compute_dispary_map_v0(const ImgT &left, const ImgT &right, const bool do_consistency_check, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map);
    template <typename ImgT, typename PixelsCostType> void compute_dispary_map_v1(const ImgT &left, const ImgT &right, const bool do_consistency_check, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map);
    template <typename ImgT, typename PixelsCostType> void compute_dispary_map_v2(const ImgT &left, const ImgT &right, const bool do_consistency_check, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map);

};


} // end of namespace doppia

#endif // SIMPLEBLOCKMATCHER_HPP
