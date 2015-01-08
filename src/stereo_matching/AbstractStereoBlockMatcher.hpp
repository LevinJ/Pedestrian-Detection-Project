#ifndef ABSTRACTSTEREOBLOCKMATCHER_HPP
#define ABSTRACTSTEREOBLOCKMATCHER_HPP

#include "AbstractStereoMatcher.hpp"

namespace doppia
{

using namespace boost::gil;

///  Simple intermediary abstract class that collects the common parameters
class AbstractStereoBlockMatcher : public AbstractStereoMatcher
{
protected:

    int window_width, window_height;

    bool do_dumb_left_right_consistency_check, interpolate_occluded_pixels;

    int left_right_disparity_tolerance;

public:

    static boost::program_options::options_description get_args_options();

    AbstractStereoBlockMatcher(const boost::program_options::variables_map &options);
    virtual ~AbstractStereoBlockMatcher();

    void compute_disparity_map();

    void compute_disparity_map(gray8c_view_t &left, gray8c_view_t &right);

    void compute_disparity_map(rgb8c_view_t  &left, rgb8c_view_t &right);

    virtual void compute_disparity_map(gray8c_view_t &left, gray8c_view_t &right, const bool left_right_are_inverted) = 0;

    virtual void compute_disparity_map(rgb8c_view_t  &left, rgb8c_view_t &right, const bool left_right_are_inverted) = 0;

protected:

    template<typename ImgView> void compute_disparity_map_impl();

    void check_left_right_consistency(const disparity_map_t::const_view_t &right_left_disparity, disparity_map_t::view_t &left_right_disparity) const;


    virtual void occlusions_interpolation(disparity_map_t::view_t &disparity) const;

    void interpolate_disparity(const disparity_map_t::view_t::x_iterator &start, const disparity_map_t::view_t::x_iterator &end,
                               int left_value, int right_value) const;

};


} // end of namespace doppia

#endif // ABSTRACTSTEREOBLOCKMATCHER_HPP
