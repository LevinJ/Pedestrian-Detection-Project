#include "AbstractStereoBlockMatcher.hpp"

#include "helpers/get_option_value.hpp"

namespace doppia
{

using namespace std;
using namespace boost;
using namespace boost::gil;

program_options::options_description AbstractStereoBlockMatcher::get_args_options()
{
    program_options::options_description desc("AbstractStereoBlockMatcher options");


    desc.add_options()

            ("window_size,w",
             program_options::value<int>()->default_value(9), "window size used during the correlation")

            ("left_right_consistency",
             program_options::value<bool>()->default_value(false), "will run the algorithm twice and check the left right consistency")

            ("left_right_tolerance",
             program_options::value<int>()->default_value(1), "maximum allowed difference between left and right disparity values")

            ("interpolate_occluded_pixels",
             program_options::value<bool>()->default_value(false), "will do a simple interpolation to estimate the disparity of the occluded pixels")

            ;

    return desc;
}


AbstractStereoBlockMatcher::AbstractStereoBlockMatcher(const program_options::variables_map &options)
    :  AbstractStereoMatcher(options)
{

    window_width = get_option_value<int>(options, "window_size");
    window_height = window_width;


    // window size has to be odd
    assert(window_width % 2 == 1);
    assert(window_height % 2 == 1);

    do_dumb_left_right_consistency_check = get_option_value<bool>(options, "left_right_consistency");
    left_right_disparity_tolerance = get_option_value<int>(options, "left_right_tolerance");

    // check exceptions ---
    {
        string stereo_method;
        if (options.count("method") != 0)
        {
            stereo_method = get_option_value<std::string>(options, "method");
        }
        else if (options.count("stereo.method") != 0)
        {
            stereo_method = get_option_value<std::string>(options, "stereo.method");
        }
	else if (options.count("cost_volume.method") != 0)
	  {
            stereo_method = get_option_value<std::string>(options, "cost_volume.method");
	  }
        else
        {
            throw std::invalid_argument("AbstractStereoBlockMatcher failed to find the stereo method");
        }

        if(stereo_method.find("simple_") == 0)
        {
            printf("Transfering left right consistency check to the algorithm implementation (not doing double calls)\n");
            // these methods implement the left right consistency check inside themselves
            do_dumb_left_right_consistency_check = false;
        }

    }


    interpolate_occluded_pixels = get_option_value<bool>(options, "interpolate_occluded_pixels");

    return;
}

AbstractStereoBlockMatcher::~AbstractStereoBlockMatcher()
{
}


void AbstractStereoBlockMatcher::compute_disparity_map()
{
    // if the disparity map was set, then we can guess that the left and right views are set too
    assert(disparity_map_p.get() != NULL);

    if ( left.current_type_is<gray8c_view_t>() )
    {
        compute_disparity_map_impl<gray8c_view_t>();
    }
    else if ( left.current_type_is<rgb8c_view_t>() )
    {
        compute_disparity_map_impl<rgb8c_view_t>();
    }
    else
    {
        throw std::runtime_error("Calling AbstractStereoMatcher::compute_disparity_map with unsuported images type");
    }

    first_disparity_map_computation = false;
    return;
}



template<typename ImgView>
void AbstractStereoBlockMatcher::compute_disparity_map_impl()
{

    if (do_dumb_left_right_consistency_check)
    {

        if (first_disparity_map_computation)
        {
            printf("Will call stereo matching two times for left right disparity consistency check\n");
        }


        // call right to left first
        {
            const bool left_and_right_are_inverted = true;
            compute_disparity_map(right._dynamic_cast<ImgView>(), left._dynamic_cast<ImgView>(), left_and_right_are_inverted);
        }

        // copy disparity map
        disparity_map_t right_left_disparity_map(disparity_map_view.dimensions());
        disparity_map_t::view_t right_left_disparity_map_view = view(right_left_disparity_map);

        boost::gil::copy_pixels(disparity_map_view, right_left_disparity_map_view);

        // call left to right
        {
            const bool left_and_right_are_inverted = false;
            compute_disparity_map(left._dynamic_cast<ImgView>(), right._dynamic_cast<ImgView>(), left_and_right_are_inverted);
        }

        // check consistency
        check_left_right_consistency(right_left_disparity_map_view, disparity_map_view);

        // final disparity map is aligned with the left image
    }
    else
    {

        const bool left_and_right_are_inverted = false;
        compute_disparity_map(left._dynamic_cast<ImgView>(), right._dynamic_cast<ImgView>(), left_and_right_are_inverted);

    }

    if (interpolate_occluded_pixels)
    {

        if (true)
        {
            printf("Interpolating occluded pixels");
        }

        occlusions_interpolation(disparity_map_view);
    }

    return;
}

void AbstractStereoBlockMatcher::compute_disparity_map(gil::gray8c_view_t &left, gil::gray8c_view_t &right)
{
    compute_disparity_map(left, right, false);
    return;
}

void AbstractStereoBlockMatcher::compute_disparity_map(gil::rgb8c_view_t  &left, gil::rgb8c_view_t &right)
{
    compute_disparity_map(left, right, false);
    return;
}



void AbstractStereoBlockMatcher::check_left_right_consistency(const disparity_map_t::const_view_t &right_left_disparity, disparity_map_t::view_t &left_right_disparity) const
{

    assert(left_right_disparity.dimensions() == right_left_disparity.dimensions());

    typedef gil::channel_type<disparity_map_t>::type disparity_map_channel_t;
    const int max_disparity_value = gil::channel_traits< disparity_map_channel_t >::max_value();

    typedef disparity_map_t::value_type pixel_t;

    const int width = left_right_disparity.width();

    const gil::bits8 disparity_tolerance = left_right_disparity_tolerance;

    // for every pixel in the images
    for (int y=0;y< left_right_disparity.height(); y+=1)
    {
        disparity_map_t::view_t::x_iterator lr_x_iterator = left_right_disparity.row_begin(y);
        disparity_map_t::const_view_t::x_iterator rl_x_iterator = right_left_disparity.row_begin(y);
        for (int left_x=0;left_x< width; left_x+=1, ++lr_x_iterator, ++rl_x_iterator)
        {
            pixel_t &lr_pixel = *lr_x_iterator;
            const int right_x = left_x - lr_pixel[0];
            if ((right_x >= 0) and (right_x < width))
            {

                // right to left disparity values are negative
                const pixel_t &rl_pixel = *(rl_x_iterator - lr_pixel[0]);

                if (false and y == 100 and left_x == 200)
                {
                    printf("lr_pixel[0] == %i; (*(rl_x_iterator - lr_pixel[0]))[0] == %i, (*(rl_x_iterator + lr_pixel[0]))[0] == %i\n",
                           lr_pixel[0],
                           rl_pixel[0],
                           (*(rl_x_iterator + lr_pixel[0]))[0]);
                }

                const gil::bits8 delta_disparity = std::abs(lr_pixel[0] - rl_pixel[0]);
                if ( delta_disparity > disparity_tolerance )
                {
                    lr_pixel[0] = max_disparity_value;
                }
                else
                {
                    // current disparity is consistent
                    lr_pixel[0] += delta_disparity / 2;
                    continue;
                }

            }
            else
            {
                lr_pixel[0] = max_disparity_value;
            }

        }
    }

    return;
}



/*
  Simple method suggested in  Hosni, Bleyer, Gelautz and Rhemann 2009
  */
void AbstractStereoBlockMatcher::occlusions_interpolation(disparity_map_t::view_t &disparity) const
{

    typedef gil::channel_type<disparity_map_t>::type disparity_map_channel_t;
    const int max_disparity_value = gil::channel_traits< disparity_map_channel_t >::max_value();

    // for each row
    for (int y=0;y< disparity.height(); y+=1)
    {
        disparity_map_t::view_t::x_iterator x_iterator = disparity.x_at(0,y);
        disparity_map_t::view_t::x_iterator segment_start_position = x_iterator;
        disparity_map_t::view_t::x_iterator segment_end_position = x_iterator;
        int left_side_disparity = max_disparity_value;
        int right_side_disparity = max_disparity_value;
        bool in_an_occluded_section = false;

        for (int x=0;x< disparity.width(); x+=1, ++x_iterator)
        {
            const int current_disparity_value = (*x_iterator)[0];

            if ((in_an_occluded_section == true) and (current_disparity_value != max_disparity_value))
            {
                in_an_occluded_section = false;
                right_side_disparity = current_disparity_value;
                segment_end_position = x_iterator;

                // found the end of an occlusion section, we interpolate it
                interpolate_disparity(segment_start_position, segment_end_position, left_side_disparity, right_side_disparity);
            }


            if ((in_an_occluded_section == false) and (current_disparity_value == max_disparity_value))
            { // start of a section to interpolate
                in_an_occluded_section = true;
                segment_start_position = x_iterator;
                segment_end_position = x_iterator;
            }

            if (in_an_occluded_section == false)
            {
                left_side_disparity = current_disparity_value;
            }

        }

        if (in_an_occluded_section)
        {
            // end of row was an occluded segment
            // found the end of an occlusion section, we interpolate it
            right_side_disparity = max_disparity_value;
            segment_end_position = x_iterator;
            interpolate_disparity(segment_start_position, segment_end_position, left_side_disparity, right_side_disparity);
        }

    }

    return;
}

void AbstractStereoBlockMatcher::interpolate_disparity(const disparity_map_t::view_t::x_iterator &start, const disparity_map_t::view_t::x_iterator &end,
                                                       int left_value, int right_value) const
{

    // could this be done using std::fill ?
    const int interpolation_value = std::min(left_value, right_value);
    disparity_map_t::view_t::x_iterator it;
    for (it = start; it != end; ++it)
    {
        *it = interpolation_value;
    }

    return;
}



} // end of namespace doppia
