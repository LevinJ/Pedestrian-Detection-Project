#include "SimpleBlockMatcher.hpp"

#include "helpers/get_option_value.hpp"

#include "cost_functions.hpp"

namespace doppia
{

using namespace std;
using namespace boost::gil;

program_options::options_description  SimpleBlockMatcher::get_args_options(void)
{
    program_options::options_description desc("SimpleBlockMatcher options");

    desc.add_options()

            ("pixels_matching",
             program_options::value<string>()->default_value("sad"),
             "pixels matching method: sad or ssd")

            ("threshold",
             program_options::value<float>()->default_value(0.5f),
             "minimum percent of pixels required to declare a match value between [0,1]")


            ("version,v",
             program_options::value<int>()->default_value(2),
             "implementation version to use. Currently versions [0,1,2] exist")
            ;

    return desc;
}

SimpleBlockMatcher::SimpleBlockMatcher(const program_options::variables_map &options)
    : AbstractStereoBlockMatcher(options)
{

    do_left_right_consistency_check = get_option_value<bool>(options, "left_right_consistency");

    if(options.count("method") > 0)
    {
        pixels_matching_method = get_option_value<std::string>(options, "method");
    }
    else if(options.count("stereo.method") > 0)
    {
        pixels_matching_method = get_option_value<std::string>(options, "stereo.method");
    }
    else if(options.count("cost_volume.method") > 0)
    {
        pixels_matching_method = get_option_value<std::string>(options, "cost_volume.method");
    }
    else
    {
        throw std::invalid_argument("SimpleBlockMatcher failed to deduce the pixels_matching method to use");
    }

    if(pixels_matching_method.find("simple_") == 0)
    {
        const int n = std::string("simple_").length();
        pixels_matching_method = pixels_matching_method.substr(n, pixels_matching_method.length() - n);
    }


    implementation_version_to_use = get_option_value<int>(options, "version");

    {
        threshold_percent = get_option_value<float>(options,"threshold");

        assert(threshold_percent > 0.0f);
        assert(threshold_percent <= 1.0f);
    }

    return;
}

SimpleBlockMatcher::~SimpleBlockMatcher()
{

    return;
}


void SimpleBlockMatcher::set_rectified_images_pair( gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right)
{
    this->AbstractStereoBlockMatcher::set_rectified_images_pair(left, right);

    // *.recreate does lazy allocation

    // resize lowest mismatch map
    mismatch_map.recreate(left.dimensions());
    mismatch_map_view = view(mismatch_map);

    // resize lowest mismatch map
    lowest_mismatch_map.recreate(left.dimensions());
    lowest_mismatch_view = view(lowest_mismatch_map);

    // resize left_to_right and right_to_left disparity maps
    left_to_right_disparity_map.recreate(left.dimensions());
    right_to_left_disparity_map.recreate(left.dimensions());

    left_to_right_disparity_map_view = view(left_to_right_disparity_map);
    right_to_left_disparity_map_view = view(right_to_left_disparity_map);

    fill_pixels(left_to_right_disparity_map_view, 0);
    fill_pixels(right_to_left_disparity_map_view, 0);

    return;
}



void  SimpleBlockMatcher::compute_disparity_map(gil::gray8c_view_t &left, gil::gray8c_view_t &right, const bool left_right_are_inverted)
{
    compute_disparity_map_impl(left, right, left_right_are_inverted);
    return;
}

void  SimpleBlockMatcher::compute_disparity_map(gil::rgb8c_view_t  &left, gil::rgb8c_view_t &right, const bool left_right_are_inverted)
{
    compute_disparity_map_impl(left, right, left_right_are_inverted);
    return;
}


template <typename ImgView>
void SimpleBlockMatcher::compute_disparity_map_impl( ImgView &left, ImgView &right, bool left_right_are_inverted)
{

    if (pixels_matching_method == "sad")
    {
        if(first_disparity_map_computation)
        {
            printf("SimpleBlockMatcher::compute_disparity_map will use sad matching\n\n");
        }
        SadCostFunction pixels_distance;
        compute_dispary_map_vN(left, right, left_right_are_inverted, pixels_distance, disparity_map_view );
    }
    else if (pixels_matching_method == "ssd")
    {
        if(first_disparity_map_computation)
        {
            printf("SimpleBlockMatcher::compute_disparity_map will use ssd matching\n\n");
        }
        SsdCostFunction pixels_distance;
        compute_dispary_map_vN(left, right, left_right_are_inverted, pixels_distance, disparity_map_view );
    }
    else if (pixels_matching_method == "lcdm")
    {
        if(first_disparity_map_computation)
        {
            printf("SimpleBlockMatcher::compute_disparity_map will use lcdm matching\n\n");
        }
        LCDMCostFunction pixels_distance;
        compute_dispary_map_vN(left, right, left_right_are_inverted, pixels_distance, disparity_map_view );
    }
    else
    {
        printf("SimpleBlockMatcher::compute_disparity_map received an unknow pixels_matching_method value == %s\n",
               pixels_matching_method.c_str());

        throw std::runtime_error("SimpleBlockMatcher::compute_disparity_map received an unknow pixels_matching_method");
    }

    return;
}


template <typename ImgView, typename PixelsCostType>
void SimpleBlockMatcher::compute_dispary_map_vN(const ImgView &left, const ImgView &right, const bool left_right_are_inverted, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map)
{

    if(left_right_are_inverted)
    {
        throw std::runtime_error("SimpleBlockMatcher::compute_dispary_map_vN should never receive left_right_are_inverted == true");
    }


    switch (implementation_version_to_use)
    {
    case 0:
        compute_dispary_map_v0(left, right, do_left_right_consistency_check, pixels_distance, disparity_map);
        break;
    case 1:
        compute_dispary_map_v1(left, right, do_left_right_consistency_check, pixels_distance, disparity_map);
        break;
    case 2:
    default:
        compute_dispary_map_v2(left, right, do_left_right_consistency_check, pixels_distance, disparity_map);
        break;
    }

    return;
}


template <typename ImgView, typename PixelsCostType>
void SimpleBlockMatcher::compute_dispary_map_v0(
    const ImgView &/*left*/, const ImgView &/*right*/,
    const bool /*do_consistency_check*/, PixelsCostType &/*pixels_distance*/, disparity_map_t::view_t &/*disparity_map*/)
{

    // here should be an easy to read, easy to understand stereo matching implementation
    // in the meantime, take a look at CensusStereoMatcher::compute_dispary_map_v1
    throw std::runtime_error("Not yet implemented");
    return;
}

template <typename ImgView, typename PixelsCostType>
void SimpleBlockMatcher::compute_dispary_map_v1(
    const ImgView &left, const ImgView &right,
    const bool do_consistency_check, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map)
{

    if(do_consistency_check)
    {
        throw std::runtime_error("SimpleBlockMatcher::compute_dispary_map_v1 with left right consistency check, is not yet implemented");
    }

    if (first_disparity_map_computation)
    {
        printf("Calling SimpleBlockMatcher::compute_dispary_map_v1, "\
               "with horizontal sliding window, per pixel mismatch cost temporal memory, image iterators and color support\n");
    }

    typedef typename ImgView::value_type pixel_t;

    const int pixels_per_block = (this->window_width*this->window_height);
    const float maximum_possible_mismatch =
            pixels_distance.template get_maximum_cost_per_pixel<pixel_t>() * pixels_per_block;
    const float maximum_allowed_mismatch = maximum_possible_mismatch*threshold_percent;

    typedef typename gil::channel_type<disparity_map_t>::type disparity_map_channel_t;
    const int max_disparity_value = gil::channel_traits< disparity_map_channel_t >::max_value();
    gil::fill_pixels(disparity_map, max_disparity_value);


    gil::fill_pixels(lowest_mismatch_view, maximum_allowed_mismatch);

    const int left_width = static_cast<int>(left.width());
    const int left_height = static_cast<int>(left.height());


#pragma omp parallel  default(shared)
    for (int disparity=0; disparity < max_disparity; disparity+=1)
    {
        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)

        const int first_left_x = disparity;

        // previous_window_column_index is little trick to initialize in the multithread environment
        int previous_window_column_index = -1;
        vector<float> previous_window_column_value(left_width);

#pragma omp for
        for (int left_y=0; left_y<left_height; left_y+=1)
        {

            // detect first call inside the for loop
            if(previous_window_column_index != left_y)
            {
                // precompute the pixels values distance of the first rows

                for (int t_left_y=left_y; t_left_y<(left_y + window_height - 1) && t_left_y < left_height; t_left_y+=1)
                {
                    const int first_right_x = first_left_x - disparity;
                    assert(first_right_x == 0);
                    typename ImgView::x_iterator
                            left_row_iterator = left.x_at(first_left_x, t_left_y), right_row_iterator = right.x_at(first_right_x, t_left_y);
                    mismatch_map_t::view_t::x_iterator
                            mismatch_map_row_iterator = mismatch_map_view.x_at(first_left_x, t_left_y);
                    for (int left_x =first_left_x;
                         left_x<left_width;
                         left_x +=1, ++left_row_iterator, ++right_row_iterator, ++mismatch_map_row_iterator)
                    {
                        //const int left_row = left_y, right_row = left_y;
                        //const int left_column = left_x, right_column = left_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        // const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_mismatch_value;
                        (*mismatch_map_row_iterator)[0] = t_mismatch_value;
                    }
                }
            }
            previous_window_column_index = left_y + 1;


            float previous_mismatch = maximum_allowed_mismatch;

            const int left_row_iterator_y = left_y + window_height - 1;
            const int right_row_iterator_y = left_row_iterator_y;
            const int first_right_x = first_left_x - disparity;
            assert(first_right_x == 0);

            if(left_row_iterator_y >= left_height || first_left_x < 0)
            {
                // skip out for range values
                continue;
            }


            typename ImgView::x_iterator
                    left_row_iterator = left.x_at(first_left_x, left_row_iterator_y),
                    right_row_iterator = right.x_at(first_right_x, right_row_iterator_y);

            { // the first window matching of each row is the exception, all the following ones are the rule

                float current_mismatch = 0.0f;

                // we aggregate over the columns
                for (int window_x = first_left_x;
                     (window_x < ((int) first_left_x + window_width)) && (window_x < left_width);
                     window_x+=1, ++left_row_iterator, ++right_row_iterator)
                {
                    mismatch_map_t::view_t::y_iterator
                            mismatch_map_column_iterator = mismatch_map_view.y_at(window_x, left_y);

                    float current_column_value = 0.0f;
                    for (int window_y = 0; window_y < (int) (window_height -1) && ((left_y + window_y) < left_height); window_y +=1, ++mismatch_map_column_iterator)
                    {
                        //const int left_row = left_y +  window_y;
                        //const int left_column = window_x;

                        // these values were already computed
                        //const float t_mismatch_value = (*mismatch_map_view.at(left_column, left_row))[0];
                        const float t_mismatch_value = (*mismatch_map_column_iterator)[0];
                        current_mismatch += t_mismatch_value;
                        current_column_value += t_mismatch_value;
                    }

                    // we compute the last pixel in the column
                    {
                        //const int window_y = window_height - 1;
                        //const int left_row = left_y +  window_y, right_row = left_y + window_y;
                        //const int left_column = window_x, right_column = window_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        // const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_distance;
                        (*mismatch_map_column_iterator)[0] = t_mismatch_value;
                        current_mismatch += t_mismatch_value;
                        current_column_value += t_mismatch_value;
                    }

                    //printf("previous_window_column_value[%i] == %.3f\n",window_x, current_column_value);
                    previous_window_column_value[window_x] = current_column_value;
                }

                previous_mismatch = current_mismatch;
            } // end of first window processing

            disparity_map_t::view_t::x_iterator
                    disparity_map_row_iterator = disparity_map.x_at(first_left_x, left_y);

            lowest_mismatch_map_t::view_t::x_iterator
                    lowest_mismatch_map_row_iterator = lowest_mismatch_view.x_at(first_left_x, left_y);

            for (int left_x=first_left_x; left_x<left_width; left_x+=1,
                 ++disparity_map_row_iterator, ++lowest_mismatch_map_row_iterator, ++left_row_iterator, ++right_row_iterator)
            {

                // compute current match ---

                // aggregate over the rightmost column
                const int column_x = left_x + (window_width - 1);
                float rightmost_column_value = 0.0f;

                if (column_x < left_width)
                {

                    mismatch_map_t::view_t::y_iterator
                            mismatch_map_col_iterator = mismatch_map_view.y_at(column_x, left_y);
                    for (int window_y = 0; window_y < (int) (window_height -1) && ((left_y + window_y) < left_height); window_y +=1, ++mismatch_map_col_iterator)
                    {
                        // these values where already computed

                        //const int left_row = left_y + window_y;
                        //const int left_column = column_x;
                        //const float t_mismatch_value = (*mismatch_map_view.at(left_column, left_row))[0];
                        const float t_mismatch_value = (*mismatch_map_col_iterator)[0];
                        rightmost_column_value += t_mismatch_value;
                    }

                    // compute the pixels values distance of the last pixel in the column
                    {
                        //const int window_y = window_height - 1;
                        //const int left_row = left_y + window_y, right_row = left_y + window_y;
                        //const int left_column = column_x, right_column = column_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        //const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_mismatch_value;
                        (*mismatch_map_col_iterator)[0] = t_mismatch_value;
                        rightmost_column_value += t_mismatch_value;
                    }

                    //printf("previous_window_column_value[%i] == %.3f\n",column_x, rightmost_column_value);
                    previous_window_column_value[column_x] = rightmost_column_value;
                }

                // retrieve left most value
                //printf("reading previous_window_column_value[%i]\n",left_x);
                const float leftmost_column_value = previous_window_column_value[left_x];

                // obtain mismatch
                const float current_mismatch = previous_mismatch - leftmost_column_value + rightmost_column_value;
                previous_mismatch = current_mismatch;

                // retrieve and update current best matches --

                //printf("disparity_map and lowest_mismatch_view at(%i, %i)\n", left_x, left_y);
                //gil::bits8 &disparity_with_lowest_mismatch = (*disparity_map.at(left_x, left_y))[0];
                //gil::bits32f &lowest_mismatch = (*lowest_mismatch_view.at(left_x, left_y))[0];
                gil::bits8 &disparity_with_lowest_mismatch = (*disparity_map_row_iterator)[0];
                gil::bits32f &lowest_mismatch = (*lowest_mismatch_map_row_iterator)[0];

                // update disparity_map and lowest_mismatch_view -
                // we prefer higher disparities (nearer estimation)
                if (current_mismatch <= lowest_mismatch)
                {
                    lowest_mismatch = current_mismatch;
                    disparity_with_lowest_mismatch = disparity;
                }
                else
                {
                    // we keep current values
                }

            } // end of "for each column"

        } // end of "for each row"


    } // end of "for each possible disparity value"


    return;
}



template <typename ImgView, typename PixelsCostType>
void SimpleBlockMatcher::compute_dispary_map_v2(
    const ImgView &left, const ImgView &right,
    const bool do_consistency_check, PixelsCostType &pixels_distance, disparity_map_t::view_t &disparity_map)
{

    if (first_disparity_map_computation)
    {
        printf("Calling SimpleBlockMatcher::compute_dispary_map_v2, "\
               "with horizontal sliding window, per pixel mismatch cost temporal memory, image iterators, color support and left right consistency check\n");
    }

    typedef typename ImgView::value_type pixel_t;

    const int pixels_per_block = (this->window_width*this->window_height);
    const float maximum_possible_mismatch =
            pixels_distance.template get_maximum_cost_per_pixel<pixel_t>() * pixels_per_block;
    const float maximum_allowed_mismatch = maximum_possible_mismatch*threshold_percent;

    typedef typename gil::channel_type<disparity_map_t>::type disparity_map_channel_t;
    const int max_disparity_value = gil::channel_traits< disparity_map_channel_t >::max_value();
    gil::fill_pixels(disparity_map, max_disparity_value);

    gil::fill_pixels(left_to_right_disparity_map_view, max_disparity_value);
    gil::fill_pixels(right_to_left_disparity_map_view, max_disparity_value);

    gil::fill_pixels(lowest_mismatch_view, maximum_allowed_mismatch);

    const int left_width = static_cast<int>(left.width());
    const int left_height = static_cast<int>(left.height());


#pragma omp parallel  default(shared)
    for (int disparity=0; disparity < max_disparity; disparity+=1)
    {
        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)

        const int first_left_x = disparity;

        // previous_window_column_index is little trick to initialize in the multithread environment
        int previous_window_column_index = -1;
        vector<float> previous_window_column_value(left_width);

#pragma omp for
        for (int left_y=0; left_y<left_height; left_y+=1)
        {

            // detect first call inside the for loop
            if(previous_window_column_index != left_y)
            {
                // precompute the pixels values distance of the first rows

                for (int t_left_y=left_y; t_left_y<(left_y + window_height - 1) && t_left_y < left_height; t_left_y+=1)
                {
                    const int first_right_x = first_left_x - disparity;
                    assert(first_right_x == 0);
                    typename ImgView::x_iterator
                            left_row_iterator = left.x_at(first_left_x, t_left_y), right_row_iterator = right.x_at(first_right_x, t_left_y);
                    mismatch_map_t::view_t::x_iterator
                            mismatch_map_row_iterator = mismatch_map_view.x_at(first_left_x, t_left_y);
                    for (int left_x =first_left_x;
                         left_x<left_width;
                         left_x +=1, ++left_row_iterator, ++right_row_iterator, ++mismatch_map_row_iterator)
                    {
                        //const int left_row = left_y, right_row = left_y;
                        //const int left_column = left_x, right_column = left_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        // const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_mismatch_value;
                        (*mismatch_map_row_iterator)[0] = t_mismatch_value;
                    }
                }
            }
            previous_window_column_index = left_y + 1;


            float previous_mismatch = maximum_allowed_mismatch;

            const int left_row_iterator_y = left_y + window_height - 1;
            const int right_row_iterator_y = left_row_iterator_y;
            const int first_right_x = first_left_x - disparity;
            assert(first_right_x == 0);

            if(left_row_iterator_y >= left_height || first_left_x < 0)
            {
                // skip out for range values
                continue;
            }


            typename ImgView::x_iterator
                    left_row_iterator = left.x_at(first_left_x, left_row_iterator_y),
                    right_row_iterator = right.x_at(first_right_x, right_row_iterator_y);

            { // the first window matching of each row is the exception, all the following ones are the rule

                float current_mismatch = 0.0f;

                // we aggregate over the columns
                for (int window_x = first_left_x;
                     (window_x < ((int) first_left_x + window_width)) && (window_x < left_width);
                     window_x+=1, ++left_row_iterator, ++right_row_iterator)
                {
                    mismatch_map_t::view_t::y_iterator
                            mismatch_map_column_iterator = mismatch_map_view.y_at(window_x, left_y);

                    float current_column_value = 0.0f;
                    for (int window_y = 0; window_y < (int) (window_height -1) && ((left_y + window_y) < left_height); window_y +=1, ++mismatch_map_column_iterator)
                    {
                        //const int left_row = left_y +  window_y;
                        //const int left_column = window_x;

                        // these values were already computed
                        //const float t_mismatch_value = (*mismatch_map_view.at(left_column, left_row))[0];
                        const float t_mismatch_value = (*mismatch_map_column_iterator)[0];
                        current_mismatch += t_mismatch_value;
                        current_column_value += t_mismatch_value;
                    }

                    // we compute the last pixel in the column
                    {
                        //const int window_y = window_height - 1;
                        //const int left_row = left_y +  window_y, right_row = left_y + window_y;
                        //const int left_column = window_x, right_column = window_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        // const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_distance;
                        (*mismatch_map_column_iterator)[0] = t_mismatch_value;
                        current_mismatch += t_mismatch_value;
                        current_column_value += t_mismatch_value;
                    }

                    //printf("previous_window_column_value[%i] == %.3f\n",window_x, current_column_value);
                    previous_window_column_value[window_x] = current_column_value;
                }

                previous_mismatch = current_mismatch;
            } // end of first window processing

            disparity_map_t::view_t::x_iterator
                    left_to_right_disparity_map_row_iterator = left_to_right_disparity_map_view.x_at(first_left_x, left_y),
                    right_to_left_disparity_map_row_iterator = right_to_left_disparity_map_view.x_at(first_left_x, left_y);

            lowest_mismatch_map_t::view_t::x_iterator
                    lowest_mismatch_map_row_iterator = lowest_mismatch_view.x_at(first_left_x, left_y);


            for (int left_x=first_left_x; left_x<left_width; left_x+=1,
                 ++left_to_right_disparity_map_row_iterator, ++right_to_left_disparity_map_row_iterator,
                 ++lowest_mismatch_map_row_iterator,
                 ++left_row_iterator, ++right_row_iterator)
            {

                // compute current match ---

                // aggregate over the rightmost column
                const int column_x = left_x + (window_width - 1);
                float rightmost_column_value = 0.0f;

                if (column_x < left_width)
                {

                    mismatch_map_t::view_t::y_iterator
                            mismatch_map_col_iterator = mismatch_map_view.y_at(column_x, left_y);
                    for (int window_y = 0; window_y < (int) (window_height -1) && ((left_y + window_y) < left_height); window_y +=1, ++mismatch_map_col_iterator)
                    {
                        // these values where already computed

                        //const int left_row = left_y + window_y;
                        //const int left_column = column_x;
                        //const float t_mismatch_value = (*mismatch_map_view.at(left_column, left_row))[0];
                        const float t_mismatch_value = (*mismatch_map_col_iterator)[0];
                        rightmost_column_value += t_mismatch_value;
                    }

                    // compute the pixels values distance of the last pixel in the column
                    {
                        //const int window_y = window_height - 1;
                        //const int left_row = left_y + window_y, right_row = left_y + window_y;
                        //const int left_column = column_x, right_column = column_x - disparity;

                        //printf("left.at(%i, %i), right.at(%i,%i)\n", left_column, left_row, right_column, right_row);
                        //const pixel_t &left_pixel_value = (*left.at(left_column, left_row))[0] ;
                        //const pixel_t &right_pixel_value = (*right.at(right_column, right_row))[0];

                        const pixel_t &left_pixel = (*left_row_iterator);
                        const pixel_t &right_pixel = (*right_row_iterator);
                        const float t_mismatch_value = pixels_distance(left_pixel, right_pixel);

                        //const float t_mismatch_value = pixels_distance(left_pixel_value, right_pixel_value);
                        //(*mismatch_map_view.at(left_column, left_row))[0] = t_mismatch_value;
                        (*mismatch_map_col_iterator)[0] = t_mismatch_value;
                        rightmost_column_value += t_mismatch_value;
                    }

                    //printf("previous_window_column_value[%i] == %.3f\n",column_x, rightmost_column_value);
                    previous_window_column_value[column_x] = rightmost_column_value;
                }

                // retrieve left most value
                //printf("reading previous_window_column_value[%i]\n",left_x);
                const float leftmost_column_value = previous_window_column_value[left_x];

                // obtain mismatch
                const float current_mismatch = previous_mismatch - leftmost_column_value + rightmost_column_value;
                previous_mismatch = current_mismatch;

                // retrieve and update current best matches --

                //printf("disparity_map and lowest_mismatch_view at(%i, %i)\n", left_x, left_y);
                gil::bits8 &left_to_right_disparity_with_lowest_mismatch = (*left_to_right_disparity_map_row_iterator)[0];
                gil::bits8 &right_to_left_disparity_with_lowest_mismatch = (*right_to_left_disparity_map_row_iterator)[0];
                gil::bits32f &lowest_mismatch = (*lowest_mismatch_map_row_iterator)[0];

                // update disparity_map and lowest_mismatch_view -

                if(do_consistency_check)
                {
                    if (current_mismatch == lowest_mismatch)
                    {
                        // with <= we keep the rightmost option
                        //lowest_mismatch = current_mismatch;
                        right_to_left_disparity_with_lowest_mismatch = disparity;
                    }
                    else if (current_mismatch < lowest_mismatch)
                    {
                        // with < we keep the leftmost option
                        lowest_mismatch = current_mismatch;
                        right_to_left_disparity_with_lowest_mismatch = disparity;
                        left_to_right_disparity_with_lowest_mismatch = disparity;
                    }
                    else
                    {
                        // we keep current values
                    }
                }
                else
                {
                    if (current_mismatch <= lowest_mismatch)
                    {
                        // with < we keep the leftmost option
                        lowest_mismatch = current_mismatch;
                        //right_to_left_disparity_with_lowest_mismatch = disparity;
                        left_to_right_disparity_with_lowest_mismatch = disparity;
                    }
                    else
                    {
                        // we keep current values
                    }
                }


            } // end of "for each column"

        } // end of "for each row"


    } // end of "for each possible disparity value"


    if(do_consistency_check)
    {
        // merge left and right disparity maps
        // left_to_right_disparity_map_view will be updated
        check_left_right_consistency(this->right_to_left_disparity_map_view, this->left_to_right_disparity_map_view);
    }

    // copy left_to_right_disparity_map_view into final disparity_map
    gil::copy_pixels(this->left_to_right_disparity_map_view, this->disparity_map_view);

    return;
}



} // end of namespace doppia

