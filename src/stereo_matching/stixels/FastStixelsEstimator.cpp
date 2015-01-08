#include "FastStixelsEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include "helpers/simd_intrisics_types.hpp"

#include <boost/gil/extension/io/png_io.hpp>

#include <algorithm>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "FastStixelsEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "FastStixelsEstimator");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "FastStixelsEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "FastStixelsEstimator");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;

FastStixelsEstimator::FastStixelsEstimator(
    const boost::program_options::variables_map &options,
    const MetricStereoCamera &camera,
    const float expected_object_height,
    const int minimum_object_height_in_pixels,
    const int stixel_width)
    : StixelsEstimator(options,
          camera,
          expected_object_height,
          minimum_object_height_in_pixels,
          stixel_width),
      disparity_offset(camera.get_calibration().get_disparity_offset_x()),
      num_disparities(get_option_value<int>(options, "max_disparity"))
{
    // nothing to do here
    return;
}

FastStixelsEstimator::~FastStixelsEstimator()
{
    // nothing to do here
    return;
}

void FastStixelsEstimator::set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right)
{
    assert( left.dimensions() == right.dimensions() );

    input_left_view = left;
    input_right_view = right;

    typedef input_image_const_view_t::point_t point_t;
    const point_t transposed_dimensions(left.height(), left.width());

    if(transposed_left_image_p == false)
    {
        transposed_left_image_p.reset(new AlignedImage(transposed_dimensions));
    }

    if(transposed_right_image_p == false)
    {
        transposed_right_image_p.reset(new AlignedImage(transposed_dimensions));
    }

    if(transposed_rectified_right_image_p == false)
    {
        transposed_rectified_right_image_p.reset(new AlignedImage(transposed_dimensions));
    }

    if(rectified_right_image_p == false)
    {
        rectified_right_image_p.reset(new AlignedImage(right.dimensions()));
    }

    assert(transposed_left_image_p->dimensions() == transposed_dimensions );
    assert(transposed_right_image_p->dimensions() == transposed_dimensions );
    assert(transposed_rectified_right_image_p->dimensions() == transposed_dimensions );
    assert(rectified_right_image_p->dimensions() == left.dimensions() );

    return;
}

/// Provide the best estimate available for the ground plane
void FastStixelsEstimator::set_ground_plane_estimate(const GroundPlane &ground_plane,
                                                     const GroundPlaneEstimator::line_t &v_disparity_ground_line)
{

    if(input_left_view.size() == 0)
    {
        throw std::runtime_error("Sorry, you need to call FastStixelsEstimator::set_rectified_images_pair "
                                 "before FastStixelsEstimator::set_ground_plane_estimate");
    }


    the_ground_plane = ground_plane;
    the_v_disparity_ground_line = v_disparity_ground_line;


    const int num_rows = input_left_view.height();
    set_v_disparity_line_bidirectional_maps(num_rows, num_disparities);
    set_v_given_disparity(num_rows, num_disparities);

    return;
}


void FastStixelsEstimator::compute()
{
    // copy the input data into the memory aligned data structures
    copy_pixels(gil::transposed_view(input_left_view), transposed_left_image_p->get_view());
    copy_pixels(gil::transposed_view(input_right_view), transposed_right_image_p->get_view());

    // create the disparity space image --
    // (using estimated ground plane)
    compute_disparity_space_cost();

    // find the optimal ground-obstacle boundary --
    // (using dynamic programming)
    compute_ground_obstacle_boundary();

    return;
}


void sum_object_cost_baseline(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    float &object_cost_float)
{
    uint32_t object_cost = 0;
    for(; left_it != left_end_it; ++left_it, ++right_it)
    {
        object_cost += sad_cost_uint16(*left_it, *right_it);
    }

    object_cost_float = object_cost / 3;
    return;
}


void sum_object_cost_simd(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    float &object_cost_float)
{

    uint32_t object_cost = 0;

    typedef AlignedImage::const_view_t::value_type pixel_t;

    const int size_in_bytes = (left_end_it - left_it) * sizeof(pixel_t);
    const int simd_size_in_bytes =  size_in_bytes - (size_in_bytes % sizeof(v16qi));
    const int simd_steps =  simd_size_in_bytes / sizeof(v16qi);

    const v16qi
            *left_v16qi_it = reinterpret_cast<const v16qi*>(left_it),
            *left_v16qi_end_it = left_v16qi_it + simd_steps,
            *right_v16qi_it = reinterpret_cast<const v16qi*>(right_it);

    // simd section --
    v2di sad;
    for(; left_v16qi_it != left_v16qi_end_it; ++left_v16qi_it, ++right_v16qi_it)
    {
        // _mm_loadu_si128*2 + _mm_sad_epu8
        // copy data from non-aligned memory to register
        const m128i left_v16qi = _mm_loadu_si128(&(left_v16qi_it->m));
        const m128i right_v16qi = _mm_loadu_si128(&(right_v16qi_it->m));
        sad.m = _mm_sad_epu8(left_v16qi, right_v16qi);
        object_cost += sad.v[0] + sad.v[2];
    }

    // non simd section at the end --
    {
        const uint8_t
                *left_uint8_end_it = reinterpret_cast<const uint8_t*>(left_end_it);

        // we continue from where we left
        const uint8_t
                *left_uint8_it = reinterpret_cast<const uint8_t*>(left_v16qi_it),
                *right_uint8_it =  reinterpret_cast<const uint8_t*>(right_v16qi_it);

        assert((left_uint8_end_it - left_uint8_it) >= 0);

        for(; left_uint8_it != left_uint8_end_it; ++left_uint8_it, ++right_uint8_it)
        {
            const int16_t delta = *left_uint8_it - *right_uint8_it;
            const uint8_t sad = abs(delta);
            object_cost += sad;
        } // end of "for each column"

        // we update left_it and right_it to point at where we left -
        left_it = left_end_it;
        right_it = reinterpret_cast<AlignedImage::const_view_t::x_iterator>(right_uint8_it);

    }

    // rescale the score --
    object_cost_float = object_cost / 3;

    return;
}


void FastStixelsEstimator::compute_object_cost(u_disparity_cost_t &object_cost) const
{
    //const int num_rows = input_left_view.height();
    const size_t num_columns = input_left_view.width();
    const size_t num_disparities = this->num_disparities;
    const int disparity_offset = this->disparity_offset;

    assert(static_cast<size_t>(object_cost.rows()) == num_disparities);
    assert(static_cast<size_t>(object_cost.cols()) == num_columns);

    //const bool use_simd = false;
    const bool use_simd = true;

    // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
    for(size_t u = 0; u < num_columns; u += 1)
    { // iterate over the columns

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_left_image_p->get_view().row_begin(u);
        //left_column_end_it = transposed_left_image_p->get_view().row_end(u);

        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
        //const int first_right_x = first_left_x - disparity;

        for(size_t d = 0; d < num_disparities; d += 1)
        {
            const int right_u = u - (d + disparity_offset);
            if( right_u < 0 )
            {
                // disparity is too large for the current column
                // cost left to zero
                continue;
            }

            const AlignedImage::const_view_t::x_iterator
                    right_column_begin_it = transposed_right_image_p->get_view().row_begin(right_u);

            // for each (u, disparity) value accumulate over the vertical axis --
            const int minimum_v_for_disparity = top_v_for_stixel_estimation_given_disparity[d];
            const size_t ground_obstacle_v_boundary = v_given_disparity[d];
            // precomputed_v_disparity_line already checked for >=0 and < num_rows

            const AlignedImage::const_view_t::x_iterator
                    left_begin_it = left_column_begin_it + minimum_v_for_disparity,
                    left_end_it = left_column_begin_it + ground_obstacle_v_boundary,
                    right_begin_it = right_column_begin_it + minimum_v_for_disparity;
            assert(left_begin_it <= left_end_it);

            AlignedImage::const_view_t::x_iterator
                    left_it = left_begin_it, right_it = right_begin_it;

            // from tentative ground upwards, over the object -
            float &t_object_cost = object_cost(d, u);
            //for(size_t v=minimum_v_for_disparity; v < ground_obstacle_v_boundary; v+=1)
            //{
            //    object_cost += rows_slice[v];
            //}
            if(use_simd)
            {
                sum_object_cost_simd(left_it, left_end_it, right_it, t_object_cost);
            }
            else
            {
                sum_object_cost_baseline(left_it, left_end_it, right_it, t_object_cost);
            }

            // normalize the object cost -
            t_object_cost /= (ground_obstacle_v_boundary - minimum_v_for_disparity);
            assert(t_object_cost >= 0);

        } // end of "for each disparity"
    } // end of "for each u"

    return;
}


void compute_transposed_rectified_right_image(
        const AbstractStixelsEstimator::input_image_const_view_t &input_right_view,
        const AlignedImage::view_t &rectified_right_view,
        const AlignedImage::view_t &transposed_rectified_right_view,
        const int disparity_offset,
        const std::vector<int> &v_given_disparity,
        const std::vector<int> &disparity_given_v)
{

    const int num_rows = input_right_view.height();
    const size_t num_columns = input_right_view.width();

    typedef AbstractStixelsEstimator::input_image_const_view_t input_image_const_view_t;

    const AbstractStixelsEstimator::input_image_const_view_t &input_view = input_right_view;
    const AlignedImage::view_t &rectified_view = rectified_right_view;
    const AlignedImage::view_t &transposed_rectified_view = transposed_rectified_right_view;

    // compute the rectified image --
    // it seems that the pixels are set to zero by default
    //gil::fill_pixels(rectified_view, AlignedImage::const_view_t::value_type(0, 0, 0));

    // we only care about rows at and below the horizon
    const int first_row = v_given_disparity[0];

    for(int row=first_row; row < num_rows; row +=1)
    {
        const int d_at_v = disparity_given_v[row] + disparity_offset;
        input_image_const_view_t::x_iterator input_it = input_view.row_begin(row);
        input_image_const_view_t::x_iterator input_end_it = input_view.row_end(row) - d_at_v;
        AlignedImage::view_t::x_iterator rectified_it = rectified_view.row_begin(row) + d_at_v;

        std::copy(input_it, input_end_it, rectified_it);
    }

    // copy the rectified image area to the transposed_rectified_image --
    {
        const AlignedImage::const_view_t rectified_subview =
                gil::subimage_view(rectified_view, 0, first_row, num_columns, num_rows - first_row);

        const AlignedImage::view_t transposed_rectified_subview =
                gil::subimage_view(transposed_rectified_view, first_row, 0, num_rows - first_row, num_columns);

        gil::copy_pixels(gil::transposed_view(rectified_subview),
                         transposed_rectified_subview);
    }

    return;
}


void FastStixelsEstimator::compute_transposed_rectified_right_image()
{
    doppia::compute_transposed_rectified_right_image(
            input_right_view,
            rectified_right_image_p->get_view(),
            transposed_rectified_right_image_p->get_view(),
            disparity_offset,
            v_given_disparity,
            disparity_given_v);

    const bool save_image = false;
    if(save_image)
    {
        log_debug() << "Created rectified_right_image.png" << std::endl;
        gil::png_write_view("rectified_right_image.png",
                            rectified_right_image_p->get_view());

        log_debug() << "Created transposed_rectified_right_image.png" << std::endl;
        gil::png_write_view("transposed_rectified_right_image.png",
                            transposed_rectified_right_image_p->get_view());

        throw std::runtime_error("FastStixelsEstimator::compute_transposed_rectified_right_image "
                                 "created the requested debug images (see log file)");
    }

    return;
}


inline
void sum_ground_cost_baseline(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    const std::vector<int> &disparity_given_v,
    const size_t u,
    const size_t ground_obstacle_v_boundary,
    const size_t num_rows,
    float &ground_cost_float)
{

    uint32_t ground_cost = 0;

    for(std::size_t v=ground_obstacle_v_boundary;
        v < num_rows and left_it != left_end_it;
        v+=1, ++left_it, ++right_it)
    {
        const int d_at_v = disparity_given_v[v];
        //ground_cost += rows_disparities_slice[v][d_at_v];

        const int right_u = u - d_at_v;
        if(true or right_u >= 0)
        {
            ground_cost += sad_cost_uint16(*left_it, *right_it);
        }
        else
        {
            // out of image
            continue;
        }
    } // end of "for each v value below the ground_obstacle_v_boundary"

    ground_cost_float = ground_cost / 3;
    return;
}


// sum_ground_cost_simd is quite similar to sum_object_cost_simd
inline
void sum_ground_cost_simd(
    AlignedImage::const_view_t::x_iterator &left_it,
    const AlignedImage::const_view_t::x_iterator &left_end_it,
    AlignedImage::const_view_t::x_iterator &right_it,
    float &ground_cost_float)
{
    uint32_t ground_cost = 0;

    typedef AlignedImage::const_view_t::value_type pixel_t;

    const int size_in_bytes = (left_end_it - left_it) * sizeof(pixel_t);
    const int simd_size_in_bytes =  size_in_bytes - (size_in_bytes % sizeof(v16qi));
    const int simd_steps =  simd_size_in_bytes / sizeof(v16qi);

    const v16qi
            *left_v16qi_it = reinterpret_cast<const v16qi*>(left_it),
            *left_v16qi_end_it = left_v16qi_it + simd_steps,
            *right_v16qi_it = reinterpret_cast<const v16qi*>(right_it);

    // simd section --
    v2di sad;
    for(; left_v16qi_it != left_v16qi_end_it; ++left_v16qi_it, ++right_v16qi_it)
    {
        // _mm_loadu_si128*2 + _mm_sad_epu8
        // copy data from non-aligned memory to register
        const m128i left_v16qi = _mm_loadu_si128(&(left_v16qi_it->m));
        const m128i right_v16qi = _mm_loadu_si128(&(right_v16qi_it->m));
        sad.m = _mm_sad_epu8(left_v16qi, right_v16qi);
        ground_cost += sad.v[0] + sad.v[2];
    }

    // non simd section at the end --
    {
        const uint8_t
                *left_uint8_end_it = reinterpret_cast<const uint8_t*>(left_end_it);

        // we continue from where we left
        const uint8_t
                *left_uint8_it = reinterpret_cast<const uint8_t*>(left_v16qi_it),
                *right_uint8_it =  reinterpret_cast<const uint8_t*>(right_v16qi_it);

        assert((left_uint8_end_it - left_uint8_it) >= 0);

        for(; left_uint8_it != left_uint8_end_it; ++left_uint8_it, ++right_uint8_it)
        {
            const int16_t delta = *left_uint8_it - *right_uint8_it;
            const uint8_t sad = abs(delta);
            ground_cost += sad;
        } // end of "for each column"

        // we update left_it and right_it to point at where we left -
        left_it = left_end_it;
        right_it = reinterpret_cast<AlignedImage::const_view_t::x_iterator>(right_uint8_it);

    }

    // rescale the score --
    ground_cost_float = ground_cost / 3;

    return;
}

inline
void FastStixelsEstimator::compute_ground_cost_v0(u_disparity_cost_t &ground_cost) const
{
    const size_t num_rows = input_left_view.height();
    const size_t num_columns = input_left_view.width();
    const size_t num_disparities = this->num_disparities;

    //const bool use_simd = false;
    const bool use_simd = true;

#pragma omp parallel for
    for(size_t u = 0; u < num_columns; u += 1)
    { // iterate over the columns

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_left_image_p->get_view().row_begin(u),
                left_column_end_it = transposed_left_image_p->get_view().row_end(u),
                right_column_begin_it = transposed_rectified_right_image_p->get_view().row_begin(u);

        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
        //const int first_right_x = first_left_x - disparity;

        for(size_t d = 0; d < num_disparities; d += 1)
        {
            if( u < d )
            {
                // disparity is too large for the current column
                // cost left to zero
                continue;
            }

            // for each (u, disparity) value accumulate over the vertical axis --
            const size_t ground_obstacle_v_boundary = v_given_disparity[d];
            // precomputed_v_disparity_line already checked for >=0 and < num_rows

            const AlignedImage::const_view_t::x_iterator
                    left_begin_it = left_column_begin_it + ground_obstacle_v_boundary,
                    left_end_it = left_column_end_it,
                    right_begin_it = right_column_begin_it + ground_obstacle_v_boundary;
            assert(left_begin_it <= left_end_it);

            AlignedImage::const_view_t::x_iterator
                    left_it = left_begin_it, right_it = right_begin_it;

            // from tentative ground downards, over the ground -
            float &t_ground_cost = ground_cost(d, u);

            if(use_simd)
            {
                sum_ground_cost_simd(left_it, left_end_it, right_it,
                                     //ground_obstacle_v_boundary, num_rows,
                                     t_ground_cost);
            }
            else
            {
                sum_ground_cost_baseline(left_it, left_end_it, right_it,
                                         disparity_given_v, u,
                                         ground_obstacle_v_boundary, num_rows,
                                         t_ground_cost);
            }

            // normalize the ground cost -
            if (ground_obstacle_v_boundary < num_rows)
            {
                t_ground_cost /= (num_rows - ground_obstacle_v_boundary);
            }

            assert(t_ground_cost >= 0);

        } // end of "for each disparity"
    } // end of "for each u"

    return;
}


/// v1 is faster than v0 (even without simd)
inline
void FastStixelsEstimator::compute_ground_cost_v1(u_disparity_cost_t &ground_cost) const
{
    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    const size_t num_disparities = this->num_disparities;
    const int disparity_offset = this->disparity_offset;

    // FIXME should be true everywhere but on rodrigob's laptop
    // (until bug found and fixed)
    // using false or true here seems not to make a relevant difference
    const bool use_simd = false;
    //const bool use_simd = true;

    // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
    for(int u = 0; u < num_columns; u += 1)
    { // iterate over the columns

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_left_image_p->get_view().row_begin(u),
                left_column_end_it = transposed_left_image_p->get_view().row_end(u),
                //right_column_begin_it = transposed_rectified_right_image_p->get_view().row_begin(u),
                right_column_end_it = transposed_rectified_right_image_p->get_view().row_end(u);

        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
        //const int first_right_x = first_left_x - disparity;

        //const size_t ground_obstacle_v_boundary_at_d_zero = v_given_disparity[0];

        // we are going to move in reverse from d = num_disparities - 1 to d = 0
        const AlignedImage::const_view_t::x_iterator
                left_begin_it = left_column_end_it - 1,
                //left_end_it = left_column_begin_it + ground_obstacle_v_boundary_at_d_zero,
                right_begin_it = right_column_end_it - 1;

        AlignedImage::const_view_t::x_iterator
                left_it = left_begin_it,
                right_it = right_begin_it;

        uint32_t column_cumulative_sum = 0;

        for(int d = num_disparities - 1; d >= 0; d -= 1)
        {
            //if(u < d )
            const int right_u = u - (d + disparity_offset);
            if(right_u < 0)
            {
                // disparity is too large for the current column
                // cost left to zero
                continue;
            }
            const size_t ground_obstacle_v_boundary_at_d = v_given_disparity[d];
            // precomputed_v_disparity_line already checked for >=0 and < num_rows

            assert(ground_obstacle_v_boundary_at_d < static_cast<size_t>(num_rows));

            // we cumulate the sum from the bottom of the image upwards
            const AlignedImage::const_view_t::x_iterator
                    left_step_end_it = left_column_begin_it + ground_obstacle_v_boundary_at_d;
            assert(left_step_end_it <= left_it);

            for(; left_it != left_step_end_it; --left_it, --right_it)
            {
                if(use_simd)
                {
                    // FIXME for some unknown reason enabling this code messes up the ground plane estimation
                    // it does compute the desired value, but it triggers a missbehaviour "somewhere else"
                    // very weird simd related bug... (that appears on my laptop but not on my desktop)

                    v8qi left_v8qi, right_v8qi;
                    left_v8qi.m = _mm_setzero_si64(); // = 0;
                    right_v8qi.m = _mm_setzero_si64(); // = 0;

                    left_v8qi.v[0] = (*left_it)[0];
                    left_v8qi.v[1] = (*left_it)[1];
                    left_v8qi.v[2] = (*left_it)[2];

                    right_v8qi.v[0] = (*right_it)[0];
                    right_v8qi.v[1] = (*right_it)[1];
                    right_v8qi.v[2] = (*right_it)[2];

                    v4hi sad_v4hi;
                    sad_v4hi.m = _mm_sad_pu8(left_v8qi.m, right_v8qi.m);
                    column_cumulative_sum += sad_v4hi.v[0];

                    /*const m64 left_m64 = _mm_set_pi8(0,0,0,0,0, (*left_it)[2], (*left_it)[1], (*left_it)[0]);
                    const m64 right_m64 = _mm_set_pi8(0,0,0,0,0, (*right_it)[2], (*right_it)[1], (*right_it)[0]);

                    const m64 sad_m64 = _mm_sad_pu8(left_m64, right_m64);
                    column_cumulative_sum += _mm_extract_pi16(sad_m64, 0);*/
                }
                else
                {
                    column_cumulative_sum += sad_cost_uint16(*left_it, *right_it);
                }
            }

            // normalize and set the ground cost -
            float &ground_cost_float = ground_cost(d, u);
            ground_cost_float = column_cumulative_sum;
            ground_cost_float /= 3*(num_rows - ground_obstacle_v_boundary_at_d); // 3 because of RGB

        } // end of "for each disparity"
    } // end of "for each u"

    return;
}



void FastStixelsEstimator::compute_object_and_ground_cost(u_disparity_cost_t &object_cost, u_disparity_cost_t &ground_cost) const
{
    const int num_rows = input_left_view.height();
    const size_t num_columns = input_left_view.width();
    const size_t num_disparities = this->num_disparities;
    const int disparity_offset = this->disparity_offset;

    //const bool use_simd = false;
    const bool use_simd = true;

    // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
    for(size_t u = 0; u < num_columns; u += 1)
    { // iterate over the columns

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_left_image_p->get_view().row_begin(u),
                left_column_end_it = transposed_left_image_p->get_view().row_end(u),
                rectified_right_column_begin_it = transposed_rectified_right_image_p->get_view().row_begin(u);

        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
        //const int first_right_x = first_left_x - disparity;

        for(size_t d = 0; d < num_disparities; d += 1)
        {
            const int right_u = u - (d + disparity_offset);
            if( right_u < 0 )
            {
                // disparity is too large for the current column
                // cost left to zero
                continue;
            }

            const AlignedImage::const_view_t::x_iterator
                    right_column_begin_it = transposed_right_image_p->get_view().row_begin(right_u);

            // for each (u, disparity) value accumulate over the vertical axis --
            const int minimum_v_for_disparity = top_v_for_stixel_estimation_given_disparity[d];
            const int ground_obstacle_v_boundary = v_given_disparity[d];
            // precomputed_v_disparity_line already checked for >=0 and < num_rows

            const AlignedImage::const_view_t::x_iterator
                    left_begin_it = left_column_begin_it + minimum_v_for_disparity,
                    left_end_it = left_column_begin_it + ground_obstacle_v_boundary,
                    right_begin_it = right_column_begin_it + minimum_v_for_disparity;
            assert(left_begin_it <= left_end_it);

            AlignedImage::const_view_t::x_iterator
                    left_it = left_begin_it, right_it = right_begin_it;

            // from tentative ground upwards, over the object -
            float &t_object_cost = object_cost(d, u);
            //for(size_t v=minimum_v_for_disparity; v < ground_obstacle_v_boundary; v+=1)
            //{
            //    object_cost += rows_slice[v];
            //}
            if(use_simd)
            {
                sum_object_cost_simd(left_it, left_end_it, right_it, t_object_cost);
            }
            else
            {
                sum_object_cost_baseline(left_it, left_end_it, right_it, t_object_cost);
            }

            // normalize the object cost -
            t_object_cost /= (ground_obstacle_v_boundary - minimum_v_for_disparity);
            assert(t_object_cost >= 0);
            // end of "compute object cost"

            // compute ground cost --
            {
                const AlignedImage::const_view_t::x_iterator
                        //left_begin_it = left_column_begin_it + ground_obstacle_v_boundary,
                        left_end_it = left_column_end_it,
                        right_begin_it = rectified_right_column_begin_it + ground_obstacle_v_boundary;
                assert(left_begin_it <= left_end_it);

                //left_it = left_begin_it; left_begin_it == left_it at this point
                right_it = right_begin_it;

                // from tentative ground downards, over the ground -
                float &t_ground_cost = ground_cost(d, u);

                if(use_simd)
                {
                    sum_ground_cost_simd(left_it, left_end_it, right_it,
                                         //ground_obstacle_v_boundary, num_rows,
                                         t_ground_cost);
                }
                else
                {
                    sum_ground_cost_baseline(left_it, left_end_it, right_it,
                                             disparity_given_v, u,
                                             ground_obstacle_v_boundary, num_rows,
                                             t_ground_cost);
                }

                // normalize the ground cost -
                if (ground_obstacle_v_boundary < num_rows)
                {
                    t_ground_cost /= (num_rows - ground_obstacle_v_boundary);
                }

                assert(t_ground_cost >= 0);
            } // end of "compute ground cost"


        } // end of "for each disparity"
    } // end of "for each u"


    return;
}


void FastStixelsEstimator::compute_disparity_space_cost()
{

    // Here (u,v) refers to the 2d image plane, just like (x,y) or (cols, rows)
    const size_t num_rows = input_left_view.height();
    const size_t num_columns = input_left_view.width();
    const size_t num_disparities = this->num_disparities;

    if(  v_given_disparity.size() != num_disparities or
            disparity_given_v.size() != num_rows)
    {
        throw std::runtime_error("FastStixelsEstimator::compute_disparity_space_cost "
                                 "called before FastStixelsEstimator::set_v_disparity_line_bidirectional_maps");
    }

    // reset and resize the object_cost and ground_cost
    // Eigen::MatrixXf::Zero(rows, cols)
    object_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);
    ground_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);

    compute_transposed_rectified_right_image();

    const bool two_in_one = false;
    //const bool two_in_one = true;
    if(two_in_one)
    {
        compute_object_and_ground_cost(object_u_disparity_cost, ground_u_disparity_cost);
    }
    else
    {
        // it seems that computing one cost and then the next one is slightly faster
        // than computing both in the same time (probably because of cache streaming usage)
        compute_object_cost(object_u_disparity_cost);
        //compute_ground_cost_v0(ground_u_disparity_cost);
        compute_ground_cost_v1(ground_u_disparity_cost);
    }


    // post filtering steps --
    {
        post_process_object_u_disparity_cost(object_u_disparity_cost);
        post_process_ground_u_disparity_cost(ground_u_disparity_cost, num_rows);
    }

    // set the final cost --
    u_disparity_cost = object_u_disparity_cost + ground_u_disparity_cost;

    // mini fix to the "left area initialization issue"
    fix_u_disparity_cost();

    return;
}

} // end of namespace doppia
