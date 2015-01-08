#include "FastStixelsEstimatorWithHeightEstimation.hpp"

#include "StixelsEstimatorWithHeightEstimation.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "video_input/MetricStereoCamera.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include "helpers/simd_intrisics_types.hpp"

#include <boost/cstdint.hpp>

#include <boost/gil/extension/io/png_io.hpp>

// include SSE2 intrinsics
#include <emmintrin.h>

#include <omp.h>

#include <cstring> // for memcpy


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "FastStixelsEstimatorWithHeightEstimation");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "FastStixelsEstimatorWithHeightEstimation");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "FastStixelsEstimatorWithHeightEstimation");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "FastStixelsEstimatorWithHeightEstimation");
}

} // end of anonymous namespace

namespace doppia {

using namespace std;
using namespace boost;

// FIXME where should I put this ?
/// min_float_disparity will be used to replace integer disparity == 0
const float min_float_disparity = 0.8f;


FastStixelsEstimatorWithHeightEstimation::FastStixelsEstimatorWithHeightEstimation(
    const boost::program_options::variables_map &options,
    const MetricStereoCamera &camera,
    const float expected_object_height,
    const int minimum_object_height_in_pixels,
    const int stixel_width)
    : FastStixelsEstimator(options,
          camera,
          expected_object_height,
          minimum_object_height_in_pixels,
          stixel_width)
{

    minimum_disparity_margin = get_option_value<int>(options, "stixel_world.max_disparity_margin");
    maximum_disparity_margin = get_option_value<int>(options, "stixel_world.min_disparity_margin");

    if(minimum_disparity_margin >=0 or maximum_disparity_margin >= 0)
    {
        log_info() << "Will use partial disparities information compute the stixels height" << std::endl;
    }
    else
    {
        log_info() << "Will use full disparity range to compute the stixels height" << std::endl;
    }

    // we set disparity_range -
    {
        if(maximum_disparity_margin >= 0 and minimum_disparity_margin >= 0)
        {
            disparity_range = maximum_disparity_margin + minimum_disparity_margin;
        }
        else if(minimum_disparity_margin >= 0)
        {
            disparity_range = num_disparities - minimum_disparity_margin;
        }
        else if(maximum_disparity_margin >= 0)
        {
            disparity_range = maximum_disparity_margin - 0;
        }
        else
        {
            disparity_range = num_disparities;
        }
    }

    stixels_height_post_processing_p.reset(new StixelsHeightPostProcessing(options, camera,
                                                                           expected_object_height,
                                                                           minimum_object_height_in_pixels,
                                                                           stixel_width));

    return;
}

FastStixelsEstimatorWithHeightEstimation::~FastStixelsEstimatorWithHeightEstimation()
{
    // nothing to do here
    return;
}



void FastStixelsEstimatorWithHeightEstimation::compute()
{
    static int num_iterations = 0;
    static double cumulated_time = 0;
    static double cumulated_time_for_height_only = 0;
    static double cumulated_time_for_distance_only = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();

    // compute the stixels distances --
    FastStixelsEstimator::compute();
    cumulated_time_for_distance_only += omp_get_wtime() - start_wall_time;

    const double start_wall_time_for_height = omp_get_wtime();
    {
        // find the optimal stixels height --
        // (using dynamic programming)
        compute_stixel_height_cost(); // compute the cost
        compute_stixels_heights(); // do dynamic programming
    }

    cumulated_time += omp_get_wtime() - start_wall_time;
    cumulated_time_for_height_only += omp_get_wtime() - start_wall_time_for_height;
    num_iterations += 1;

    if((num_iterations % num_iterations_for_timing) == 0)
    {
        printf("Average FastStixelsEstimatorWithHeightEstimation::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
        printf("Average stixels height estimation speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time_for_height_only, num_iterations_for_timing );
        printf("Average stixels distance estimation speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time_for_distance_only, num_iterations_for_timing );
        cumulated_time = 0;
        cumulated_time_for_height_only = 0;
        cumulated_time_for_distance_only = 0;
    }

    return;
}

const stixel_height_cost_t &
FastStixelsEstimatorWithHeightEstimation::get_stixel_height_cost() const
{
    return stixel_height_cost;
}


const stixel_height_cost_t &
FastStixelsEstimatorWithHeightEstimation::get_disparity_likelihood_map() const
{
    return disparity_likelihood_map;
}


/// disparity_likelihood_map is expected to be in the range [0, 1]
/// 1 indicates "stixel depth is good", 0 indicates "stixel depth is bad"
void disparity_likelihood_map_to_stixel_height_cost_v0(
    const Eigen::MatrixXf &disparity_likelihood_map,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    Eigen::MatrixXf &stixel_height_cost)
{
    const int num_rows = disparity_likelihood_map.rows();
    const int num_columns = disparity_likelihood_map.cols();

    // compute edge costs --
    //stixel_height_cost = Eigen::MatrixXf::Zero(num_rows, num_columns);
    stixel_height_cost = Eigen::MatrixXf::Constant(num_rows, num_columns, num_rows);

#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];

        const Eigen::VectorXf disparity_likelihood_map_col = disparity_likelihood_map.col(col).array() - 0.5;

        stixel_height_cost(0, col) = 1; // set upper line to max value
        for(int row=1; row < (bottom_v - 1); row += 1)
        {
            const float upper_part_sum =
                    (disparity_likelihood_map_col.segment(0, row).array() + 0.5).abs().sum();
            const float upper_cost = upper_part_sum;

            const float lower_part_sum =
                    (disparity_likelihood_map_col.segment(row, bottom_v - row).array() - 0.5).abs().sum();
            const float lower_cost = lower_part_sum;

            const float total_cost = (lower_cost + upper_cost)*2; // /bottom_v;
            assert(total_cost >= 0);
            stixel_height_cost(row, col) = total_cost;
        } // end of "for each row between 0 and bottom v"
    } // end of "for each column"

    return;
}

/// disparity_likelihood_map is expected to be in the range [0, 1]
/// 1 indicates "stixel depth is good", 0 indicates "stixel depth is bad"
void disparity_likelihood_map_to_stixel_height_cost_v1(
    const Eigen::MatrixXf &disparity_likelihood_map,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    Eigen::MatrixXf &stixel_height_cost)
{
    const int num_rows = disparity_likelihood_map.rows();
    const int num_columns = disparity_likelihood_map.cols();

    // compute edge costs --
    //stixel_height_cost = Eigen::MatrixXf::Zero(num_rows, num_columns);
    stixel_height_cost = Eigen::MatrixXf::Constant(num_rows, num_columns, num_rows);

#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];

        //const Eigen::VectorXf disparity_likelihood_map_col = disparity_likelihood_map.col(col).array() - 0.5;

        const Eigen::VectorXf abs_disparity_likelihood_map_col_plus_half =
                (disparity_likelihood_map.col(col).segment(0, bottom_v).array()
                 + (- 0.5 + 0.5)).abs(); // -0.5 + 0.5

        const Eigen::VectorXf abs_disparity_likelihood_map_col_minus_half =
                (disparity_likelihood_map.col(col).segment(0, bottom_v).array()
                 + (- 0.5 - 0.5)).abs(); // -0.5 -0.5

        stixel_height_cost(0, col) = 1; // set upper line to max value
        float upper_part_sum = abs_disparity_likelihood_map_col_plus_half(0);
        float lower_part_sum = abs_disparity_likelihood_map_col_minus_half.segment(0, bottom_v).sum();

        for(int row=1; row < (bottom_v - 1); row += 1)
        {
            upper_part_sum += abs_disparity_likelihood_map_col_plus_half(row);
            const float upper_cost = upper_part_sum;

            lower_part_sum -= abs_disparity_likelihood_map_col_minus_half(row);
            const float lower_cost = lower_part_sum;

            const float total_cost = (lower_cost + upper_cost)*2; // /bottom_v;
            assert(total_cost >= 0);
            stixel_height_cost(row, col) = total_cost;
        } // end of "for each row between 0 and bottom v"
    } // end of "for each column"

    return;
}

/// disparity_likelihood_map is expected to be in the range [0, 1]
/// 1 indicates "stixel depth is good", 0 indicates "stixel depth is bad"
/// (v2 seems slower than v1) (was a bad idea since Eigen defaults to column-major matrices)
void disparity_likelihood_map_to_stixel_height_cost_v2(
    const Eigen::MatrixXf &disparity_likelihood_map,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    Eigen::MatrixXf &stixel_height_cost)
{
    const int num_rows = disparity_likelihood_map.rows();
    const int num_columns = disparity_likelihood_map.cols();

    // compute edge costs --
    //stixel_height_cost = Eigen::MatrixXf::Zero(num_rows, num_columns);
    stixel_height_cost = Eigen::MatrixXf::Constant(num_rows, num_columns, num_rows);

    const Eigen::MatrixXf transposed_disparity_likelihood_map = disparity_likelihood_map.transpose();

#pragma omp parallel for
    for(int col = 0; col < num_columns; col += 1)
    {
        const int bottom_v = u_v_ground_obstacle_boundary[col];

        //const Eigen::VectorXf disparity_likelihood_map_col = disparity_likelihood_map.col(col).array() - 0.5;

        const Eigen::VectorXf abs_disparity_likelihood_map_col_plus_half =
                (transposed_disparity_likelihood_map.row(col).segment(0, bottom_v).array()
                 + (- 0.5 + 0.5)).abs(); // -0.5 + 0.5

        const Eigen::VectorXf abs_disparity_likelihood_map_col_minus_half =
                (transposed_disparity_likelihood_map.row(col).segment(0, bottom_v).array()
                 + (- 0.5 - 0.5)).abs(); // -0.5 -0.5

        stixel_height_cost(0, col) = 1; // set upper line to max value
        float upper_part_sum = abs_disparity_likelihood_map_col_plus_half(0);
        float lower_part_sum = abs_disparity_likelihood_map_col_minus_half.sum();

        for(int row=1; row < (bottom_v - 1); row += 1)
        {
            upper_part_sum += abs_disparity_likelihood_map_col_plus_half(row);
            const float upper_cost = upper_part_sum;

            lower_part_sum -= abs_disparity_likelihood_map_col_minus_half(row);
            const float lower_cost = lower_part_sum;

            const float total_cost = (lower_cost + upper_cost)*2; // /bottom_v;
            assert(total_cost >= 0);
            stixel_height_cost(row, col) = total_cost;
        } // end of "for each row between 0 and bottom v"
    } // end of "for each column"

    return;
}


/// Compute the heights cost matrix, without using any depth map
void FastStixelsEstimatorWithHeightEstimation::compute_stixel_height_cost()
{

    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;

    // compute a depth likelihood score along the stixel ---
    disparity_likelihood_map = Eigen::MatrixXf::Zero(num_rows, num_columns);

    // at each pixel will compute a value proportional to the likelihood
    // of the pixel having a true disparity equal to the stixel disparity map

    //compute_disparity_likelihood_map_v0(disparity_likelihood_map);
    //compute_disparity_likelihood_map_v1(disparity_likelihood_map);
    //compute_disparity_likelihood_map_v2(disparity_likelihood_map);
    compute_disparity_likelihood_map_v3(disparity_likelihood_map);

    //return; // FIXME just for debugging

    // compute stixel height cost based on disparity likelihood ---
    //disparity_likelihood_map_to_stixel_height_cost_v0(
    //            disparity_likelihood_map, u_v_ground_obstacle_boundary, stixel_height_cost);
    disparity_likelihood_map_to_stixel_height_cost_v1(
                disparity_likelihood_map, u_v_ground_obstacle_boundary, stixel_height_cost);
    //disparity_likelihood_map_to_stixel_height_cost_v2(
    //            disparity_likelihood_map, u_v_ground_obstacle_boundary, stixel_height_cost);


    // small post-processing
    // FIXME hardcoded value
    const int smoothing_iterations = 0; //0; 3; // smoothing seems to make almost no difference (improvement < 1%)
    //const int smoothing_iterations = 3; //0; 3;
    apply_horizontal_smoothing(stixel_height_cost, smoothing_iterations);

    return;
} // end of FastStixelsEstimatorWithHeightEstimation::compute_stixel_height_cost


inline
void FastStixelsEstimatorWithHeightEstimation::compute_horizontal_5px_sad_border_case(
    const int column, const int num_columns,
    const int object_bottom_v,
    const int disparity, vector<uint16_t> &disparity_cost)
{
    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;
    const int left_start = column - 2;
    const int left_end = column + 2;

    // ptrdiff_t, size_t
    const int right_start = left_start - disparity;
    const int right_end = left_end - disparity;

    const bool no_pixel_in_image = right_end < 0;
    if(no_pixel_in_image)
    {
        // we assume disparity_cost was already set to zero
        return;
    }


    // for each pixel above the ground, in the column
    // compute the 5x1 horizontal SAD
    vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();
    for(int v=0; v < object_bottom_v; v+=1, ++disparity_cost_it)
    {
        const input_image_const_view_t::x_iterator
                left_row_begin_it = input_left_view.row_begin(v),
                left_row_end_it = input_left_view.row_end(v),
                left_begin_it = left_row_begin_it + left_start,
                left_end_it = left_row_begin_it + left_end,
                right_row_begin_it = input_right_view.row_begin(v),
                right_begin_it = right_row_begin_it + right_start;

        input_image_const_view_t::x_iterator
                left_it = left_begin_it,
                right_it = right_begin_it;

        uint16_t &cost = *disparity_cost_it;
        for(; left_it != left_end_it and left_it != left_row_end_it; ++left_it, ++right_it)
        {
            if(right_it < right_row_begin_it
                    or left_it < left_row_begin_it)
            {
                // out of range
                continue;
            }

            cost += sad_cost_uint16(*left_it, *right_it);
        } // end of "for each of the 5 pair of pixels"


    } // end of "for each pixel above the ground"

    return;
}

inline
void FastStixelsEstimatorWithHeightEstimation::compute_horizontal_5px_sad_baseline(const int column,
                                                                                   const int object_bottom_v,
                                                                                   const int disparity,
                                                                                   vector<uint16_t> &disparity_cost)
{
    // we assume that all pixels are inside the image region
    const int left_start = column - 2;
    const int left_end = column + 2;
    const int right_start = left_start - disparity;

    const size_t row_step = input_left_view.row_begin(1) - input_left_view.row_begin(0);

    vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();
    input_image_const_view_t::x_iterator
            left_begin_it = input_left_view.row_begin(0) + left_start,
            left_end_it = input_left_view.row_begin(0) + left_end,
            right_begin_it = input_right_view.row_begin(0) + right_start;

    for(int v=0; v < object_bottom_v;
        v+=1,
        ++disparity_cost_it,
        left_begin_it+=row_step,
        left_end_it+=row_step,
        right_begin_it+=row_step)
    {
        input_image_const_view_t::x_iterator
                left_it = left_begin_it,
                right_it = right_begin_it;

        uint16_t &cost = *disparity_cost_it;
        for(; left_it != left_end_it; ++left_it, ++right_it)
        {
            cost += sad_cost_uint16(*left_it, *right_it);
        }
    } // end of "for each pixel above the ground"

    return;
}


// compute the raw SAD (without doing a division by the number of channels)
// this is a very suboptimal used of SIMD capabilities
// FIXME is this even faster than sad_cost_uint16 ?
inline uint16_t sad_cost_uint16_simd(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b)
{
    // FIXME for some unknown reason enabling this code messes up the ground plane estimation
    // it does compute the desired value, but it triggers a missbehaviour "somewhere else"
    // very weird simd related bug... (that appears on my laptop but not on my desktop)

    v8qi left_v8qi, right_v8qi;
    left_v8qi.m = _mm_setzero_si64(); // = 0;
    right_v8qi.m = _mm_setzero_si64(); // = 0;

    left_v8qi.v[0] = pixel_a[0];
    left_v8qi.v[1] = pixel_a[1];
    left_v8qi.v[2] = pixel_a[2];

    right_v8qi.v[0] = pixel_b[0];
    right_v8qi.v[1] = pixel_b[1];
    right_v8qi.v[2] = pixel_b[2];

    v4hi sad_v4hi;
    sad_v4hi.m = _mm_sad_pu8(left_v8qi.m, right_v8qi.m);
    return sad_v4hi.v[0];

    //const m64 left_m64 = _mm_set_pi8(0,0,0,0,0, (*left_it)[2], (*left_it)[1], (*left_it)[0]);
    //const m64 right_m64 = _mm_set_pi8(0,0,0,0,0, (*right_it)[2], (*right_it)[1], (*right_it)[0]);
    //const m64 sad_m64 = _mm_sad_pu8(left_m64, right_m64);
    //return _mm_extract_pi16(sad_m64, 0);
}


inline
void FastStixelsEstimatorWithHeightEstimation::compute_transposed_horizontal_5px_sad_baseline(
    const int column,
    const int object_bottom_v,
    const int disparity,
    vector<uint16_t> &disparity_cost)
{
    assert((transposed_left_image_p.get() != NULL) and (transposed_left_image_p->empty() == false));
    assert((transposed_right_image_p.get() != NULL) and (transposed_right_image_p->empty() == false));

    // in this function using simd == false is better
    const bool use_simd = false;
    //const bool use_simd = true;

    const AlignedImage::const_view_t
            &transposed_left_view = transposed_left_image_p->get_view(),
            &transposed_right_view = transposed_right_image_p->get_view();

    // we assume that all pixels are inside the image region
    const int left_start_column = column - 2;
    //const int left_end = column + 2;
    const int right_start_column = left_start_column - disparity;

    for(int i=0; i < 5; i+=1)
    {
        vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();

        const AlignedImage::const_view_t::x_iterator
                left_begin_it = transposed_left_view.row_begin(left_start_column + i),
                left_end_it = left_begin_it + object_bottom_v,
                right_begin_it = transposed_right_view.row_begin(right_start_column + i);

        AlignedImage::const_view_t::x_iterator
                left_it = left_begin_it,
                right_it = right_begin_it;

        for(; left_it != left_end_it;
            ++disparity_cost_it, ++left_it, ++right_it)
        {
            uint16_t &cost = *disparity_cost_it;

            if(use_simd)
            {
                cost += sad_cost_uint16_simd(*left_it, *right_it);
            }
            else
            {
                cost += sad_cost_uint16(*left_it, *right_it);
            }

        } // end of "for each pixel above the ground"

    } // end of "for each of the 5 columns"

    return;
}


inline
void FastStixelsEstimatorWithHeightEstimation::compute_horizontal_5px_sad_simd(const int column,
                                                                               const int object_bottom_v,
                                                                               const int disparity,
                                                                               vector<uint16_t> &disparity_cost)
{


    //return; // FIXME just for debugging

    // we assume that all pixels are inside the image region
    const int left_start = column - 2;
    //const int left_end = column + 2;
    const int right_start = left_start - disparity;

    const size_t row_step = input_left_view.row_begin(1) - input_left_view.row_begin(0);

    vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();
    input_image_const_view_t::x_iterator
            left_begin_it = input_left_view.row_begin(0) + left_start,
            //left_end_it = input_left_view.row_begin(0) + left_end,
            right_begin_it = input_right_view.row_begin(0) + right_start;

    for(int v=0; v < object_bottom_v;
        v+=1,
        ++disparity_cost_it,
        left_begin_it+=row_step,
        //left_end_it+=row_step,
        right_begin_it+=row_step)
    {
        // we consider 5*3 bytes == 15, so we only need to discard the last byte
        const v16qi
                * const left_v16qi_it = reinterpret_cast<const v16qi*>(left_begin_it),
                * const right_v16qi_it = reinterpret_cast<const v16qi*>(right_begin_it);

        uint16_t &cost = *disparity_cost_it;

        // simd section --

        // _mm_loadu_si128*2 + erase last byte + _mm_sad_epu8
        // copy data from non-aligned memory to register

        // const bool clean_last_byte = true; // FIXME just for debugging
        const bool clean_last_byte = false;

        if(clean_last_byte)
        {
            v16qi left_v16qi, right_v16qi;
            left_v16qi.m = _mm_loadu_si128(&(left_v16qi_it->m));
            right_v16qi.m = _mm_loadu_si128(&(right_v16qi_it->m));
            left_v16qi.v[7] = 0; right_v16qi.v[7] = 0; // 7 or 0 ?

            v2di sad;
            sad.m = _mm_sad_epu8(left_v16qi.m, right_v16qi.m);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16
        }
        else
        {
            const m128i left_m128i = _mm_loadu_si128(&(left_v16qi_it->m));
            const m128i right_m128i = _mm_loadu_si128(&(right_v16qi_it->m));
            v2di sad;
            sad.m = _mm_sad_epu8(left_m128i, right_m128i);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16
        }

    } // end of "for each pixel above the ground"

    return;
}


inline
void FastStixelsEstimatorWithHeightEstimation::compute_horizontal_5px_sad_simd_and_copy_data(
    const int column,
    const int object_bottom_v,
    const int disparity,
    AlignedImage &left_data,
    vector<uint16_t> &disparity_cost)
{

    // we assume that all pixels are inside the image region
    const ptrdiff_t left_start = column - 2;
    //const ptrdiff_t left_end = column + 2;
    const ptrdiff_t right_start = left_start - disparity;

    const size_t row_step = input_left_view.row_begin(1) - input_left_view.row_begin(0);

    const AlignedImage::view_t &left_data_view = left_data.get_view();
    const size_t data_row_uint8_step = (reinterpret_cast<uint8_t*>(left_data_view.row_begin(1)) -
                                        reinterpret_cast<uint8_t*>(left_data_view.row_begin(0)));
    //printf("data_row_uint8_step == %zi\n", data_row_uint8_step);
    //printf("left_data_view.row_begin(1) == %zi, left_data_view.row_begin(0) == %zi\n",
    //       left_data_view.row_begin(1), left_data_view.row_begin(0));

    //printf("row_step == %zi\n", row_step);
    //printf("left_start == %zi\n", left_start);

    vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();
    input_image_const_view_t::x_iterator
            left_begin_it = input_left_view.row_begin(0) + left_start,
            //left_end_it = input_left_view.row_begin(0) + left_end,
            right_begin_it = input_right_view.row_begin(0) + right_start;

    uint8_t *
            left_data_begin_uint8_it = reinterpret_cast<uint8_t*>(left_data_view.row_begin(0));

    for(int v=0; v < object_bottom_v;
        v+=1,
        ++disparity_cost_it,
        left_begin_it+=row_step,
        //left_end_it+=row_step,
        right_begin_it+=row_step,
        left_data_begin_uint8_it += data_row_uint8_step)
    {
        // we consider 5*3 bytes == 15, so we only need to discard the last byte
        const v16qi
                * const left_v16qi_it = reinterpret_cast<const v16qi*>(left_begin_it),
                * const right_v16qi_it = reinterpret_cast<const v16qi*>(right_begin_it);

        m128i * const left_data_m128i_it = reinterpret_cast<m128i*>(left_data_begin_uint8_it);

        uint16_t &cost = *disparity_cost_it;

        // simd section --

        // _mm_loadu_si128*2 + erase last byte + _mm_sad_epu8
        // copy data from non-aligned memory to register

        // const bool clean_last_byte = true; // FIXME just for debugging
        const bool clean_last_byte = false;

        if(clean_last_byte)
        {
            v16qi left_v16qi, right_v16qi;
            left_v16qi.m = _mm_loadu_si128(&(left_v16qi_it->m));
            right_v16qi.m = _mm_loadu_si128(&(right_v16qi_it->m));
            left_v16qi.v[7] = 0; right_v16qi.v[7] = 0; // 7 or 0 ?

            v2di sad;
            sad.m = _mm_sad_epu8(left_v16qi.m, right_v16qi.m);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16

            // _mm_stream copies the data to 16 bytes aligned memory,
            // without poluting the cache
            //*left_data_m128i_it = left_v16qi.m;
            _mm_stream_si128(left_data_m128i_it, left_v16qi.m);
            if(false)
            {
                const uint8_t
                        * const left_uint8_it = reinterpret_cast<const uint8_t*>(left_begin_it);
                std::copy(left_uint8_it, left_uint8_it + 15, left_data_begin_uint8_it);
            }
        }
        else
        {
            //printf("Hello !\n");
            //printf("left v16qi %li\n", left_v16qi_it->v[0]);
            //printf("left m128i %li\n", *left_data_m128i_it);

            const m128i left_m128i = _mm_loadu_si128(&(left_v16qi_it->m));
            const m128i right_m128i = _mm_loadu_si128(&(right_v16qi_it->m));
            v2di sad;
            sad.m = _mm_sad_epu8(left_m128i, right_m128i);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16

            // _mm_stream copies the data to 16 bytes aligned memory,
            // without poluting the cache
            //*left_data_m128i_it = left_v16qi_it->m; // accessing ->m creates a SegFault
            //*left_data_m128i_it = left_m128i;
            _mm_stream_si128(left_data_m128i_it, left_m128i);
            if(false)
            {
                const uint8_t
                        * const left_uint8_it = reinterpret_cast<const uint8_t*>(left_begin_it);
                std::copy(left_uint8_it, left_uint8_it + 16, left_data_begin_uint8_it);
            }
        }

    } // end of "for each pixel above the ground"

    return;
}

inline
void FastStixelsEstimatorWithHeightEstimation::compute_horizontal_5px_sad_simd_using_data(
    const int column,
    const int object_bottom_v,
    const int disparity,
    const AlignedImage &left_data,
    vector<uint16_t> &disparity_cost)
{


    //return; // FIXME just for debugging

    // we assume that all pixels are inside the image region
    const ptrdiff_t left_start = column - 2;
    //const ptrdiff_t left_end = column + 2;
    const ptrdiff_t right_start = left_start - disparity;

    const size_t row_step = input_left_view.row_begin(1) - input_left_view.row_begin(0);

    const AlignedImage::const_view_t &left_data_view = left_data.get_view();
    const size_t data_row_uint8_step = (reinterpret_cast<const uint8_t*>(left_data_view.row_begin(1)) -
                                        reinterpret_cast<const uint8_t*>(left_data_view.row_begin(0)));


    vector<uint16_t>::iterator disparity_cost_it = disparity_cost.begin();
    input_image_const_view_t::x_iterator
            //left_begin_it = input_left_view.row_begin(0) + left_start,
            //left_end_it = input_left_view.row_begin(0) + left_end,
            right_begin_it = input_right_view.row_begin(0) + right_start;

    const uint8_t *
            left_data_begin_uint8_it = reinterpret_cast<const uint8_t*>(left_data_view.row_begin(0));

    for(int v=0; v < object_bottom_v;
        v+=1,
        ++disparity_cost_it,
        //left_begin_it+=row_step,
        //left_end_it+=row_step,
        right_begin_it+=row_step,
        left_data_begin_uint8_it+=data_row_uint8_step)
    {
        // we consider 5*3 bytes == 15, so we only need to discard the last byte
        const v16qi
                //* const left_v16qi_it = reinterpret_cast<const v16qi*>(left_begin_it),
                * const right_v16qi_it = reinterpret_cast<const v16qi*>(right_begin_it);

        const m128i * const left_data_m128i_it = reinterpret_cast<const m128i*>(left_data_begin_uint8_it);

        uint16_t &cost = *disparity_cost_it;

        // simd section --

        // _mm_loadu_si128*2 + erase last byte + _mm_sad_epu8
        // copy data from non-aligned memory to register

        // const bool clean_last_byte = true; // FIXME just for debugging
        const bool clean_last_byte = false;

        if(clean_last_byte)
        {
            v16qi right_v16qi;
            right_v16qi.m = _mm_loadu_si128(&(right_v16qi_it->m));
            right_v16qi.v[7] = 0; // 7 or 0 ?
            // the left side was already handled at compute_horizontal_5px_sad_simd_and_copy_data

            v2di sad;
            sad.m = _mm_sad_epu8(*left_data_m128i_it, right_v16qi.m);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16
        }
        else
        {
            const m128i right_m128i = _mm_loadu_si128(&(right_v16qi_it->m));
            v2di sad;
            sad.m = _mm_sad_epu8(*left_data_m128i_it, right_m128i);
            cost += (sad.v[0] + sad.v[2]); // uint32 to uint16
        }

    } // end of "for each pixel above the ground"

    return;
}




template<typename VectorUint16>
inline
void do_vertical_averaging(
    const int object_bottom_v,
    const VectorUint16 &horizontal_sad_cost, VectorUint16 &disparity_cost)
{

    assert(horizontal_sad_cost.size() == disparity_cost.size());

    const bool do_the_vertical_averaging = true; // FIXME only for debugging
    //const bool do_the_vertical_averaging = false;

    if(do_the_vertical_averaging)
    {
        // for each pixel in the column
        uint16_t running_sum = 0; // the running sum of 5x5 RGB pixels SAD should fit in 14 bits

        // compute the running 5x1 vertical average --
        // first 5 pixels separatelly
        for(int v=0, v_minus_2=-2; v < 5 and v < object_bottom_v; v+=1, v_minus_2 +=1)
        {
            running_sum += horizontal_sad_cost[v];
            if(v_minus_2 > 0)
            {
                disparity_cost[v_minus_2] = running_sum;
            }
        }

        // rest of pixels
        for(int v=5, v_minus_2 = 3, v_minus_5 = 0;
            v < object_bottom_v;
            v+=1, v_minus_2 += 1, v_minus_5 += 1)
        {
            running_sum -= horizontal_sad_cost[v_minus_5];
            running_sum += horizontal_sad_cost[v];
            disparity_cost[v_minus_2] = running_sum;
        }
    }
    else
    { // do_the_vertical_averaging == false

        // the disparity cost is simply the horizontal_sad values
        std::copy(horizontal_sad_cost.begin(), horizontal_sad_cost.end(), disparity_cost.begin());
    }


    return;
}




inline
void FastStixelsEstimatorWithHeightEstimation::compute_column_disparity_cost(
    const int column, const int num_columns,
    const int disparity,
    vector<uint16_t> &disparity_cost)
{

    assert(disparity >= 0);

    const int object_bottom_v = u_v_ground_obstacle_boundary[column];

#if not defined(NDEBUG)
    const int num_rows = disparity_cost.size();
    assert(input_left_view.height() == num_rows);
    assert(object_bottom_v <= num_rows);
#endif

    const bool is_border_case =
            ((column - 2 - disparity) < 0)
            or ((column - 2) < 0)
            or ((column + 2) >= num_columns);

    vector<uint16_t> horizontal_sad_cost(disparity_cost.size(), 0);

    //const bool use_simd = false;
    const bool use_simd = true;

    // for each pixel above the ground, in the column
    // compute the 5x1 horizontal SAD
    if(is_border_case)
    {
        return; // FIXME only for debugging
        printf("Calling compute_horizontal_5px_sad_border_case at column %i\n", column);
        compute_horizontal_5px_sad_border_case(
                    column, num_columns,
                    object_bottom_v, disparity,
                    horizontal_sad_cost);
    }
    else
    { // is not border case
        // compute the 5x1 horizontal SAD using SIMD instructions
        if(use_simd)
        {
            compute_horizontal_5px_sad_simd(
                        column, object_bottom_v, disparity,
                        horizontal_sad_cost);
        }
        else
        {
            compute_horizontal_5px_sad_baseline(
                        column, object_bottom_v, disparity,
                        horizontal_sad_cost);
        }

    } // end of "if border case or not"

    do_vertical_averaging(object_bottom_v, horizontal_sad_cost, disparity_cost);

    return;
}



inline
void FastStixelsEstimatorWithHeightEstimation::compute_column_disparity_cost_and_copy_data(
    const int column,
    const int num_columns,
    const int disparity,
    AlignedImage &left_data,
    std::vector<uint16_t> &disparity_cost)
{

    assert(disparity >= 0);
    const int object_bottom_v = u_v_ground_obstacle_boundary[column];

#if not defined(NDEBUG)
    const int num_rows = disparity_cost.size();
    assert(input_left_view.height() == num_rows);
    assert(object_bottom_v <= num_rows);
#endif

    const bool is_border_case =
            ((column - 2 - disparity) < 0)
            or ((column - 2) < 0)
            or ((column + 2) >= num_columns);

    vector<uint16_t> horizontal_sad_cost(disparity_cost.size(), 0);

    //const bool use_simd = false;
    const bool use_simd = true;

    // for each pixel above the ground, in the column
    // compute the 5x1 horizontal SAD
    if(is_border_case)
    {
        return; // FIXME only for debugging
        printf("Calling compute_horizontal_5px_sad_border_case at column %i\n", column);
        compute_horizontal_5px_sad_border_case(
                    column, num_columns,
                    object_bottom_v, disparity,
                    horizontal_sad_cost);
    }
    else
    { // is not border case
        // compute the 5x1 horizontal SAD using SIMD instructions
        if(use_simd)
        {
            compute_horizontal_5px_sad_simd_and_copy_data(
                        column, object_bottom_v, disparity,
                        left_data, horizontal_sad_cost);
        }
        else
        {
            compute_horizontal_5px_sad_baseline(
                        column, object_bottom_v, disparity,
                        horizontal_sad_cost);
        }

    } // end of "if border case or not"

    do_vertical_averaging(object_bottom_v, horizontal_sad_cost, disparity_cost);

    return;
}


inline
void FastStixelsEstimatorWithHeightEstimation::compute_column_disparity_cost_using_data(
    const int column,
    const int num_columns,
    const int disparity,
    const AlignedImage &left_data,
    std::vector<uint16_t> &disparity_cost)
{

    assert(disparity >= 0);

    const int object_bottom_v = u_v_ground_obstacle_boundary[column];

#if not defined(NDEBUG)
    const int num_rows = disparity_cost.size();
    assert(input_left_view.height() == num_rows);
    assert(object_bottom_v <= num_rows);
#endif

    const bool is_border_case =
            ((column - 2 - disparity) < 0)
            or ((column - 2) < 0)
            or ((column + 2) >= num_columns);

    vector<uint16_t> horizontal_sad_cost(disparity_cost.size(), 0);

    //const bool use_simd = false;
    const bool use_simd = true;

    // for each pixel above the ground, in the column
    // compute the 5x1 horizontal SAD
    if(is_border_case)
    {
        return; // FIXME only for debugging
        printf("Calling compute_horizontal_5px_sad_border_case at column %i\n", column);
        compute_horizontal_5px_sad_border_case(
                    column, num_columns,
                    object_bottom_v, disparity,
                    horizontal_sad_cost);
    }
    else
    { // is not border case
        // compute the 5x1 horizontal SAD using SIMD instructions
        if(use_simd)
        {
            compute_horizontal_5px_sad_simd_using_data(
                        column, object_bottom_v, disparity,
                        left_data, horizontal_sad_cost);
        }
        else
        {
            compute_horizontal_5px_sad_baseline(
                        column, object_bottom_v, disparity,
                        horizontal_sad_cost);
        }

    } // end of "if border case or not"

    do_vertical_averaging(object_bottom_v, horizontal_sad_cost, disparity_cost);

    return;
}



inline
void FastStixelsEstimatorWithHeightEstimation::compute_column_disparity_cost_v2(
    const int column, const int num_columns,
    const int disparity,
    vector<uint16_t> &horizontal_sad_cost,
    vector<uint16_t> &disparity_cost)
{

    assert(disparity >= 0);
    assert(horizontal_sad_cost.size() == disparity_cost.size());

    assert(input_left_view.height() == disparity_cost.size());

    const int object_bottom_v = u_v_ground_obstacle_boundary[column];
#if not defined(NDEBUG)
    const int num_rows = disparity_cost.size();
    assert(object_bottom_v <= num_rows);
#endif

    const bool is_border_case =
            ((column - 2 - disparity) < 0)
            or ((column - 2) < 0)
            or ((column + 2) >= num_columns);

    // reset the cost
    horizontal_sad_cost.assign(horizontal_sad_cost.size(), 0);
    //vector<uint16_t> horizontal_sad_cost(disparity_cost.size(), 0);


    // for each pixel above the ground, in the column
    // compute the 5x1 horizontal SAD
    if(is_border_case)
    {
        return; // FIXME only for debugging
        printf("Calling compute_horizontal_5px_sad_border_case at column %i\n", column);
        compute_horizontal_5px_sad_border_case(
                    column, num_columns,
                    object_bottom_v, disparity,
                    horizontal_sad_cost);
    }
    else
    { // is not border case

        compute_transposed_horizontal_5px_sad_baseline(
                    column, object_bottom_v, disparity,
                    horizontal_sad_cost);

    } // end of "if border case or not"

    do_vertical_averaging(object_bottom_v, horizontal_sad_cost, disparity_cost);

    return;
}

/// disparity_likelihood_map is in range [0,1]
void FastStixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map_v0(Eigen::MatrixXf &disparity_likelihood_map)
{

    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;

    assert(disparity_likelihood_map.rows() == num_rows);
    assert(disparity_likelihood_map.cols() == num_columns);


    // FIXME hardcoded value
    //const float max_delta_cost = 50;
    //const float max_delta_cost = 250;
    const float max_delta_cost = 10;

    const float max_delta_cost_5x5 = max_delta_cost*5*5; // scale to a 5x5 window sum

    // FIXME hardcoded value
    //const float negative_factor = 1; // 0.5 // 0.1 // 0
    const float negative_factor = 2;


    // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
    for(int u=0; u < num_columns; u+=1)
    {
        int minimum_disparity, maximum_disparity, stixel_disparity;
        get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);

        const int object_bottom_v = u_v_ground_obstacle_boundary[u];
        assert(object_bottom_v <= num_rows);

        vector<uint16_t> stixel_disparity_cost(num_rows, 0);

        Eigen::VectorXf column_disparity_likelihood = Eigen::VectorXf::Zero(num_rows);
        // FIXME is it faster to use size object_bottom_v instead of num_rows ?

        // compute the stixel disparity cost --
        compute_column_disparity_cost(u, num_columns, stixel_disparity, stixel_disparity_cost);

        const float num_disparities_considered = (maximum_disparity - minimum_disparity) - 1;
        assert(num_disparities_considered > 0);

        //continue; // FIXME only for debugging

        // for each disparity in the search range (other than the stixel disparity) --
        for(int d=minimum_disparity; d < maximum_disparity; d+=1)
        {
            if(d == stixel_disparity)
            {
                // we skip the stixel_disparity
                continue;
            }

            // the temporary disparity being analyzed
            vector<uint16_t> t_disparity_cost(num_rows, 0);

            // compute the disparity cost --
            compute_column_disparity_cost(u, num_columns, d, t_disparity_cost);

            //continue; // FIXME only for debugging

            // for each pixel in the column --
            vector<uint16_t>::const_iterator
                    cost_it = t_disparity_cost.begin(),
                    stixel_cost_it = stixel_disparity_cost.begin();
            for(int v=0;
                //v < num_rows;
                v < object_bottom_v;
                v+=1, ++cost_it, ++stixel_cost_it)
            {
                const uint16_t &cost = *cost_it;
                const uint16_t &stixel_cost = *stixel_cost_it;
                float &disparity_likelihood = column_disparity_likelihood(v);

                // count the result with respect to the stixel disparity -
                const float delta_cost = (cost - stixel_cost);
                const float abs_delta_cost = std::min(max_delta_cost_5x5, std::abs(delta_cost));
                const float normalized_abs_delta_cost = abs_delta_cost / max_delta_cost_5x5;
                // normalized_delta_cost is in between [0 and 1]

                if(cost > stixel_cost)
                {
                    //disparity_likelihood += 1;
                    disparity_likelihood += normalized_abs_delta_cost;
                }
                else
                {
                    disparity_likelihood -= negative_factor*normalized_abs_delta_cost;
                    //disparity_likelihood -= negative_factor;
                }

            } // end of "for each pixel in the column"

        } // end of "for each disparity in the search range"

        column_disparity_likelihood /= num_disparities_considered;
        column_disparity_likelihood = \
                column_disparity_likelihood.array().max(Eigen::ArrayXf::Zero(num_rows));  // = std::max(0.0f, disparity_likelihood);

        // disparity_likelihood is in [0,1], the higher the likelihood, the better
        disparity_likelihood_map.col(u) = column_disparity_likelihood;

    } // end of "for each column on the image"

    return;
}


void FastStixelsEstimatorWithHeightEstimation::load_data_into_disparity_likelihood_helper()
{
    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;


    if(disparity_likelihood_helper_image.empty())
    {
        // if helper image not allocated, we resize

        // we store in one large aligned memory 2d array
        // in each row we have the sequence of left and right 5 pixels for the column
        // we have as many rows as columns x disparities
        disparity_likelihood_helper_image.resize(2*(5+1)*num_rows, num_columns*disparity_range);

    }

    const AlignedImage::view_t &helper_data_view =  disparity_likelihood_helper_image.get_view();
    boost::gil::fill_pixels(helper_data_view, AlignedImage::view_t::value_type(0,0,0)); // is this really necessary ?


    // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
    for(int u=0; u < num_columns; u+=1)
    {

        int minimum_disparity, maximum_disparity, stixel_disparity;
        get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);


        // for each disparity in the search range (including the stixel disparity) --
        for(int d=minimum_disparity; d < maximum_disparity; d+=1)
        {

            const int helper_data_row_index = u*disparity_range + d - minimum_disparity;

            boost::uint8_t * const
                    helper_data_begin_it = reinterpret_cast<boost::uint8_t *>(
                        helper_data_view.row_begin(helper_data_row_index));

            const int &disparity = d;
            assert(disparity >= 0);

            const int &column = u;
            const int object_bottom_v = u_v_ground_obstacle_boundary[column];
            assert(object_bottom_v <= num_rows);

            const bool is_border_case =
                    ((column - 2 - disparity) < 0)
                    or ((column - 2) < 0)
                    or ((column + 2) >= num_columns);

            if(is_border_case)
            {
                //return; // FIXME only for debugging
                //printf("Calling compute_horizontal_5px_sad_border_case at column %i\n", column);
                //compute_horizontal_5px_sad_border_case(...) // FIXME only for debugging
            }
            else
            { // is not border case

                // we assume that all pixels are inside the image region
                const int left_start = column - 2;
                //const int left_end = column + 2;
                const int right_start = left_start - disparity;

                const size_t row_step = input_left_view.row_begin(1) - input_left_view.row_begin(0);

                // left and right data will be aligned
                boost::uint8_t
                        *left_data_it = helper_data_begin_it,
                        *right_data_it = helper_data_begin_it + 16;

                input_image_const_view_t::x_iterator
                        left_begin_it = input_left_view.row_begin(0) + left_start,
                        //left_end_it = input_left_view.row_begin(0) + left_end,
                        right_begin_it = input_right_view.row_begin(0) + right_start;

                for(int v=0; v < object_bottom_v;
                    v+=1,
                    left_begin_it+=row_step,
                    //left_end_it+=row_step,
                    right_begin_it+=row_step,
                    left_data_it += 32,
                    right_data_it += 32)
                {
                    const boost::uint8_t
                            *left_it = reinterpret_cast<const boost::uint8_t *>(left_begin_it),
                            *right_it = reinterpret_cast<const boost::uint8_t *>(right_begin_it);

                    // copy 15 bytes (5 rgb pixels) from the image to the helper data buffer
                    // memcpy or std::copy are both super slow
                    std::memcpy(left_data_it, left_it, 15);
                    std::memcpy(right_data_it, right_it, 15);
                    //std::copy(left_it, left_it + 15, left_data_it);
                    //std::copy(right_it, right_it + 15, right_data_it);

                } // end of "for each pixel above the ground"

            } // end of "if border case or not"

        } // end of "for each disparity in the search range"

    } // end of "for each column on the image"

    return;
}


void FastStixelsEstimatorWithHeightEstimation::compute_colum_disparity_cost_from_disparity_likelihood_helper()
{
    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;


    if(column_disparity_cost.empty())
    { // not yet initialized
        column_disparity_cost.resize(boost::extents[num_columns*disparity_range][num_rows]);
    }

    if(column_disparity_horizontal_cost.empty())
    { // not yet initialized
        column_disparity_horizontal_cost.resize(boost::extents[num_columns*disparity_range][num_rows]);
    }

    const AlignedImage::view_t &helper_data_view =  disparity_likelihood_helper_image.get_view();

    const bool averaging_inside_the_for_loop = false;

    // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
    for(int u=0; u < num_columns; u+=1)
    {
        const int &column = u;
        const int object_bottom_v = u_v_ground_obstacle_boundary[column];
        assert(object_bottom_v <= num_rows);

        int minimum_disparity, maximum_disparity, stixel_disparity;
        get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);

        // for each disparity in the search range (including the stixel disparity) --
        for(int d=minimum_disparity; d < maximum_disparity; d+=1)
        {
            const size_t helper_data_row_index = u*disparity_range + (d - minimum_disparity);

            //printf("helper_data_row_index == %i, column_disparity_cost.shape()[0] == %zi, "
            //       "disparity_range == %i, delta_d == %i\n",
            //       helper_data_row_index, column_disparity_cost.shape()[0],
            //       disparity_range, (d - minimum_disparity));

            assert(helper_data_row_index >= 0);
            assert(helper_data_row_index < column_disparity_cost.shape()[0]);

            m128i * const
                    helper_data_begin_it = reinterpret_cast<m128i *>(
                        helper_data_view.row_begin(helper_data_row_index));

            column_disparity_cost_t::reference
                    disparity_horizontal_cost = column_disparity_horizontal_cost[helper_data_row_index],
                    disparity_cost = column_disparity_cost[helper_data_row_index];

            column_disparity_cost_t::reference::iterator
                    disparity_cost_it = disparity_horizontal_cost.begin();

            m128i
                    *left_it = helper_data_begin_it,
                    *right_it = helper_data_begin_it + 1;

            for(int v=0; v < object_bottom_v;
                v+=1,
                ++disparity_cost_it,
                ++left_it,
                ++right_it)
            {
                // data is aligned so we can directly access it for SAD computation
                v2di sad;
                sad.m = _mm_sad_epu8(*left_it, *right_it);
                (*disparity_cost_it) += (sad.v[0] + sad.v[2]); // uint32 to uint16
            } // end "for each pixel above the ground"

            if(averaging_inside_the_for_loop)
            {
                do_vertical_averaging(object_bottom_v, disparity_horizontal_cost, disparity_cost);
            }

        } // end of "for each disparity in the search range"
    } // end of "for each column on the image"

    if(averaging_inside_the_for_loop == false)
    {
        // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
        for(int u=0; u < num_columns; u+=1)
        {
            const int &column = u;
            const int object_bottom_v = u_v_ground_obstacle_boundary[column];
            assert(object_bottom_v <= num_rows);

            int minimum_disparity, maximum_disparity, stixel_disparity;
            get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);

            // for each disparity in the search range (including the stixel disparity) --
            for(int d=minimum_disparity; d < maximum_disparity; d+=1)
            {
                const size_t helper_data_row_index = u*disparity_range + (d - minimum_disparity);

                //printf("helper_data_row_index == %i, column_disparity_cost.shape()[0] == %zi\n",
                //       helper_data_row_index, column_disparity_cost.shape()[0]);

                assert(helper_data_row_index >= 0);
                assert(helper_data_row_index < column_disparity_cost.shape()[0]);

                column_disparity_cost_t::reference
                        disparity_horizontal_cost = column_disparity_horizontal_cost[helper_data_row_index],
                        disparity_cost = column_disparity_cost[helper_data_row_index];

                do_vertical_averaging(object_bottom_v, disparity_horizontal_cost, disparity_cost);

            } // end of "for each disparity in the search range"
        } // end of "for each column on the image"

    } // end of "averaging_inside_the_for_loop == false"

    return;
}


void FastStixelsEstimatorWithHeightEstimation::disparity_likelihood_from_colum_disparity_cost(Eigen::MatrixXf &disparity_likelihood_map)
{

    // this method is not implemented since already just calling
    // load_data_into_disparity_likelihood_helper();
    // runs at 6 Hz, which slower than desired for
    // FastStixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map_v1


    return;
}

/// disparity_likelihood_map is in range [0,1]
void FastStixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map_v1(Eigen::MatrixXf &disparity_likelihood_map)
{

    // before computing, we rearrange all the data into a more "simd friendly" order --
    load_data_into_disparity_likelihood_helper();

    // compute the 5x5 costs --
    //compute_colum_disparity_cost_from_disparity_likelihood_helper();

    // use the 5x5 costs to obtain the disparity_likelihood_map --
    //disparity_likelihood_from_colum_disparity_cost(disparity_likelihood_map);

    return;
}

/// disparity_likelihood_map is in range [0,1]
void FastStixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map_v2(Eigen::MatrixXf &disparity_likelihood_map)
{

    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;

    assert(disparity_likelihood_map.rows() == num_rows);
    assert(disparity_likelihood_map.cols() == num_columns);


    // FIXME hardcoded value
    //const float max_delta_cost = 50;
    //const float max_delta_cost = 250;
    const float max_delta_cost = 10;

    const float max_delta_cost_5x5 = max_delta_cost*5*5; // scale to a 5x5 window sum

    // FIXME hardcoded value
    //const float negative_factor = 1; // 0.5 // 0.1 // 0
    const float negative_factor = 2;


    // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
    for(int u=0; u < num_columns; u+=1)
    {
        int minimum_disparity, maximum_disparity, stixel_disparity;
        get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);

        vector<uint16_t>
                stixel_disparity_cost(num_rows, 0),
                t_horizontal_sad_cost(num_rows, 0), // temporary container
                t_disparity_cost(num_rows, 0);

        Eigen::VectorXf column_disparity_likelihood = Eigen::VectorXf::Zero(num_rows);

        // compute the stixel disparity cost --
        compute_column_disparity_cost_v2(u, num_columns, stixel_disparity, t_horizontal_sad_cost, stixel_disparity_cost);

        const float num_disparities_considered = (maximum_disparity - minimum_disparity) - 1;
        assert(num_disparities_considered > 0);

        //continue; // FIXME only for debugging

        // for each disparity in the search range (other than the stixel disparity) --
        for(int d=minimum_disparity; d < maximum_disparity; d+=1)
        {
            if(d == stixel_disparity)
            {
                // we skip the stixel_disparity
                continue;
            }

            // the temporary disparity being analyzed
            t_disparity_cost.assign(num_rows, 0); // reset to zero

            // compute the disparity cost --
            compute_column_disparity_cost_v2(u, num_columns, d, t_horizontal_sad_cost, t_disparity_cost);

            //continue; // FIXME only for debugging

            // for each pixel in the column --
            vector<uint16_t>::const_iterator
                    cost_it = t_disparity_cost.begin(),
                    stixel_cost_it = stixel_disparity_cost.begin();
            for(int v=0; v < num_rows; v+=1, ++cost_it, ++stixel_cost_it)
            {
                const uint16_t &cost = *cost_it;
                const uint16_t &stixel_cost = *stixel_cost_it;
                float &disparity_likelihood = column_disparity_likelihood(v);

                // count the result with respect to the stixel disparity -
                const float delta_cost = (cost - stixel_cost);
                const float abs_delta_cost = std::min(max_delta_cost_5x5, std::abs(delta_cost));
                const float normalized_abs_delta_cost = abs_delta_cost / max_delta_cost_5x5;
                // normalized_delta_cost is in between [0 and 1]

                if(cost > stixel_cost)
                {
                    //disparity_likelihood += 1;
                    disparity_likelihood += normalized_abs_delta_cost;
                }
                else
                {
                    disparity_likelihood -= negative_factor*normalized_abs_delta_cost;
                    //disparity_likelihood -= negative_factor;
                }

            } // end of "for each pixel in the column"

        } // end of "for each disparity in the search range"

        column_disparity_likelihood /= num_disparities_considered;
        column_disparity_likelihood = column_disparity_likelihood.array().max(Eigen::ArrayXf::Zero(num_rows));  // = std::max(0.0f, disparity_likelihood);

        // disparity_likelihood is in [0,1], the higher the likelihood, the better
        disparity_likelihood_map.col(u) = column_disparity_likelihood;

    } // end of "for each column on the image"

    return;
}


/// disparity_likelihood_map is in range [0,1]
void FastStixelsEstimatorWithHeightEstimation::compute_disparity_likelihood_map_v3(
    Eigen::MatrixXf &disparity_likelihood_map)
{

    const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const int num_disparities = this->num_disparities;

    assert(disparity_likelihood_map.rows() == num_rows);
    assert(disparity_likelihood_map.cols() == num_columns);

    // FIXME hardcoded value
    //const float max_delta_cost = 50;
    //const float max_delta_cost = 250;
    const float max_delta_cost = 10;

    const float max_delta_cost_5x5 = max_delta_cost*5*5; // scale to a 5x5 window sum

    // FIXME hardcoded value
    //const float negative_factor = 1; // 0.5 // 0.1 // 0
    const float negative_factor = 2;


    // for each column on the image --
#pragma omp parallel for // FIXME only for debugging
    for(int u=0; u < num_columns; u+=1)
    {
        int minimum_disparity, maximum_disparity, stixel_disparity;
        get_colum_disparities_range(u, minimum_disparity, maximum_disparity, stixel_disparity);

        const int object_bottom_v = u_v_ground_obstacle_boundary[u];
        assert(object_bottom_v <= num_rows);

        vector<uint16_t> stixel_disparity_cost(num_rows, 0);

        // data aligned buffer (6 pixels, because 5 + 1)
        AlignedImage left_column_data(6, num_rows);

        Eigen::VectorXf column_disparity_likelihood = Eigen::VectorXf::Zero(num_rows);

        // compute the stixel disparity cost --
        //compute_column_disparity_cost(u, num_columns, stixel_disparity, stixel_disparity_cost);
        compute_column_disparity_cost_and_copy_data(u, num_columns, stixel_disparity,
                                                    left_column_data, stixel_disparity_cost);


        if(false and (u == 250))
        {
            boost::gil::png_write_view("left_column_data.png", left_column_data.get_view());

            throw std::runtime_error("Debugging left_column_data.png content");
        }

        const float num_disparities_considered = (maximum_disparity - minimum_disparity) - 1;
        assert(num_disparities_considered > 0);

        //continue; // FIXME only for debugging

        // for each disparity in the search range (other than the stixel disparity) --
        for(int d=minimum_disparity; d < maximum_disparity; d+=1)
        {
            if(d == stixel_disparity)
            {
                // we skip the stixel_disparity
                continue;
            }

            // the temporary disparity being analyzed
            vector<uint16_t> t_disparity_cost(num_rows, 0);

            // compute the disparity cost --
            //compute_column_disparity_cost(u, num_columns, d, t_disparity_cost);
            compute_column_disparity_cost_using_data(u, num_columns, d,
                                                     left_column_data, t_disparity_cost);



            //continue; // FIXME only for debugging

            // for each pixel in the column --
            vector<uint16_t>::const_iterator
                    cost_it = t_disparity_cost.begin(),
                    stixel_cost_it = stixel_disparity_cost.begin();
            for(int v=0;
                //v < num_rows;
                v < object_bottom_v;
                v+=1, ++cost_it, ++stixel_cost_it)
            {
                const uint16_t &cost = *cost_it;
                const uint16_t &stixel_cost = *stixel_cost_it;
                float &disparity_likelihood = column_disparity_likelihood(v);

                // count the result with respect to the stixel disparity -
                const float delta_cost = (cost - stixel_cost);
                const float abs_delta_cost = std::min(max_delta_cost_5x5, std::abs(delta_cost));
                const float normalized_abs_delta_cost = abs_delta_cost / max_delta_cost_5x5;
                // normalized_delta_cost is in between [0 and 1]

                if(cost > stixel_cost)
                {
                    //disparity_likelihood += 1;
                    disparity_likelihood += normalized_abs_delta_cost;
                }
                else
                {
                    disparity_likelihood -= negative_factor*normalized_abs_delta_cost;
                    //disparity_likelihood -= negative_factor;
                }

            } // end of "for each pixel in the column"

        } // end of "for each disparity in the search range"

        column_disparity_likelihood /= num_disparities_considered;
        column_disparity_likelihood = column_disparity_likelihood.array().max(Eigen::ArrayXf::Zero(num_rows));  // = std::max(0.0f, disparity_likelihood);

        // disparity_likelihood is in [0,1], the higher the likelihood, the better
        disparity_likelihood_map.col(u) = column_disparity_likelihood;

    } // end of "for each column on the image"

    return;
}


inline
void fast_compute_stixels_heights_using_dynamic_programming_v0(
    const stixel_height_cost_t &stixel_height_cost,
    Eigen::MatrixXf &M_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    const MetricStereoCamera &stereo_camera,
    stixels_t &the_stixels)
{
    //printf("PING fast_compute_stixels_heights_using_dynamic_programming_v0\n");

    // This functions follows a logic similar to StixelsEstimator::compute_ground_obstacle_boundary()
    // but this time we use a cost function similar to the one defined in Badino et al. DAGM 2009
    // http://www.lelaps.de/papers/badino_dagm09.pdf

    //const int num_rows = stixel_height_cost.rows();
    const int num_columns = stixel_height_cost.cols();

    M_cost = stixel_height_cost;

    // FIXME hardcoded parameters
    const float k1 = 1; // [scaling factor]
    const float max_distance_for_influence = 3; // [meters]

    // do left to right pass (cumulate cost) ---

    // first column is already initialized with the stixel_height_cost value

    for(int col = 1; col < num_columns; col += 1)
    {
        const int previous_col = col - 1;
        const int previous_bottom_v = u_v_ground_obstacle_boundary[previous_col];
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        const Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);

        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));
        const float distance_factor = k1*std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));

#pragma omp parallel for
        for(int row = 10; row < bottom_v; row +=1 )
        {
            // M_cost(r, c) = stixel_height_cost + min_{rr}( M_cost(rr, c-1) + S(r,rr) )
            // S defined similar to Badino 2009, equation 5

            float min_M_plus_S = std::numeric_limits<float>::max();
            for(int rr = 10; rr < previous_bottom_v; rr +=1 )
            {
                const float rows_factor = std::abs(row - rr);
                const float cost_S = rows_factor*distance_factor;
                const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                min_M_plus_S = std::min(min_M_plus_S, M_plus_S);

            } // end of "for each row in previous stixel"

            M_cost(row, col) += min_M_plus_S;

        } // end of "for each row in current stixel"

    } // end of "for each column"

    //stixel_height_cost = M_cost; // for visualization only

    // do right to left pass  (find optimal boundary) ---

    // we find the first minimum
    int previous_r_star = 0;
    M_cost.col(num_columns -1).minCoeff(&previous_r_star);

    the_stixels[num_columns -1].top_y = previous_r_star;

    for(int col = num_columns -2; col >=0; col -= 1)
    {
        const int previous_col = col + 1;
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);
        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));
        const float distance_factor = k1*std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));

        // r_star = argmin_{r}( M(r, col) + S(previous_r_star, r) )
        int r_star = 0;
        float min_M_plus_S = std::numeric_limits<float>::max();
        for(int row = 10; row < bottom_v; row +=1 )
        {
            const float rows_factor = std::abs(row - previous_r_star);
            const float cost_S = rows_factor*distance_factor;
            const float M_plus_S = M_cost(row, col) + cost_S;

            if(M_plus_S < min_M_plus_S)
            {
                min_M_plus_S = M_plus_S;
                r_star = row;
            }
        } // end of "for each row in current stixel"

        //printf("PING the_stixels[%i].top_y = %i\n", col, r_star);
        current_stixel.top_y = r_star;
        current_stixel.default_height_value = false;
        previous_r_star = r_star;

    } // end of "for each column"

    return;
} // end of fast_compute_stixels_heights_using_dynamic_programming_v0


inline
void fast_compute_stixels_heights_using_dynamic_programming_v1(
    const stixel_height_cost_t &stixel_height_cost,
    Eigen::MatrixXf &M_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    const MetricStereoCamera &stereo_camera,
    stixels_t &the_stixels)
{
    //printf("PING fast_compute_stixels_heights_using_dynamic_programming_v1\n");

    // This functions follows a logic similar to StixelsEstimator::compute_ground_obstacle_boundary()
    // but this time we use a cost function similar to the one defined in Badino et al. DAGM 2009
    // http://www.lelaps.de/papers/badino_dagm09.pdf

    //const int num_rows = stixel_height_cost.rows();
    const int num_columns = stixel_height_cost.cols();

    M_cost = stixel_height_cost;

    // FIXME hardcoded parameters
    const float k1 = 1; // [scaling factor]
    const float max_distance_for_influence = 3; // [meters]
    const int begin_row = 10; // [pixels]

    // do left to right pass (cumulate cost) ---

    // first column is already initialized with the stixel_height_cost value

    vector<float> distance_factors(num_columns);

    for(int col = 1; col < num_columns; col += 1)
    {
        const int previous_col = col - 1;
        const int previous_bottom_v = u_v_ground_obstacle_boundary[previous_col];
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        const Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);

        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));
        const float distance_factor = k1*std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));
        distance_factors[col - 1] = distance_factor;

#pragma omp parallel for
        for(int row = begin_row; row < bottom_v; row +=1 )
        {
            // M_cost(r, c) = stixel_height_cost + min_{rr}( M_cost(rr, c-1) + S(r,rr) )
            // S defined similar to Badino 2009, equation 5

            float min_M_plus_S = std::numeric_limits<float>::max();
            for(int rr = begin_row; rr < previous_bottom_v; rr +=1 )
            {
                const int rows_factor = std::abs(row - rr);
                const float cost_S = rows_factor*distance_factor;
                const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                min_M_plus_S = std::min(min_M_plus_S, M_plus_S);
            } // end of "for each row in previous stixel"

            M_cost(row, col) += min_M_plus_S;
        } // end of "for each row in current stixel"
    } // end of "for each column"

    //stixel_height_cost = M_cost; // for visualization only

    // do right to left pass  (find optimal boundary) ---

    // we find the first minimum
    int previous_r_star = 0;
    M_cost.col(num_columns -1).minCoeff(&previous_r_star);

    the_stixels[num_columns -1].top_y = previous_r_star;

    for(int col = num_columns -2; col >=0; col -= 1)
    {
        //const int previous_col = col + 1;
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        //const Stixel &previous_stixel = the_stixels[previous_col];
        Stixel &current_stixel = the_stixels[col];

        const float &distance_factor = distance_factors[col];

        // r_star = argmin_{r}( M(r, col) + S(previous_r_star, r) )
        int r_star = 0;
        float min_M_plus_S = std::numeric_limits<float>::max();

        // using parallel reduction is faster than using Eigen minCoeff over a vector set in parallel
#pragma omp parallel
        {
            int local_r_star = 0;
            float local_min_M_plus_S = std::numeric_limits<float>::max();

#pragma omp for nowait
            for(int row = begin_row; row < bottom_v; row +=1 )
            {
                const int rows_factor = std::abs(row - previous_r_star);
                const float cost_S = rows_factor*distance_factor;
                const float M_plus_S = M_cost(row, col) + cost_S;
                //col_M_plus_S(row) = M_plus_S;
                if(M_plus_S < local_min_M_plus_S)
                {
                    local_min_M_plus_S = M_plus_S;
                    local_r_star = row;
                }
            } // end of "for each row in current stixel"

#pragma omp critical
            {
                if(local_min_M_plus_S < min_M_plus_S)
                {
                    min_M_plus_S = local_min_M_plus_S;
                    r_star = local_r_star;
                }
            } // end of "omp critical"
        } // end of "omp parallel"

        //printf("PING the_stixels[%i].top_y = %i\n", col, r_star);
        current_stixel.top_y = r_star;
        current_stixel.default_height_value = false;
        previous_r_star = r_star;

    } // end of "for each column"


    return;
} // end of fast_compute_stixels_heights_using_dynamic_programming_v1



inline
void fast_compute_stixels_heights_using_dynamic_programming_v2(
    const stixel_height_cost_t &stixel_height_cost,
    Eigen::MatrixXf &M_cost,
    const std::vector<int> &u_v_ground_obstacle_boundary,
    const MetricStereoCamera &stereo_camera,
    stixels_t &the_stixels)
{
    //printf("PING fast_compute_stixels_heights_using_dynamic_programming_v1\n");

    // This functions follows a logic similar to StixelsEstimator::compute_ground_obstacle_boundary()
    // but this time we use a cost function similar to the one defined in Badino et al. DAGM 2009
    // http://www.lelaps.de/papers/badino_dagm09.pdf

    //const int num_rows = stixel_height_cost.rows();
    const int num_columns = stixel_height_cost.cols();

    M_cost = stixel_height_cost;

    // FIXME hardcoded parameters
    const float k1 = 1; // [scaling factor]
    const float max_distance_for_influence = 3; // [meters]
    const int begin_row = 10; // [pixels]

    // do left to right pass (cumulate cost) ---

    // first column is already initialized with the stixel_height_cost value

    vector<float> distance_factors(num_columns);

    for(int col = 1; col < num_columns; col += 1)
    {
        const int previous_col = col - 1;
        const int previous_bottom_v = u_v_ground_obstacle_boundary[previous_col];
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        const Stixel &previous_stixel = the_stixels[previous_col];
        const Stixel &current_stixel = the_stixels[col];

        const float previous_stixel_disparity = std::max<float>(min_float_disparity, previous_stixel.disparity);
        const float current_stixel_disparity = std::max<float>(min_float_disparity, current_stixel.disparity);

        const float delta_distance = std::abs(
                    stereo_camera.disparity_to_depth(previous_stixel_disparity) -
                    stereo_camera.disparity_to_depth(current_stixel_disparity));
        const float distance_factor = k1*std::max(0.0f, 1 - (delta_distance/max_distance_for_influence));
        distance_factors[col - 1] = distance_factor;

#pragma omp parallel for
        for(int row = begin_row; row < bottom_v; row +=1 )
        {
            // M_cost(r, c) = stixel_height_cost + min_{rr}( M_cost(rr, c-1) + S(r,rr) )
            // S defined similar to Badino 2009, equation 5

            float min_M_plus_S = std::numeric_limits<float>::max();
            /*          for(int rr = begin_row; rr < previous_bottom_v; rr +=1 )
            {
                const int rows_factor = std::abs(row - rr);
                const float cost_S = rows_factor*distance_factor;
                const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                min_M_plus_S = std::min(min_M_plus_S, M_plus_S);
            } // end of "for each row in previous stixel"
*/
            {
                // rr < row --
                const int end_positive_row_factors = std::min(row, previous_bottom_v);
                //const int first_rows_factor = row - begin_row;
                //float cost_S = first_rows_factor*distance_factor;
                for(int rr = begin_row; rr < end_positive_row_factors; rr +=1 )
                {
                    const int rows_factor = row - rr; //std::abs(row - rr);
                    const float cost_S = rows_factor*distance_factor;
                    const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                    min_M_plus_S = std::min(min_M_plus_S, M_plus_S);

                    // rr increases, so rows_factor decreases,
                    // so cost_S decreases
                    //cost_S -= distance_factor;
                }

                // rr >= row --
                for(int rr = end_positive_row_factors; rr < previous_bottom_v; rr +=1 )
                {
                    const int rows_factor = rr - row; //std::abs(row - rr);
                    const float cost_S = rows_factor*distance_factor;
                    const float M_plus_S = M_cost(rr, previous_col) + cost_S;

                    min_M_plus_S = std::min(min_M_plus_S, M_plus_S);

                    // rr increases, so rows_factor increases,
                    // so cost_S increases
                    //cost_S += distance_factor;
                }

            } // end of "for each row in previous stixel"


            M_cost(row, col) += min_M_plus_S;

        } // end of "for each row in current stixel"

    } // end of "for each column"

    //stixel_height_cost = M_cost; // for visualization only

    // do right to left pass  (find optimal boundary) ---

    // we find the first minimum
    int previous_r_star = 0;
    M_cost.col(num_columns -1).minCoeff(&previous_r_star);

    the_stixels[num_columns -1].top_y = previous_r_star;

    for(int col = num_columns -2; col >=0; col -= 1)
    {
        //const int previous_col = col + 1;
        const int bottom_v = u_v_ground_obstacle_boundary[col];
        //const Stixel &previous_stixel = the_stixels[previous_col];
        Stixel &current_stixel = the_stixels[col];

        const float &distance_factor = distance_factors[col];

        // r_star = argmin_{r}( M(r, col) + S(previous_r_star, r) )
        int r_star = 0;
        float min_M_plus_S = std::numeric_limits<float>::max();

        // using parallel reduction is faster than using Eigen minCoeff over a vector set in parallel
#pragma omp parallel
        {
            int local_r_star = 0;
            float local_min_M_plus_S = std::numeric_limits<float>::max();

            {
#pragma omp for nowait
                for(int row = begin_row; row < bottom_v; row +=1 )
                {
                    const int rows_factor = std::abs(row - previous_r_star);
                    const float cost_S = rows_factor*distance_factor;
                    const float M_plus_S = M_cost(row, col) + cost_S;
                    //col_M_plus_S(row) = M_plus_S;
                    if(M_plus_S < local_min_M_plus_S)
                    {
                        local_min_M_plus_S = M_plus_S;
                        local_r_star = row;
                    }
                }
                /*
                // row < previous_r_star --
                const int end_positive_row_factors = std::min(begin_row, previous_r_star);
                const int first_rows_factor = previous_r_star - begin_row;
                float cost_S = first_rows_factor*distance_factor;

                for(int row = begin_row; row < end_positive_row_factors; row +=1 )
                {
                    //const float rows_factor = previous_r_star - row; //std::abs(row - previous_r_star);
                    //const float cost_S = rows_factor*distance_factor;

                    const float M_plus_S = M_cost(row, col) + cost_S;
                    //col_M_plus_S(row) = M_plus_S;
                    if(M_plus_S < local_min_M_plus_S)
                    {
                        local_min_M_plus_S = M_plus_S;
                        local_r_star = row;
                    }

                    // row increases, so rows_factor decreases,
                    // so cost_S decreases
                    cost_S -= distance_factor;
                }


                // row >= previous_r_star
                for(int row = begin_row; row < bottom_v; row +=1 )
                {
                    //const float rows_factor = row - previous_r_star; //std::abs(row - previous_r_star);
                    //const float cost_S = rows_factor*distance_factor;

                    const float M_plus_S = M_cost(row, col) + cost_S;
                    //col_M_plus_S(row) = M_plus_S;
                    if(M_plus_S < local_min_M_plus_S)
                    {
                        local_min_M_plus_S = M_plus_S;
                        local_r_star = row;
                    }

                    // row increases, so rows_factor increases,
                    // so cost_S increases
                    cost_S += distance_factor;
                }
*/
            }  // end of "for each row in current stixel"

#pragma omp critical
            {
                if(local_min_M_plus_S < min_M_plus_S)
                {
                    min_M_plus_S = local_min_M_plus_S;
                    r_star = local_r_star;
                }
            } // end of "omp critical"
        } // end of "omp parallel"

        //printf("PING the_stixels[%i].top_y = %i\n", col, r_star);
        current_stixel.top_y = r_star;
        current_stixel.default_height_value = false;
        previous_r_star = r_star;

    } // end of "for each column"


    return;
} // end of fast_compute_stixels_heights_using_dynamic_programming_v2



/// Apply dynamic programming over the stixels heights
void FastStixelsEstimatorWithHeightEstimation::compute_stixels_heights()
{
    // v0 is slower than v1, v2 is slower than v1
    //fast_compute_stixels_heights_using_dynamic_programming_v0(
    fast_compute_stixels_heights_using_dynamic_programming_v1(
                //fast_compute_stixels_heights_using_dynamic_programming_v2(
                stixel_height_cost,
                M_cost,
                u_v_ground_obstacle_boundary,
                stereo_camera,
                the_stixels);

    enforce_reasonable_stixels_heights();
    return;
} // end of FastStixelsEstimatorWithHeightEstimation::compute_stixels_heights

void FastStixelsEstimatorWithHeightEstimation::enforce_reasonable_stixels_heights()
{
    stixels_height_post_processing_p->operator ()(
                expected_v_given_disparity,
                the_ground_plane,
                the_stixels);
    return;
} // end of FastStixelsEstimatorWithHeightEstimation::enforce_reasonable_stixels_heights


void FastStixelsEstimatorWithHeightEstimation::get_colum_disparities_range(
    const int column,
    int &minimum_disparity, int &maximum_disparity, int &stixel_disparity)
{
    // FIXME should we precompute these ?

    stixel_disparity = u_disparity_ground_obstacle_boundary[column];

    // compute the disparities range --
    maximum_disparity = this->num_disparities;
    minimum_disparity = 0;

    // will use_partial_depth_range if
    // maximum_disparity_margin >=0 or minimum_disparity_margin >= 0
    if(maximum_disparity_margin >= 0)
    {
        //const int maximum_disparity_margin = 5;
        //const int maximum_disparity = disparity_given_v[v];
        maximum_disparity = std::min(stixel_disparity + maximum_disparity_margin, num_disparities);
    }

    if(minimum_disparity_margin >= 0)
    {
        //const int minimum_disparity_margin = 15;
        minimum_disparity = std::max(0, stixel_disparity - minimum_disparity_margin);
    }

    assert(minimum_disparity <= maximum_disparity);

    return;
}


} // end of namespace doppia
