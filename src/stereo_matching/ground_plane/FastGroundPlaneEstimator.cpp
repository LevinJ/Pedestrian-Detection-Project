#include "FastGroundPlaneEstimator.hpp"

#include "image_processing/ResidualImageFilter.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/AlignedImage.hpp"
#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"

#include <boost/gil/gil_all.hpp>
#include <boost/foreach.hpp>

#include "helpers/simd_intrisics_types.hpp"

// include SSE2 intrinsics
#include <emmintrin.h>

#include <omp.h>

#include <utility>
#include <cstdio>
#include <algorithm> // defines min and max

namespace {

using namespace logging;

std::ostream &log_info()
{
    return log(InfoMessage, "FastGroundPlaneEstimator");
}

std::ostream &log_warning()
{
    return log(WarningMessage, "FastGroundPlaneEstimator");
}

std::ostream &log_error()
{

    return log(ErrorMessage, "FastGroundPlaneEstimator");
}

std::ostream &log_debug()
{

    return log(DebugMessage, "FastGroundPlaneEstimator");
}

} // end of private namespace

namespace doppia {

using namespace boost;
using namespace boost::gil;

typedef FastGroundPlaneEstimator::points_t points_t;
typedef FastGroundPlaneEstimator::v_disparity_row_slice_t v_disparity_row_slice_t;


program_options::options_description FastGroundPlaneEstimator::get_args_options()
{
    program_options::options_description desc("FastGroundPlaneEstimator options");

    desc.add_options()

            ("ground_plane_estimator.use_residual",
             program_options::value<bool>()->default_value(false),
             "use (fast) residual computation")

            ("ground_plane_estimator.y_stride",
             program_options::value<int>()->default_value(1),
             "vertical step size defining which rows should be used to collect ground plane information. "
             "y_stride <= 1, means all rows (below the horizon) will be used. "
             "y_stride = 10, means that one of ten rows will be used.")

            ;

    return desc;

}


FastGroundPlaneEstimator::FastGroundPlaneEstimator(
        const boost::program_options::variables_map &options,
        const StereoCameraCalibration &stereo_calibration)
    : BaseGroundPlaneEstimator(options, stereo_calibration),
      max_disparity(128),
      num_ground_plane_estimation_failures(0)
{
    should_do_residual_computation = get_option_value<bool>(options, "ground_plane_estimator.use_residual");
    irls_lines_detector_p.reset(new IrlsLinesDetector(options));
    lines_detector_p = irls_lines_detector_p;

    const int y_stride_value = get_option_value<int>(options, "ground_plane_estimator.y_stride");

    if(y_stride_value <= 1)
    {
        y_stride = 1;
    }
    else if (y_stride_value < 100)
    {
        y_stride = static_cast<boost::uint8_t>(y_stride_value);
    }
    else
    {
        throw std::runtime_error("ground_plane_estimator.y_stride has a unexpectedly high value, "
                                 "reasonable values are in the range 1 to 20.");
    }


    silent_mode = true;
    if(options.count("silent_mode"))
    {
        silent_mode = get_option_value<bool>(options, "silent_mode");
    }

    // set cost_sum_saturation -
    {
        // cost_saturation: saturation value for each pixel color channel
        uint8_t cost_saturation = 5;
        if(should_do_residual_computation)
        {
            cost_saturation = 5; //100;
        }
        cost_sum_saturation = cost_saturation*3*16;
    }

    residual_image_filter_p.reset(new ResidualImageFilter());

    confidence_is_up_to_date = false;
    estimated_confidence = 0;

    return;
}


FastGroundPlaneEstimator::~FastGroundPlaneEstimator()
{
    // nothing to do here
    return;
}


void FastGroundPlaneEstimator::set_rectified_images_pair(input_image_view_t &left, input_image_view_t &right)
{
    assert( left.dimensions() == right.dimensions() );

    typedef input_image_view_t::point_t point_t;

    // we will only use the bottom half of the input images
    point_t top_left(0, left.height()/2);
    point_t dimensions(left.width(), left.height()/2);
    input_left_view = subimage_view(left, top_left, dimensions);
    input_right_view = subimage_view(right, top_left, dimensions);

    if(left_image_p == false)
    {
        left_image_p.reset(new AlignedImage(dimensions));
    }

    if(right_image_p == false)
    {
        right_image_p.reset(new AlignedImage(dimensions));
    }

    assert(left_image_p->dimensions() == input_left_view.dimensions() );
    assert(right_image_p->dimensions() == input_right_view.dimensions() );

    if(should_do_residual_computation)
    {
        // compute the residual image and store in the new view
        (*residual_image_filter_p)(input_left_view, left_image_p->get_view());
        (*residual_image_filter_p)(input_right_view, right_image_p->get_view());
    }
    else
    {
        // copy the input data into the memory aligned data structures
        copy_pixels(input_left_view, left_image_p->get_view());
        copy_pixels(input_right_view, right_image_p->get_view());
    }

    // lazy resize the v_disparity_data and update the v_disparity_view
    if(v_disparity_data.shape()[0] != static_cast<size_t>(dimensions.y))
    {
        v_disparity_data.resize(boost::extents[dimensions.y][max_disparity]);
        v_disparity_view = interleaved_view(
                    v_disparity_data.shape()[1], v_disparity_data.shape()[0],
                    reinterpret_cast<gray32_pixel_t *>(v_disparity_data.data()),
                    v_disparity_data.strides()[0] * sizeof(gray32_pixel_t));

        //printf("v_disparity_view dimensions (%zi, %zi)\n",
        //       v_disparity_view.width(), v_disparity_view.height());
        //printf("rowsize_in_bytes %zi\n",
        //       v_disparity_data.strides()[0] * sizeof(gray32_pixel_t));

    }

    // reset the points list and weigths estimates --
    points.clear();
    row_weights = Eigen::VectorXf::Ones(input_left_view.height());
    return;
}


const FastGroundPlaneEstimator::v_disparity_data_t &FastGroundPlaneEstimator::get_v_disparity() const
{
    return v_disparity_data;
}


const FastGroundPlaneEstimator::v_disparity_const_view_t &FastGroundPlaneEstimator::get_v_disparity_view() const
{
    return v_disparity_view;
}


const FastGroundPlaneEstimator::points_t &FastGroundPlaneEstimator::get_points() const
{
    return points;
}


const float FastGroundPlaneEstimator::get_confidence()
{
    if(confidence_is_up_to_date == false)
    {
        const float l1_residual = irls_lines_detector_p->compute_l1_residual();
        const float max_expected_l1_residual = (max_disparity / 4)*points.size();
        estimated_confidence = 1.0f - std::min(1.0f, l1_residual / max_expected_l1_residual);
        confidence_is_up_to_date = true;
    }

    return estimated_confidence;
}


bool FastGroundPlaneEstimator::is_computing_residual_image() const
{
    return should_do_residual_computation;
}


const FastGroundPlaneEstimator::input_image_view_t FastGroundPlaneEstimator::get_left_half_view() const
{
    return left_image_p->get_view();
}


void set_points_weights(const points_t &points,
                        const Eigen::VectorXf &row_weights,
                        Eigen::VectorXf &points_weights)
{
    points_weights.setOnes(points.size());

    int i=0;
    BOOST_FOREACH(points_t::const_reference point, points)
    {
        const int &point_y = point.second;
        points_weights(i) = row_weights(point_y);
        i+=1;
    }

    return;
}


void FastGroundPlaneEstimator::compute()
{    
    static int num_iterations = 0;
    static double cumulated_time = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();

    // compute v_disparity --
    compute_v_disparity_data();

    set_points_weights(points, row_weights, points_weights);
    // compute line --
    estimate_ground_plane();

    confidence_is_up_to_date = false;

    // timing ---
    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;

    if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
    {
        printf("FastGroundPlaneEstimator::compute speed \033[36m%.2lf [Hz]\033[0m (average in the last %i iterations)\n",
               num_iterations / cumulated_time, num_iterations );
    }

    return;
}


void FastGroundPlaneEstimator::compute_v_disparity_data()
{
    const bool use_simd = true;
    if(use_simd)
    {
        // for each pixel and each disparity value
        // guided provides 470 Hz versus 450 or lesss for the other schedule options
#pragma omp parallel for schedule(guided)
        for(int row=0; row < left_image_p->dimensions().y; row += y_stride)
        {
            v_disparity_row_slice_t row_slice = v_disparity_data[row];
            compute_v_disparity_row_simd(left_image_p->get_view(), right_image_p->get_view(),
                                         row, row_slice);
        }
    }
    else
    {
        // for each pixel and each disparity value
#pragma omp parallel for
        for(int row=0; row < left_image_p->dimensions().y; row += y_stride)
        {
            v_disparity_row_slice_t row_slice = v_disparity_data[row];
            compute_v_disparity_row_baseline(left_image_p->get_view(), right_image_p->get_view(),
                                             row, row_slice);
        }
    }

    //printf("num_points == %i\n", points.size());
    return;
} // end of FastGroundPlaneEstimator::compute_v_disparity_data


inline
void select_points_and_weight(
        const size_t max_disparity,
        const int row,
        const FastGroundPlaneEstimator::v_disparity_row_slice_t &v_disparity_row,
        const v_disparity_row_slice_t::value_type min_cost,
        IrlsLinesDetector::points_t &points,
        Eigen::VectorXf &row_weights)
{
    typedef v_disparity_row_slice_t::value_type cost_t;

    // FIXME hardcoded value
    const int max_num_points_in_row = 20;
    const cost_t delta_cost = 1; // 2 // 5
    const cost_t threshold_cost = min_cost + delta_cost;
    int num_points_in_row = 0;
    for(size_t d=0; d < max_disparity; d+=1)
    {
        if( v_disparity_row[d] <= threshold_cost)
        {
            const int x = d, y = row;

#pragma omp critical
            {
                points.push_back( std::make_pair(x,y) );
            }
            num_points_in_row += 1;
            if(num_points_in_row > max_num_points_in_row)
            {
                // this line is useless, no need to collect more points
                break;
            }
        }
        else
        {
            // we discard this point
            continue;
        }
    } // end of "for each disparity"

    if(num_points_in_row > 0)
    {
        // rows with less points give more confidence
        row_weights(row) = 1.0/num_points_in_row;
    }

    return;
}


inline
void FastGroundPlaneEstimator::compute_v_disparity_row_baseline(
        const input_image_view_t &left, const input_image_view_t &right,
        const int row,
        v_disparity_row_slice_t &v_disparity_row)
{
    typedef input_image_view_t::value_type pixel_t;

    const uint32_t the_cost_sum_saturation = cost_sum_saturation;

    assert(max_disparity <= v_disparity_row.size());

    const int disparity_offset = stereo_calibration.get_disparity_offset_x();

    typedef v_disparity_row_slice_t::value_type cost_t;
    cost_t min_cost = std::numeric_limits<cost_t>::max();
    //cost_t max_cost = 0;

    const input_image_view_t::x_iterator
            left_row_begin_it = left.row_begin(row),
            left_row_end_it = left.row_end(row),
            right_row_begin_it = right.row_begin(row);

    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;
    for(size_t d=0; d < max_disparity; d+=1)
    {
        cost_t v_disparity_cost = 0;

        input_image_view_t::x_iterator
                // FIXME this is most certainly a memory access error (d + d_offset)
                left_row_it = left_row_begin_it + (d + disparity_offset),
                right_row_it = right_row_begin_it;

        //for(size_t col=0; col < input_dimensions.x; col +=1)
        for(; left_row_it != left_row_end_it; ++left_row_it, ++right_row_it)
        {
            const cost_t cost = sad_cost_uint16(*left_row_it, *right_row_it);
            v_disparity_cost += std::min(cost, the_cost_sum_saturation);
        } // end of "for each column"

        // we divide once at the end of the sums
        // this is ok to delay the division because
        // log2(1024*255*3) ~= 20 [bits]
        // so there is no risk of overflow inside 32bits
        v_disparity_cost /= 3;
        v_disparity_row[d] = v_disparity_cost;

        min_cost = std::min(v_disparity_cost, min_cost);
        //max_cost = std::max(v_disparity_cost, max_cost);
    } // end of "for each disparity"


    // select points to use for ground estimation --
    select_points_and_weight(max_disparity, row,
                             v_disparity_row, min_cost,
                             points, row_weights);


    return;
} // end of FastGroundPlaneEstimator::compute_v_disparity_row_baseline


inline
void FastGroundPlaneEstimator::compute_v_disparity_row_simd(
        const input_image_view_t &left, const input_image_view_t &right,
        const int row,
        v_disparity_row_slice_t v_disparity_row)
{

    typedef input_image_view_t::value_type pixel_t;

    const int disparity_offset = stereo_calibration.get_disparity_offset_x();
    //printf("disparity_offset == %i\n", disparity_offset);

    if (false and disparity_offset < 0)
    {
        throw std::runtime_error("FastGroundPlaneEstimator::compute_v_disparity_row_simd "
                                 "does not yet support negative disparity offsets");
    }


    const uint32_t the_cost_sum_saturation = cost_sum_saturation;

    assert(max_disparity <= v_disparity_row.size());
    typedef v_disparity_row_slice_t::value_type cost_t;
    cost_t min_cost = std::numeric_limits<cost_t>::max();
    //cost_t max_cost = 0;

    const int
            width_in_pixels = left.width(),
            width_in_bytes = width_in_pixels * sizeof(pixel_t);

    const input_image_view_t::x_iterator
            left_row_begin_it = left.row_begin(row),
            right_row_begin_it = right.row_begin(row);

    // reinterpret_cast == danger !
    //const v16qi
            //*left_row_v16qi_begin_it = reinterpret_cast<const v16qi*>(left_row_begin_it),
           // *right_row_v16qi_begin_it = reinterpret_cast<const v16qi*>(right_row_begin_it);

    const uint8_t
            *left_row_uint8_begin_it = reinterpret_cast<const uint8_t*>(left_row_begin_it),
            *right_row_uint8_begin_it = reinterpret_cast<const uint8_t*>(right_row_begin_it),
            *left_row_uint8_end_it = left_row_uint8_begin_it + width_in_bytes,
            *right_row_uint8_end_it = right_row_uint8_begin_it + width_in_bytes;

    assert(left_row_uint8_end_it == reinterpret_cast<const uint8_t*>(left.row_end(row)));

    // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
    //const int first_right_x = first_left_x - disparity;
    for(size_t d=0; d < max_disparity; d+=1)
    {
        v_disparity_row_slice_t::value_type v_disparity_cost = 0;

        const int
                // disparity_in_bytes may be negative
                disparity_in_bytes = (d + disparity_offset) * sizeof(pixel_t),
                left_width_in_bytes = std::min(width_in_bytes - disparity_in_bytes, width_in_bytes),
                left_simd_width_in_bytes =  left_width_in_bytes - (left_width_in_bytes % sizeof(v16qi)),
                left_simd_cols =  left_simd_width_in_bytes / sizeof(v16qi),
                simd_cols = left_simd_cols;

        // simd section --
        const v16qi
                *left_row_v16qi_it = reinterpret_cast<const v16qi*>(left_row_uint8_begin_it + disparity_in_bytes),
                *left_row_v16qi_begin_it = reinterpret_cast<const v16qi*>(left_row_uint8_begin_it),
                *left_row_v16qi_end_it = left_row_v16qi_it + simd_cols,
                *right_row_v16qi_it = reinterpret_cast<const v16qi*>(right_row_uint8_begin_it);

        {
            while(left_row_v16qi_it < left_row_v16qi_begin_it)
            {
                // we simply skip the initial pixels
                ++left_row_v16qi_it;
                ++right_row_v16qi_it;
            }

            v2di sad;
            for(;
                left_row_v16qi_it != left_row_v16qi_end_it;
                ++left_row_v16qi_it, ++right_row_v16qi_it)
            {
                // _mm_loadu_si128 + _mm_sad_epu8
                // copy data from non-aligned memory to register
                const m128i left_v16qi = _mm_loadu_si128(&(left_row_v16qi_it->m));
                sad.m = _mm_sad_epu8(left_v16qi, right_row_v16qi_it->m);
                //v_disparity_cost += sad.v[0] + sad.v[2];
                v_disparity_cost += std::min(sad.v[0] + sad.v[2], the_cost_sum_saturation);

            } // end of "for each column"

        }

        // non simd section at the end --
        {
            // we continue from where we left
            const uint8_t
                    *left_row_uint8_it = reinterpret_cast<const uint8_t*>(left_row_v16qi_it),
                    *right_row_uint8_it =  reinterpret_cast<const uint8_t*>(right_row_v16qi_it);

            assert(left_row_uint8_it <= left_row_uint8_end_it);
            assert(right_row_uint8_it <= right_row_uint8_end_it);

            for(; left_row_uint8_it != left_row_uint8_end_it and right_row_uint8_it != right_row_uint8_end_it;
                ++left_row_uint8_it, ++right_row_uint8_it)
            {
                const int16_t delta = *left_row_uint8_it - *right_row_uint8_it;
                const uint8_t sad = std::abs(delta);
                v_disparity_cost += sad;
            } // end of "for each column"
        }

        // we divide once at the end of the sums
        // this is ok to delay the division because
        // log2(1024*255*3) ~= 20 bits
        // so there is no risk of overflow inside 32 bits
        v_disparity_cost /= 3;

        v_disparity_row[d] = v_disparity_cost;

        min_cost = std::min(v_disparity_cost, min_cost);
        //max_cost = std::max(v_disparity_cost, max_cost);
    } // end of "for each disparity"


    // select points to use for ground estimation --
    select_points_and_weight(max_disparity, row,
                             v_disparity_row, min_cost,
                             points, row_weights);

    return;
} // end of FastGroundPlaneEstimator::compute_v_disparity_row_simd


bool FastGroundPlaneEstimator::find_ground_line(AbstractLinesDetector::line_t &ground_line) const
{
    // find the most likely plane (line) in the v-disparity image ---
    AbstractLinesDetector::lines_t found_lines;
    bool found_ground_plane = false;

    // we correct the origin of our estimate lines
    // since we computed them using the lower half of the image
    const int origin_offset = input_left_view.height();
    //printf("origin_offset == %i\n", origin_offset);

    const line_t line_prior = ground_plane_to_v_disparity_line(estimated_ground_plane);

    line_t line_prior_with_offset = line_prior;
    line_prior_with_offset.origin()(0) -= origin_offset;
    irls_lines_detector_p->set_initial_estimate(line_prior_with_offset);

    (*irls_lines_detector_p)(points, points_weights, found_lines);
    //printf("irls_lines_detector_p found %zi lines\n", found_lines.size());

    BOOST_FOREACH(line_t &line, found_lines)
    {
        line.origin()(0) += origin_offset;
    }

    // given two bounding lines we verify the x=0 line and the y=max_y line
    // this checks bound quite well the desired ground line

    const float direction_fraction = 1.5; // FIXME hardcoded value
    const float max_line_direction = prior_max_v_disparity_line.direction()(0)*direction_fraction,
            min_line_direction = prior_min_v_disparity_line.direction()(0)/direction_fraction;

    const float max_line_y0 = prior_max_v_disparity_line.origin()(0),
            min_line_y0 = prior_min_v_disparity_line.origin()(0);

    const float min_y0 = max_line_y0, max_y0 = min_line_y0;

    const float y_intercept = input_left_view.height();
    const float max_x_intercept = (y_intercept - max_line_y0) / max_line_direction,
            min_x_intercept = (y_intercept - min_line_y0) / min_line_direction;

    assert(min_y0 < max_y0);
    assert(min_x_intercept < max_x_intercept);

    BOOST_FOREACH(line_t t_ground_line, found_lines)
    {
        const float t_y0 = t_ground_line.origin()(0);
        const float t_direction =  t_ground_line.direction()(0);
        const float t_x_intercept = (y_intercept - t_y0) / t_direction;

        const bool print_xy_check = false;
        if(print_xy_check)
        {
            printf("prior origin == %.3f, direction == %.3f\n",
                   line_prior.origin()(0), line_prior.direction()(0));

            printf("line origin == %.3f, direction == %.3f\n",
                   t_ground_line.origin()(0), t_ground_line.direction()(0));

            printf("max_y0 == %.3f, t_y0 == %.3f, min_y0 = %.3f\n",
                   max_y0, t_y0, min_y0);

            printf("max_x_intercept == %.3f, t_x_intercept == %.3f, min_x_intercept = %.3f\n",
                   max_x_intercept, t_x_intercept, min_x_intercept);

            printf("max_line_direction == %.3f, t_direction == %.3f, min_line_direction = %.3f\n",
                   max_line_direction, t_direction, min_line_direction);
        }

        if(t_y0 <= max_y0 and t_y0 >= min_y0 and
                t_x_intercept <= max_x_intercept and t_x_intercept >= min_x_intercept and
                t_direction <= max_line_direction and t_direction >= min_line_direction )
        {
            ground_line = t_ground_line;
            found_ground_plane = true;
            break;
        }
        else
        {
            continue;
        }

    } // end of "for each found line"

    return found_ground_plane;
}


void FastGroundPlaneEstimator::estimate_ground_plane()
{
    const bool found_ground_plane = find_ground_line(v_disparity_ground_line);

    const float weight = std::max(0.0f, 1.0f - get_confidence());

    const float minimum_weight = rejection_threshold;
    //const float minimum_weight = 0.0015;

    // retrieve ground plane parameters --
    if((found_ground_plane == true) and (weight >  minimum_weight))
    {
        set_ground_plane_estimate(
                    v_disparity_line_to_ground_plane(v_disparity_ground_line), weight);

        const bool print_estimated_plane = false;
        if(print_estimated_plane)
        {
            log_debug() << "Found a ground plane with " <<
                           "heigth == " << estimated_ground_plane.get_height() << " [meters]"
                           " and pitch == " << estimated_ground_plane.get_pitch() * 180 / M_PI << " [degrees]" <<
                           std::endl;
            log_debug() << "Ground plane comes from line with " <<
                           "origin == " << v_disparity_ground_line.origin()(0) << " [pixels]"
                           " and direction == " << v_disparity_ground_line.direction()(0) << " [-]" <<
                           std::endl;
        }
    }
    else
    {
        num_ground_plane_estimation_failures += 1;

        // in case this happened during the first call
        // we set the v_disparity_ground_line using the current ground plane estimate
        v_disparity_ground_line = ground_plane_to_v_disparity_line( get_ground_plane() );

        static int num_ground_warnings = 0;
        //const int max_num_ground_warnings = 1000;
        //const int max_num_ground_warnings = 50;
        const int max_num_ground_warnings = 25;

        if(num_ground_warnings < max_num_ground_warnings)
        {
            log_warning() << "Did not find a ground plane, keeping previous estimate." << std::endl;
            num_ground_warnings += 1;
        }
        else if(num_ground_warnings == max_num_ground_warnings)
        {
            log_warning() << "Warned too many times about problems finding ground plane, going silent." << std::endl;
            num_ground_warnings += 1;
        }
        else
        {
            // we do nothing
        }

        const float weight = 1.0;
        // we keep previous estimate
        set_ground_plane_estimate(estimated_ground_plane, weight);


        //        const bool save_failures_v_disparity_image = false;
        //        if(save_failures_v_disparity_image)
        //        {
        //            const std::string filename = boost::str(
        //                        boost::format("failure_v_disparity_%i.png") % num_ground_plane_estimation_failures );
        //            log_info() << "Created image " << filename << std::endl;
        //            boost::gil::png_write_view(filename, v_disparity_image_view);
        //        }
    }

    return;
} // end of FastGroundPlaneEstimator::estimate_ground_plane


} // end of namespace doppia
