#include "ImagePlaneStixelsEstimator.hpp"

#include "FastStixelsEstimator.hpp" // for helper functions

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/AlignedImage.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include "helpers/simd_intrisics_types.hpp"

#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"

#include "opencv2/imgproc/imgproc.hpp"


#include <stdexcept>
#include <cstdio>
#include <limits>
#include <cstdlib> // for abs<int>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ImagePlaneStixelsEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ImagePlaneStixelsEstimator");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "ImagePlaneStixelsEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ImagePlaneStixelsEstimator");
}

} // end of anonymous namespace


namespace doppia {

using namespace boost;

boost::program_options::options_description ImagePlaneStixelsEstimator::get_args_options()
{
    boost::program_options::options_description desc("ImagePlaneStixelsEstimator options");

    desc.add_options()

            ("stixel_world.num_row_steps",
             boost::program_options::value<int>()->default_value(15),
             "how many row steps (row stripes) below the horizon are used to search the ground-object boundary? "
             "Values between 5 and half the image height are reasonable.")

            ("stixel_world.estimate_object_bottom",
             boost::program_options::value<bool>()->default_value(true),
             "try to estimate the stixel bottom using the image gradient or simply use blind row bands of equal height.")

            ;

    return desc;
}



ImagePlaneStixelsEstimator::ImagePlaneStixelsEstimator(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width)
    : BaseStixelsEstimator(camera, expected_object_height, minimum_object_height_in_pixels, stixel_width),
      disparity_offset(camera.get_calibration().get_disparity_offset_x()),
      num_disparities(get_option_value<int>(options, "max_disparity")),
      num_row_steps(std::max(1, get_option_value<int>(options, "stixel_world.num_row_steps"))),
      should_estimate_stixel_bottom(get_option_value<bool>(options, "stixel_world.estimate_object_bottom")),
      stixel_support_width(std::max(1, get_option_value<int>(options, "stixel_world.stixel_support_width")))
{
    ground_cost_weight = get_option_value<float>(options, "stixel_world.ground_cost_weight");
    ground_cost_threshold = get_option_value<float>(options, "stixel_world.ground_cost_threshold");

    if(ground_cost_threshold >= 1)
    {
        throw std::invalid_argument("stixel_world.ground_cost_threshold should be < 1");
    }

    u_disparity_boundary_diagonal_weight = get_option_value<float>(options, "stixel_world.u_disparity_boundary_diagonal_weight");

    return;
}


ImagePlaneStixelsEstimator::~ImagePlaneStixelsEstimator()
{
    // nothing to do here
    return;
}


void ImagePlaneStixelsEstimator::set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right)
{

    assert( left.dimensions() == right.dimensions() );

    input_left_view = left;
    input_right_view = right;

    typedef input_image_const_view_t::point_t point_t;
    const point_t transposed_dimensions(left.height(), left.width());

    if(stixel_support_width == 1)
    {
        //const size_t num_stixels = left.width()/stixel_width;

        // we use this size to match the gil::subsampled_view dimensions
        const size_t slim_width = (left.width() + (stixel_width -1)) / stixel_width;

        const point_t transposed_slim_dimensions(left.height(), slim_width); //num_stixels);

        if(transposed_slim_left_image_p == false)
        {
            transposed_slim_left_image_p.reset(new AlignedImage(transposed_slim_dimensions));
        }

        assert(transposed_slim_left_image_p->dimensions() == transposed_slim_dimensions );
    }
    else
    {
        if(transposed_left_image_p == false)
        {
            transposed_left_image_p.reset(new AlignedImage(transposed_dimensions));
        }

        assert(transposed_left_image_p->dimensions() == transposed_dimensions );

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

    assert(transposed_right_image_p->dimensions() == transposed_dimensions );
    assert(transposed_rectified_right_image_p->dimensions() == transposed_dimensions );
    assert(rectified_right_image_p->dimensions() == left.dimensions() );

    return;
}


void ImagePlaneStixelsEstimator::set_ground_plane_estimate(
        const GroundPlane &ground_plane,
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


void ImagePlaneStixelsEstimator::compute()
{

    // for each column, and for each row_step find the most likely object bottom --
    find_stixels_bottom_candidates();

    // for each column, and for each row_step; compute the ground and object evidence --
    collect_stereo_evidence();

    // estimate the distance using dynamic programming --
    estimate_stixels_bottom();

    return;
}


const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ImagePlaneStixelsEstimator::get_object_cost_per_stixel_and_row_step() const
{
    return object_cost_per_stixel_and_row_step;
}


const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ImagePlaneStixelsEstimator::get_ground_cost_per_stixel_and_row_step() const
{
    return ground_cost_per_stixel_and_row_step;
}


const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ImagePlaneStixelsEstimator::get_cost_per_stixel_and_row_step() const
{
    return cost_per_stixel_and_row_step;
}


const std::vector<int> &ImagePlaneStixelsEstimator::get_stixel_and_row_step_ground_obstacle_boundary() const
{
    return stixel_and_row_step_ground_obstacle_boundary;
}


void compute_y_derivative(cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    _dst.create( src.size(), CV_MAKETYPE(CV_16S, src.channels()) );
    cv::Mat dst = _dst.getMat();

    static cv::Ptr<cv::FilterEngine> dy_filter;

    if(dy_filter.empty())
    {
        const cv::Mat dy_kernel = (cv::Mat_<boost::int8_t>(3, 1) << -1, 0, 1);
        dy_filter = cv::createLinearFilter(src.type(), dst.type(), dy_kernel);
    }

    dy_filter->apply(src, dst);

    return;
}



void ImagePlaneStixelsEstimator::find_stixels_bottom_candidates()
{
    // copy the input data into the memory aligned data structures

    if(transposed_slim_left_image_p)
    {
        assert(stixel_support_width == 1);
        copy_pixels(gil::transposed_view( gil::subsampled_view(input_left_view, stixel_width, 1) ),
                    transposed_slim_left_image_p->get_view());
    }
    else
    {
        copy_pixels(gil::transposed_view(input_left_view), transposed_left_image_p->get_view());
    }

    if(should_estimate_stixel_bottom)
    {
        // v0 reaches 130 Hz, v1 reaches 148 Hz
        //find_stixels_bottom_candidates_v0_baseline();
        find_stixels_bottom_candidates_v1_compute_only_what_is_used();
    }
    else
    {
        set_fix_stixels_bottom_candidates();
    }

    return;
}


void ImagePlaneStixelsEstimator::find_stixels_bottom_candidates_v0_baseline()
{
    gil::opencv::ipl_image_wrapper left_wrap = gil::opencv::create_ipl_image(input_left_view);
    cv::Mat left_mat(left_wrap.get());
    cv::cvtColor(left_mat, gray_left_mat, CV_RGB2GRAY);

    compute_y_derivative(gray_left_mat, df_dy_mat);


    if(df_dy_mat.type() != CV_16SC1)
    {
        log_error() << "df_dy_mat.type() == " << df_dy_mat.type() << std::endl;
        log_error() << "df_dy_mat.depth() == " << df_dy_mat.depth() << std::endl;
        log_error() << "df_dy_mat.channels() == " << df_dy_mat.channels() << std::endl;
        throw std::runtime_error("compute_y_derivative returned matrices of unexpected type");
    }


    const size_t
            horizon_row = v_given_disparity[0],
            stixels_rows = input_left_view.height() - horizon_row,
            num_stixels = input_left_view.width() / stixel_width,
            row_step_size = stixels_rows / num_row_steps,
            input_width = input_left_view.width(),
            input_height = input_left_view.height();

    if((bottom_v_given_stixel_and_row_step.shape()[0] != num_stixels) or
       (bottom_v_given_stixel_and_row_step.shape()[1] != num_row_steps))
    {
        // row_given_... and disparity_given... are kept together
        bottom_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        top_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        disparity_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
    }

    // we use the left pixel of each stixel column to choose the height
    // FIXME: should the most confident pixel from all the columns covered by the stixel
    for(size_t column=0, stixel_index = 0; column < input_width; column += stixel_width, stixel_index += 1)
    {
        size_t row=horizon_row;
        for(size_t row_step_index=0; row_step_index < num_row_steps; row_step_index +=1)
        {

            int max_dy_value = 0;
            size_t max_dy_row = row; // current row

            for(size_t delta_row=0; (delta_row < row_step_size) and (row < input_height); delta_row +=1, row +=1)
            {
#if GCC_VERSION > 40400
	      const boost::int16_t dy_value = std::abs<boost::int16_t>(df_dy_mat.at<boost::int16_t>(row, column));
#else
	      const boost::int16_t dy_value = std::abs<float>(df_dy_mat.at<boost::int16_t>(row, column));
#endif
                if(dy_value > max_dy_value)
                {
                    max_dy_value = dy_value;
                    max_dy_row = row;
                }
            } // end of "for each row in the row step"


            bottom_v_given_stixel_and_row_step[stixel_index][row_step_index] = max_dy_row;
            const int disparity = disparity_given_v[max_dy_row];
            disparity_given_stixel_and_row_step[stixel_index][row_step_index] = disparity + disparity_offset;
            top_v_given_stixel_and_row_step[stixel_index][row_step_index] = \
                    top_v_for_stixel_estimation_given_disparity[disparity];

        } // end of "for each row step"

    } // end of "for each stixel left column"

    return;
}


inline
void find_stixels_bottom_candidates_v1(
        const size_t left_u,
        const size_t stixel_index,
        const size_t num_row_steps,
        const float row_step_size,
        const int disparity_offset,
        const size_t horizon_row,
        const AlignedImage::const_view_t &transposed_left_view,
        const std::vector<int> &disparity_given_v,
        const std::vector<int> &top_v_for_stixel_estimation_given_disparity,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t &bottom_v_given_stixel_and_row_step,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t &top_v_given_stixel_and_row_step,
        ImagePlaneStixelsEstimator::disparity_given_stixel_and_row_step_t &disparity_given_stixel_and_row_step)
{

    const AlignedImage::const_view_t::x_iterator
            left_u_column_begin_it = transposed_left_view.row_begin(left_u),
            left_u_column_end_it = transposed_left_view.row_end(left_u);

    size_t row = std::max<size_t>(1, horizon_row);
    const size_t previous_row = row - 1, next_row = row + 1;
    AlignedImage::const_view_t::x_iterator
            previous_row_it = left_u_column_begin_it + previous_row,
            next_row_it = left_u_column_begin_it + next_row;

    for(size_t row_step_index=0; row_step_index < num_row_steps; row_step_index +=1)
    {
        boost::uint16_t max_dy_value = 0;
        size_t max_dy_row = previous_row + 1; // current row

        const size_t max_row_at_current_step = horizon_row + static_cast<size_t>(row_step_index*row_step_size);

        //for(size_t delta_row=0; (delta_row < row_step_size) and (next_row_it != left_u_column_end_it);
        //    delta_row +=1, row+=1, ++previous_row_it, ++next_row_it)
        for(; (row < max_row_at_current_step) and (next_row_it != left_u_column_end_it);
            row+=1, ++previous_row_it, ++next_row_it)
        {
            //AlignedImage::const_view_t::value_type
            const boost::gil::rgb8c_pixel_t &previous_pixel = *(previous_row_it), &next_pixel = *(next_row_it);
            const boost::uint16_t dy_value = sad_cost_uint16(previous_pixel, next_pixel);

            if(dy_value > max_dy_value)
            {
                max_dy_value = dy_value;
                max_dy_row = row;
            }
        } // end of "for each row in the row step"

        assert(max_dy_row < transposed_left_view.width());

        bottom_v_given_stixel_and_row_step[stixel_index][row_step_index] = max_dy_row;
        const int disparity = disparity_given_v[max_dy_row];
        disparity_given_stixel_and_row_step[stixel_index][row_step_index] = disparity + disparity_offset;
        top_v_given_stixel_and_row_step[stixel_index][row_step_index] = \
                top_v_for_stixel_estimation_given_disparity[disparity];

    } // end of "for each row step"

    return;
}


void ImagePlaneStixelsEstimator::find_stixels_bottom_candidates_v1_compute_only_what_is_used()
{

    const size_t
            horizon_row = v_given_disparity[0],
            stixels_rows = input_left_view.height() - horizon_row,
            //row_step_size = stixels_rows / num_row_steps,
            num_stixels = input_left_view.width() / stixel_width,
            input_width = input_left_view.width();
    //input_height = input_left_view.height();

    // needs to be float when num_row_steps is a high number
    const float row_step_size = stixels_rows / static_cast<float>(num_row_steps);

    if((bottom_v_given_stixel_and_row_step.shape()[0] != num_stixels) or
       (bottom_v_given_stixel_and_row_step.shape()[1] != num_row_steps))
    {
        // row_given_... and disparity_given... are kept together
        bottom_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        top_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        disparity_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
    }

    if(transposed_slim_left_image_p)
    {
        const AlignedImage::const_view_t &transposed_slim_left_view = transposed_slim_left_image_p->get_view();

        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        {
            // we use the left pixel of each stixel column to choose the height
            const size_t left_u = stixel_index; // each column correspond to one stixel

            find_stixels_bottom_candidates_v1(left_u, stixel_index,
                                              num_row_steps, row_step_size,
                                              disparity_offset, horizon_row,
                                              transposed_slim_left_view,
                                              disparity_given_v,
                                              top_v_for_stixel_estimation_given_disparity,
                                              bottom_v_given_stixel_and_row_step,
                                              top_v_given_stixel_and_row_step,
                                              disparity_given_stixel_and_row_step);
        } // end of "for each stixel left column"
    }
    else
    {
        const AlignedImage::const_view_t &transposed_left_view = transposed_left_image_p->get_view();

        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        {
            // we use the left pixel of each stixel column to choose the height
            // FIXME: should we use the most confident pixel from all the columns covered by the stixel ?
            const size_t left_u = stixel_index * stixel_width;

            if(left_u >= input_width)
            {
                // reached the image border
                continue; // no breaks allowed inside omp parallel
            }

            find_stixels_bottom_candidates_v1(left_u, stixel_index,
                                              num_row_steps, row_step_size,
                                              disparity_offset, horizon_row,
                                              transposed_left_view,
                                              disparity_given_v,
                                              top_v_for_stixel_estimation_given_disparity,
                                              bottom_v_given_stixel_and_row_step,
                                              top_v_given_stixel_and_row_step,
                                              disparity_given_stixel_and_row_step);
        } // end of "for each stixel left column"
    }
    return;
}


/// dumb version of find_stixels_bottom_candidates that set the bottom candidates at fix positions
void ImagePlaneStixelsEstimator::set_fix_stixels_bottom_candidates()
{
    const size_t
            horizon_row = v_given_disparity[0],
            stixels_rows = input_left_view.height() - horizon_row,
            //row_step_size = stixels_rows / num_row_steps,
            num_stixels = input_left_view.width() / stixel_width,
            input_width = input_left_view.width();
    //input_height = input_left_view.height();

    // needs to be float when num_row_steps is a high number
    const float row_step_size = stixels_rows / static_cast<float>(num_row_steps);

    if((bottom_v_given_stixel_and_row_step.shape()[0] != num_stixels) or
       (bottom_v_given_stixel_and_row_step.shape()[1] != num_row_steps))
    {
        // row_given_... and disparity_given... are kept together
        bottom_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        top_v_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
        disparity_given_stixel_and_row_step.resize(boost::extents[num_stixels][num_row_steps]);
    }

    // guided schedule seeems to provide the best performance (better than default and static)
    //#pragma omp parallel for schedule(guided)
    for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
    {
        // we use the left pixel of each stixel column to choose the height
        // FIXME: should we use the most confident pixel from all the columns covered by the stixel ?
        const size_t left_u = stixel_index * stixel_width;

        if(left_u >= input_width)
        {
            // reached the image border
            continue; // no breaks allowed inside omp parallel
        }

        for(size_t row_step_index=0; row_step_index < num_row_steps; row_step_index +=1)
        {
            const size_t max_row_at_current_step = horizon_row + static_cast<size_t>(row_step_index*row_step_size);

            assert(max_row_at_current_step < input_left_view.height());
            bottom_v_given_stixel_and_row_step[stixel_index][row_step_index] = max_row_at_current_step;
            disparity_given_stixel_and_row_step[stixel_index][row_step_index] = disparity_given_v[max_row_at_current_step];
        } // end of "for each row step"

    } // end of "for each stixel left column"

    return;
}


void ImagePlaneStixelsEstimator::collect_stereo_evidence()
{
    // copy the input data into the memory aligned data structures
    // (left view as transposed in find_stixels_bottom_candidates)
    copy_pixels(gil::transposed_view(input_right_view), transposed_right_image_p->get_view());

    assert(bottom_v_given_stixel_and_row_step.empty() == false);
    assert(top_v_given_stixel_and_row_step.empty() == false);


    // Here (u,v) refers to the 2d image plane, just like (x,y) or (cols, rows)
    const size_t
            num_rows = input_left_view.height(),
            //num_columns = input_left_view.width(),
            num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];

    if(v_given_disparity.size() != static_cast<size_t>(num_disparities) or
       disparity_given_v.size() != num_rows)
    {
        throw std::runtime_error("ImagePlaneStixelsEstimator::compute_disparity_space_cost "
                                 "called before FastStixelsEstimator::set_v_disparity_line_bidirectional_maps");
    }

    // reset and resize the object_cost and ground_cost
    // Eigen::MatrixXf::Zero(rows, cols)
    cost_per_stixel_and_row_step_t
            &object_cost = object_cost_per_stixel_and_row_step,
            &ground_cost = ground_cost_per_stixel_and_row_step;
    //object_cost = Eigen::MatrixXf::Zero(num_stixels, num_row_steps);
    // we use max instead of zero to avoid problems in the lower left corner of the matrix
    object_cost = Eigen::MatrixXf::Constant(num_stixels, num_row_steps, std::numeric_limits<float>::max());
    ground_cost = Eigen::MatrixXf::Zero(num_stixels, num_row_steps);


    compute_transposed_rectified_right_image();

    {
        // it seems that computing one cost and then the next one is slightly faster
        // than computing both in the same time (probably because of cache streaming usage)
        compute_object_cost(object_cost);
        compute_ground_cost(ground_cost);
    }


    // post filtering steps --
    {
        //post_process_object_cost(object_cost);
        //post_process_ground_cost(ground_cost, num_rows);
    }

    // set the final cost --
    cost_per_stixel_and_row_step = object_cost + ground_cost;

    // mini fix to the "left area initialization issue"
    //fix_cost_per_stixel_and_row_step();

    return;
}


void ImagePlaneStixelsEstimator::compute_transposed_rectified_right_image()
{
    doppia::compute_transposed_rectified_right_image(
                input_right_view,
                rectified_right_image_p->get_view(),
                transposed_rectified_right_image_p->get_view(),
                disparity_offset,
                v_given_disparity,
                disparity_given_v);

    return;
}


/// in this case we also know stixel_support_width == 1
inline
void compute_object_cost_stixel_using_slim_left(
        const int stixel_index,
        const int stixel_width,
        const size_t num_row_steps,
        const int num_columns,
        const int disparity_offset,
        const AlignedImage::const_view_t &transposed_left_view,
        const AlignedImage::const_view_t &transposed_right_view,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        bottom_v_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        top_v_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::disparity_given_stixel_and_row_step_t::const_reference
        disparity_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &object_cost)
{
    //const bool use_simd = false;
    const bool use_simd = true;

    for(size_t row_step_index = 0; row_step_index < num_row_steps; row_step_index += 1)
    {
        // precomputed_v_disparity_line already checked for >=0 and < num_rows
        const size_t
                bottom_v = bottom_v_given_stixel_and_row_step_at_stixel_index[row_step_index],
                top_v = top_v_given_stixel_and_row_step_at_stixel_index[row_step_index];

        // here d already includes the disparity_offset
        const int d = disparity_given_stixel_and_row_step_at_stixel_index[row_step_index];

        float &t_object_cost = object_cost(stixel_index, row_step_index);

        uint32_t num_pixels_summed = 0;

        const int left_u = stixel_index;
        assert((left_u  >= 0) and (left_u < num_columns));

        // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
        const int right_u = stixel_index*stixel_width - d;
        if(right_u < 0)
        {
            // disparity is too large for the current column
            // cost left to zero
            continue;
        }

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_left_view.row_begin(left_u),
                right_column_begin_it = transposed_right_view.row_begin(right_u);

        // for each (u, disparity) value accumulate over the vertical axis --

        const AlignedImage::const_view_t::x_iterator
                left_begin_it = left_column_begin_it + top_v,
                left_end_it = left_column_begin_it + bottom_v,
                right_begin_it = right_column_begin_it + top_v;
        assert(left_begin_it <= left_end_it);

        AlignedImage::const_view_t::x_iterator
                left_it = left_begin_it, right_it = right_begin_it;

        // from tentative ground upwards, over the object -

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

        num_pixels_summed += (bottom_v - top_v);
        assert(num_pixels_summed > 0); // since minimum_v_for_disparity was computed to ensure a minimum height

        // normalize the object cost -
        t_object_cost /= num_pixels_summed;
        assert(t_object_cost >= 0);
    } // end of "for each row step"

    return;
}


inline
void compute_object_cost_with_any_stixel_support_width(
        const size_t stixel_left_u,
        const size_t stixel_index,
        const size_t num_row_steps,
        const int stixel_support_width,
        const int num_columns,
        const AlignedImage::const_view_t &transposed_left_view,
        const AlignedImage::const_view_t &transposed_right_view,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        bottom_v_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        top_v_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::disparity_given_stixel_and_row_step_t::const_reference
        disparity_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &object_cost)
{

    //const bool use_simd = false;
    const bool use_simd = true;


    for(size_t row_step_index = 0; row_step_index < num_row_steps; row_step_index += 1)
    {

        // precomputed_v_disparity_line already checked for >=0 and < num_rows
        const size_t
                bottom_v = bottom_v_given_stixel_and_row_step_at_stixel_index[row_step_index],
                top_v = top_v_given_stixel_and_row_step_at_stixel_index[row_step_index];

        // here d already includes the disparity_offset
        const int d = disparity_given_stixel_and_row_step_at_stixel_index[row_step_index];

        float &t_object_cost = object_cost(stixel_index, row_step_index);

        uint32_t num_pixels_summed = 0;

        // FIXME if stixel_support_width is > 2, should make it symmetric (e.g. -1,0,1)
        for(int delta_u = 0; delta_u < stixel_support_width; delta_u += 1)
        {
            const int left_u = stixel_left_u + delta_u;
            if((left_u  < 0) or (left_u >= num_columns))
            {
                // we skip this column
                continue;
            }

            // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)
            const int right_u = left_u - d;
            if(right_u < 0)
            {
                // disparity is too large for the current column
                // cost left to zero
                continue;
            }

            const AlignedImage::const_view_t::x_iterator
                    left_column_begin_it = transposed_left_view.row_begin(left_u),
                    right_column_begin_it = transposed_right_view.row_begin(right_u);

            // for each (u, disparity) value accumulate over the vertical axis --

            const AlignedImage::const_view_t::x_iterator
                    left_begin_it = left_column_begin_it + top_v,
                    left_end_it = left_column_begin_it + bottom_v,
                    right_begin_it = right_column_begin_it + top_v;
            assert(left_begin_it <= left_end_it);

            AlignedImage::const_view_t::x_iterator
                    left_it = left_begin_it, right_it = right_begin_it;

            // from tentative ground upwards, over the object -

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

            num_pixels_summed += (bottom_v - top_v);

        } // end of "for each column in the stixel"

        // normalize the object cost -
        if(num_pixels_summed > 0)
        {
            t_object_cost /= num_pixels_summed;
        }
        assert(t_object_cost >= 0);

    } // end of "for each row step"

    return;
}


void ImagePlaneStixelsEstimator::compute_object_cost(cost_per_stixel_and_row_step_t &object_cost) const
{
    //const int num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const size_t num_disparities = this->num_disparities;
    const size_t num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];

    if(transposed_slim_left_image_p)
    {
        assert(stixel_support_width == 1);

        const AlignedImage::const_view_t
                transposed_slim_left_view = transposed_slim_left_image_p->get_view(),
                transposed_right_view = transposed_right_image_p->get_view();

        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        { // iterate over the stixels

            row_given_stixel_and_row_step_t::const_reference
                    bottom_v_given_stixel_and_row_step_at_stixel_index = bottom_v_given_stixel_and_row_step[stixel_index],
                    top_v_given_stixel_and_row_step_at_stixel_index = top_v_given_stixel_and_row_step[stixel_index];

            disparity_given_stixel_and_row_step_t::const_reference
                    disparity_given_stixel_and_row_step_at_stixel_index = disparity_given_stixel_and_row_step[stixel_index];

            compute_object_cost_stixel_using_slim_left(
                        stixel_index,
                        stixel_width,
                        num_row_steps,
                        num_columns, disparity_offset,
                        transposed_slim_left_view, transposed_right_view,
                        bottom_v_given_stixel_and_row_step_at_stixel_index,
                        top_v_given_stixel_and_row_step_at_stixel_index,
                        disparity_given_stixel_and_row_step_at_stixel_index,
                        object_cost);

        } // end of "for each stixel"
    }
    else
    { // transposed_slim_left_image_p == false, thus transposed_left_image_p == true

        const AlignedImage::const_view_t
                transposed_left_view = transposed_left_image_p->get_view(),
                transposed_right_view = transposed_right_image_p->get_view();

        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        { // iterate over the stixels

            const size_t stixel_left_u = stixel_index*stixel_width;

            row_given_stixel_and_row_step_t::const_reference
                    bottom_v_given_stixel_and_row_step_at_stixel_index = bottom_v_given_stixel_and_row_step[stixel_index],
                    top_v_given_stixel_and_row_step_at_stixel_index = top_v_given_stixel_and_row_step[stixel_index];

            disparity_given_stixel_and_row_step_t::const_reference
                    disparity_given_stixel_and_row_step_at_stixel_index = disparity_given_stixel_and_row_step[stixel_index];

            compute_object_cost_with_any_stixel_support_width(
                        stixel_left_u,
                        stixel_index,
                        num_row_steps, stixel_support_width,
                        num_columns,
                        transposed_left_view, transposed_right_view,
                        bottom_v_given_stixel_and_row_step_at_stixel_index,
                        top_v_given_stixel_and_row_step_at_stixel_index,
                        disparity_given_stixel_and_row_step_at_stixel_index,
                        object_cost);

        } // end of "for each stixel"

    }

    return;
}


// compute the raw SAD (without doing a division by the number of channels)
inline uint16_t simd_sad_cost_uint16(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b)
{
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


/// in this case we also know stixel_support_width == 1
inline
void compute_ground_cost_using_slim_left(
        const size_t stixel_index,
        const int stixel_width,
        const size_t num_row_steps,
        const int num_columns, const size_t num_rows,
        const AlignedImage::const_view_t &transposed_slim_left_view,
        const AlignedImage::const_view_t &transposed_rectified_right_view,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        row_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ground_cost)
{

    // in the current code, the simd version is slower that without simd
    //(which means improper use of simd and/or good auto-vectorization)
    const bool use_simd = false;
    //const bool use_simd = true;

    for(size_t row_step_index = 0; row_step_index < num_row_steps; row_step_index += 1)
    {
        // precomputed_v_disparity_line already checked for >=0 and < num_rows
        const size_t
                row = row_given_stixel_and_row_step_at_stixel_index[row_step_index],
                ground_obstacle_v_boundary_at_d = row;

        assert(ground_obstacle_v_boundary_at_d < num_rows);

        uint32_t column_cumulative_sum = 0;
        uint32_t num_pixels_summed = 0;

        const int left_u = stixel_index;
        assert((left_u  >= 0) or (left_u < num_columns));

        const int right_u = stixel_index*stixel_width;
        assert((right_u  >= 0) or (right_u < num_columns));

        const AlignedImage::const_view_t::x_iterator
                left_column_begin_it = transposed_slim_left_view.row_begin(left_u),
                left_column_end_it = transposed_slim_left_view.row_end(left_u),
                //right_column_begin_it = transposed_rectified_right_view.row_begin(right_u),
                right_column_end_it = transposed_rectified_right_view.row_end(right_u);

        // we are going to move in reverse from d = num_disparities - 1 to d = 0
        const AlignedImage::const_view_t::x_iterator
                left_begin_it = left_column_end_it - 1,
                right_begin_it = right_column_end_it - 1,
                left_step_end_it = left_column_begin_it + ground_obstacle_v_boundary_at_d;

        // we cumulate the sum from the bottom of the image upwards
        AlignedImage::const_view_t::x_iterator
                left_it = left_begin_it,
                right_it = right_begin_it;
        assert(left_step_end_it <= left_it);

        for(; left_it != left_step_end_it; --left_it, --right_it)
        {
            if(use_simd)
            {
                // FIXME for some unknown reason enabling this code messes up the ground plane estimation
                // it does compute the desired value, but it triggers a missbehaviour "somewhere else"
                // very weird simd related bug... (that appears on my laptop but not on my desktop)
                column_cumulative_sum += simd_sad_cost_uint16(*left_it, *right_it);
            }
            else
            {
                column_cumulative_sum += sad_cost_uint16(*left_it, *right_it);
            }
        }

        num_pixels_summed += 3*(num_rows - ground_obstacle_v_boundary_at_d); // 3 because of RGB


        // normalize and set the ground cost -
        float &ground_cost_float = ground_cost(stixel_index, row_step_index);
        ground_cost_float = column_cumulative_sum;
        if(num_pixels_summed > 0)
        {
            ground_cost_float /= num_pixels_summed;
        }
        assert(ground_cost_float >= 0);

    } // end of "for each row step"


    return;
}


inline
void compute_ground_cost_with_any_stixel_support_width(
        const size_t stixel_left_u,
        const size_t stixel_index,
        const size_t num_row_steps,
        const int stixel_support_width,
        const int num_columns, const size_t num_rows,
        const AlignedImage::const_view_t &transposed_left_view,
        const AlignedImage::const_view_t &transposed_rectified_right_view,
        ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t::const_reference
        row_given_stixel_and_row_step_at_stixel_index,
        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ground_cost)
{

    // in the current code, the simd version is slower that without simd
    //(which means improper use of simd and/or good auto-vectorization)
    const bool use_simd = false;
    //const bool use_simd = true;

    for(size_t row_step_index = 0; row_step_index < num_row_steps; row_step_index += 1)
    {
        // precomputed_v_disparity_line already checked for >=0 and < num_rows
        const size_t
                row = row_given_stixel_and_row_step_at_stixel_index[row_step_index],
                ground_obstacle_v_boundary_at_d = row;

        assert(ground_obstacle_v_boundary_at_d < num_rows);

        uint32_t column_cumulative_sum = 0;
        uint32_t num_pixels_summed = 0;

        // FIXME if stixel_support_width is > 2, should make it symmetric (e.g. -1,0,1)
        for(int delta_u = 0; delta_u < stixel_support_width; delta_u += 1)
        {
            const int left_u = stixel_left_u + delta_u;
            if((left_u  < 0) or (left_u >= num_columns))
            {
                // we skip this column
                continue;
            }

            const AlignedImage::const_view_t::x_iterator
                    left_column_begin_it = transposed_left_view.row_begin(left_u),
                    left_column_end_it = transposed_left_view.row_end(left_u),
                    //right_column_begin_it = transposed_rectified_right_view.row_begin(left_u),
                    right_column_end_it = transposed_rectified_right_view.row_end(left_u);

            // we are going to move in reverse from d = num_disparities - 1 to d = 0
            const AlignedImage::const_view_t::x_iterator
                    left_begin_it = left_column_end_it - 1,
                    right_begin_it = right_column_end_it - 1,
                    left_step_end_it = left_column_begin_it + ground_obstacle_v_boundary_at_d;

            // we cumulate the sum from the bottom of the image upwards
            AlignedImage::const_view_t::x_iterator
                    left_it = left_begin_it,
                    right_it = right_begin_it;
            assert(left_step_end_it <= left_it);

            for(; left_it != left_step_end_it; --left_it, --right_it)
            {
                if(use_simd)
                {
                    // FIXME for some unknown reason enabling this code messes up the ground plane estimation
                    // it does compute the desired value, but it triggers a missbehaviour "somewhere else"
                    // very weird simd related bug... (that appears on my laptop but not on my desktop)
                    column_cumulative_sum += simd_sad_cost_uint16(*left_it, *right_it);
                }
                else
                {
                    column_cumulative_sum += sad_cost_uint16(*left_it, *right_it);
                }
            }

            num_pixels_summed += 3*(num_rows - ground_obstacle_v_boundary_at_d); // 3 because of RGB

        } // end of "for each column in the stixel"

        // normalize and set the ground cost -
        float &ground_cost_float = ground_cost(stixel_index, row_step_index);
        ground_cost_float = column_cumulative_sum;
        if(num_pixels_summed > 0)
        {
            ground_cost_float /= num_pixels_summed;
        }
        assert(ground_cost_float >= 0);

    } // end of "for each row step"


    return;
}


void ImagePlaneStixelsEstimator::compute_ground_cost(ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ground_cost) const
{

    const size_t num_rows = input_left_view.height();
    const int num_columns = input_left_view.width();
    //const size_t num_disparities = this->num_disparities;
    const size_t num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];

    if(transposed_slim_left_image_p)
    {
        assert(stixel_support_width == 1);

        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        { // iterate over the stixels

            const AlignedImage::const_view_t
                    transposed_slim_left_view = transposed_slim_left_image_p->get_view(),
                    transposed_rectified_right_view = transposed_rectified_right_image_p->get_view();

            row_given_stixel_and_row_step_t::const_reference
                    row_given_stixel_and_row_step_at_stixel_index = bottom_v_given_stixel_and_row_step[stixel_index];

            compute_ground_cost_using_slim_left(
                        stixel_index,
                        stixel_width,
                        num_row_steps,
                        num_columns, num_rows,
                        transposed_slim_left_view, transposed_rectified_right_view,
                        row_given_stixel_and_row_step_at_stixel_index,
                        ground_cost);

        } // end of "for each stixel"

    }
    else
    { // transposed_slim_left_image_p == false, thus transposed_left_image_p == true


        // guided schedule seeems to provide the best performance (better than default and static)
#pragma omp parallel for schedule(guided)
        for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
        { // iterate over the stixels

            const size_t stixel_left_u = stixel_index*stixel_width;
            // a pixel (x,y) on the left image should be matched on the right image on the range ([0,x],y)

            row_given_stixel_and_row_step_t::const_reference
                    row_given_stixel_and_row_step_at_stixel_index = bottom_v_given_stixel_and_row_step[stixel_index];

            compute_ground_cost_with_any_stixel_support_width(
                        stixel_left_u,
                        stixel_index,
                        num_row_steps,
                        stixel_support_width,
                        num_columns, num_rows,
                        transposed_left_image_p->get_view(),
                        transposed_rectified_right_image_p->get_view(),
                        row_given_stixel_and_row_step_at_stixel_index,
                        ground_cost);

        } // end of "for each stixel"

    }


    return;
}


void ImagePlaneStixelsEstimator::estimate_stixels_bottom()
{

    const bool use_simplest_thing_that_could_ever_work = false;
    if(use_simplest_thing_that_could_ever_work)
    {
        estimate_stixels_bottom_using_argmax_per_stixel();
    }
    else
    {
        estimate_stixels_bottom_using_dynamic_programming();
    }
    return;
}


void ImagePlaneStixelsEstimator::estimate_stixels_bottom_using_argmax_per_stixel()
{
    // to make things simple we always output one stixel per column
    // we store the object bottom for each image column
    u_v_ground_obstacle_boundary.resize(input_left_view.width());

    const size_t num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];
    stixel_and_row_step_ground_obstacle_boundary.resize(num_stixels);

    const cost_per_stixel_and_row_step_t &cost = cost_per_stixel_and_row_step;

    for(size_t stixel_index=0; stixel_index < num_stixels; stixel_index += 1)
    {
        // find the minimum cost row_step
        int min_cost_row_step = 0;
        cost.row(stixel_index).minCoeff(&min_cost_row_step);

        stixel_and_row_step_ground_obstacle_boundary[stixel_index] = min_cost_row_step;

        const row_t row = bottom_v_given_stixel_and_row_step[stixel_index][min_cost_row_step];

        for(size_t stixel_u = stixel_index*stixel_width;
            (stixel_u < (stixel_index+1)*stixel_width) and (stixel_u < u_v_ground_obstacle_boundary.size());
            stixel_u += 1)
        {
            u_v_ground_obstacle_boundary[stixel_u] = row;
        } // end of "for each column covered by the stixel

    } // end of "for each stixel"


    // use u_v_ground_obstacle_boundary to build the final stixels output
    u_v_disparity_boundary_to_stixels();
    return;
}


inline
void ImagePlaneStixelsEstimator::estimate_stixels_bottom_using_dynamic_programming()
{
    // v0 uses backtracking for the second pass
    // (which on StixelsEstimator was slower than recomputing everything)
    estimate_stixels_bottom_using_dynamic_programming_v0_backtracking();
    return;
}


void ImagePlaneStixelsEstimator::estimate_stixels_bottom_using_dynamic_programming_v0_backtracking()
{

    const size_t num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];
    const int stixel_width = this->stixel_width;

    stixel_and_row_step_ground_obstacle_boundary.resize(num_stixels);

    {
        // run dynamic programming over the cost image --
        // this is follows the spirit of section III.C of Kubota et al. 2007 paper,
        // but we operate in the space of stixels versus row_step instead of
        // u-disparity, which changes the contraints handling.

        const int
                num_stixels = cost_per_stixel_and_row_step.rows(),
                num_row_steps = cost_per_stixel_and_row_step.cols();

        const float diagonal_weight = u_disparity_boundary_diagonal_weight;

        const cost_per_stixel_and_row_step_t &const_M_cost = M_cost;

        if((min_M_minus_c_indices.shape()[1] != static_cast<size_t>(num_row_steps))
           or (min_M_minus_c_indices.shape()[0] != static_cast<size_t>(num_stixels)))
        {
            min_M_minus_c_indices.resize(boost::extents[num_stixels][num_row_steps]);
            // all accessed values are set, so there is no need to initialize
        }


        // right to left pass --

        // Kubota et al. 2007 penalizes using object cost
        // this is completelly arbritrary, we use here
        // object_cost + ground_cost

        //const cost_per_stixel_and_row_step_t &c_i_cost = ground_cost_per_stixel_and_row_step;
        const cost_per_stixel_and_row_step_t &c_i_cost = object_cost_per_stixel_and_row_step;
        //const cost_per_stixel_and_row_step_t &c_i_cost = cost_per_stixel_and_row_step;

        {
            // we first copy all m_i(d_i) values
            M_cost = cost_per_stixel_and_row_step;

#pragma omp parallel
            for(int stixel_index = num_stixels - 2; stixel_index >=0; stixel_index -= 1)
            {
                // equation 3 with d_{i-1} replaced by e
                // we do min instead of max because we are using correlation cost
                // M_i(d_i) = m_i(d_i) + min_e[ M_{i-1}(e) - c_i(d_i, e) ]
                min_M_minus_c_indices_t::reference min_M_minus_c_indices_stixel = min_M_minus_c_indices[stixel_index];

                const int next_stixel_index_column = stixel_index + 1;

                // with or without static scheduling we got roughly the same performance
                // however, in this case, "default" seems to be the best choice
#pragma omp for //schedule(static)
                for(int row_step_index=0; row_step_index < num_row_steps; row_step_index+=1)
                {
                    const int d = disparity_given_stixel_and_row_step[stixel_index][row_step_index];
                    float min_M_minus_c = std::numeric_limits<float>::max();
                    int min_M_minus_c_index = 0;

                    // since upper row do mean lower disparities,
                    // we can check safely all pixels "above"
                    {
                        const int e_end = row_step_index;

                        for(int e=0; e < e_end ; e+=1)
                        {
                            // implementing the definition of c_i(d_i,e) at equation 5
                            // c is c_i(d, e);
                            // c = 0
                            //const float t_cost = const_M_cost(next_column, e) - c; // and c = 0;
                            const float t_cost = const_M_cost(next_stixel_index_column, e);

                            // we do min instead of max because we are using correlation cost
                            //min_M_minus_c = std::min(t_cost, min_M_minus_c);
                            if(t_cost < min_M_minus_c)
                            {
                                min_M_minus_c = t_cost;
                                min_M_minus_c_index = e;
                            }
                        } // end of "for each disparity e"
                    }

                    // e == row_step_index
                    // "straigh horizontal line" case
                    {
                        const int e = row_step_index;

                        const float w = -0.5; // FIXME hardcoded test parameter
                        const float c = -w*c_i_cost(stixel_index, e);
                        const float t_cost = const_M_cost(next_stixel_index_column, e) - c;

                        // we do min instead of max because we are using correlation cost
                        //min_M_minus_c = std::min(t_cost, min_M_minus_c);
                        if(t_cost < min_M_minus_c)
                        {
                            min_M_minus_c = t_cost;
                            min_M_minus_c_index = e;
                        }
                    }

                    // we now check next stixel on same row_step or below,
                    // until we cross the disparity diagonal constraint
                    {
                        // we will go over every row, but will break as soon as we cross the row barrier
                        const int e_end = num_row_steps;

                        for(int e = row_step_index + 1; e < e_end ; e+=1)
                        {
                            const float t_cost = const_M_cost(next_stixel_index_column, e);

                            const int next_stixel_d = disparity_given_stixel_and_row_step[next_stixel_index_column][e];

                            const int delta_d = next_stixel_d - d;

                            if(delta_d < stixel_width)
                            { // above the diagonal constraint, same operation as before

                                if(t_cost < min_M_minus_c)
                                {
                                    min_M_minus_c = t_cost;
                                    min_M_minus_c_index = e;
                                }

                            }
                            else //if(delta_d >= stixel_width)
                            { // below or exactly on the diagonal
                                // even if not exactly on the diagonal we apply the diagonal constraint

                                // implementing the definition of c_i(d_i,e) at equation 5
                                // c is c_i(d, e);
                                const float c = -diagonal_weight - c_i_cost(stixel_index, e);
                                //const float c = -diagonal_weight;

                                const float t_cost = const_M_cost(next_stixel_index_column, e) - c;

                                if(t_cost < min_M_minus_c)
                                {
                                    min_M_minus_c = t_cost;
                                    min_M_minus_c_index = e;
                                }

                                // no need to explore the rows below since they are non-valid
                                break;
                            }

                        } // end of "for current and next row"

                    } // end of checks for "same row or (row + 1)

                    M_cost(stixel_index, row_step_index) += min_M_minus_c;
                    min_M_minus_c_indices_stixel[row_step_index] = min_M_minus_c_index;

                } // end of "for each row step"
            } // end of "for each stixel", i.e. "for each column in the image"

        } // end of right to left pass

        // left to right pass --
        {
            stixel_and_row_step_ground_obstacle_boundary.resize(num_stixels);

            // we set the first value directly
            {
                int &row_step_star = stixel_and_row_step_ground_obstacle_boundary[0];
                // minCoeff takes the "lowest index", but
                // we search for the maximum index with minimum value
                const_M_cost.row(0).minCoeff(&row_step_star);
            }

            {
                int &previous_row_step_star = stixel_and_row_step_ground_obstacle_boundary[0];
                int previous_stixel_index = 0;

                // the rest are set using the stored min_M_minus_c_indices
                for(int stixel_index = 1; stixel_index < num_stixels; stixel_index += 1, previous_stixel_index = stixel_index)
                {
                    int &row_step_star = stixel_and_row_step_ground_obstacle_boundary[stixel_index];
                    // (no boundary check for speed reasons, in debug mode Eigen does the checks)
                    row_step_star = min_M_minus_c_indices[previous_stixel_index][previous_row_step_star];
                    previous_row_step_star = row_step_star;
                } // end of "for each column", i.e. "for each u value"
            }

        } // end of left to right pass

    }

    // at this point stixel_and_row_step_ground_obstacle_boundary is now set

    { // we store the final result

        // to make things simple we always output one stixel per column
        // we store the object bottom for each image column
        u_v_ground_obstacle_boundary.resize(input_left_view.width());

        const bool interpolate_rows = false;

        if(interpolate_rows)
        {
            for(size_t stixel_index=0; stixel_index < num_stixels; stixel_index += 1)
            {
                const size_t next_stixel_index = std::min(stixel_index + 1, num_stixels - 1);
                const size_t
                        min_cost_row_step_index = stixel_and_row_step_ground_obstacle_boundary[stixel_index],
                        min_cost_next_row_step_index = stixel_and_row_step_ground_obstacle_boundary[next_stixel_index];

                const row_t begin_row = bottom_v_given_stixel_and_row_step[stixel_index][min_cost_row_step_index],
                        end_row = bottom_v_given_stixel_and_row_step[next_stixel_index][min_cost_next_row_step_index];

                const float row_slope = static_cast<float>(end_row - begin_row) / stixel_width;
                const size_t
                        begin_u = stixel_index*stixel_width,
                        end_u = std::min((stixel_index+1)*stixel_width, u_v_ground_obstacle_boundary.size());
                float delta_u = 0;
                for(size_t stixel_u = begin_u; stixel_u < end_u; stixel_u += 1, delta_u +=1)
                {
                    const row_t row = begin_row + static_cast<row_t>(row_slope*delta_u);
                    u_v_ground_obstacle_boundary[stixel_u] = row;
                } // end of "for each column covered by the stixel

            } // end of "for each stixel"
        }
        else
        { // no rows interpolation, we use constant value

            for(size_t stixel_index=0; stixel_index < num_stixels; stixel_index += 1)
            {
                const size_t min_cost_row_step_index = stixel_and_row_step_ground_obstacle_boundary[stixel_index];
                const row_t row = bottom_v_given_stixel_and_row_step[stixel_index][min_cost_row_step_index];
                const size_t
                        begin_u = stixel_index*stixel_width,
                        end_u = std::min((stixel_index+1)*stixel_width, u_v_ground_obstacle_boundary.size());

                for(size_t stixel_u = begin_u; stixel_u < end_u; stixel_u += 1)
                {
                    u_v_ground_obstacle_boundary[stixel_u] = row;
                } // end of "for each column covered by the stixel
            } // end of "for each stixel"

        }

        // use u_v_ground_obstacle_boundary to build the final stixels output
        u_v_disparity_boundary_to_stixels();
    }

    return;
}


void ImagePlaneStixelsEstimator::u_v_disparity_boundary_to_stixels()
{
    // to make things simple we always output one stixel per column
    const int num_columns = input_left_view.width();
    the_stixels.resize(num_columns);

    const size_t max_row_step_index = num_row_steps - 1;

    bool previous_is_occluded = false;
    const size_t num_stixels = bottom_v_given_stixel_and_row_step.shape()[0];
    for(size_t stixel_index = 0; stixel_index < num_stixels; stixel_index += 1)
    {

        const int
                begin_u = stixel_index*stixel_width,
                end_u = std::min<int>(num_columns, (stixel_index+1)*stixel_width),
                next_begin_u = std::min(end_u, num_columns -1);

        const int
                &begin_bottom_v = u_v_ground_obstacle_boundary[begin_u],
                &begin_disparity = disparity_given_v[begin_bottom_v],
                &next_v = u_v_ground_obstacle_boundary[next_begin_u],
                &next_disparity = disparity_given_v[next_v];

        const bool is_obviously_occluded = ((next_disparity - begin_disparity) >= stixel_width);

        const size_t
                next_stixel_index = std::min(stixel_index+1, num_stixels - 1),
                current_min_cost_row_step_current = stixel_and_row_step_ground_obstacle_boundary[stixel_index],
                next_min_cost_row_step_current = stixel_and_row_step_ground_obstacle_boundary[next_stixel_index];

        // disparity is increasing
        const bool is_possibly_occluded = next_min_cost_row_step_current > current_min_cost_row_step_current;

        bool seems_occluded = false; // border case, where it is touching the occlusion boundary
        if((not is_obviously_occluded) and is_possibly_occluded)
        {
            if(next_min_cost_row_step_current < max_row_step_index)
            {
                const int next_plus_one_disparity = \
                        disparity_given_stixel_and_row_step[next_stixel_index][next_min_cost_row_step_current + 1];

                seems_occluded = ((next_plus_one_disparity - begin_disparity) >= stixel_width);
            }

        } // end of "if not obviously occluded"

        const bool is_occluded = is_obviously_occluded or (is_possibly_occluded and seems_occluded);

        for(int u = begin_u; u < end_u; u += 1)
        {
            const int &bottom_v = u_v_ground_obstacle_boundary[u];
            // map from v to disparity based on the ground estimate
            const int &disparity = disparity_given_v[bottom_v];
            const int &top_v = expected_v_given_disparity[disparity];

            Stixel &t_stixel = the_stixels[u];
            t_stixel.width = 1;
            t_stixel.x = u;
            t_stixel.bottom_y = bottom_v;
            t_stixel.top_y = top_v;
            t_stixel.default_height_value = true;
            t_stixel.disparity = disparity;
            t_stixel.type = Stixel::Unknown;

            //if(is_occluded)
            if(is_occluded and previous_is_occluded)
                //if(is_occluded or previous_is_occluded)
            {
                t_stixel.type = Stixel::Occluded;
            }

        } // end of "for each image column covered by the stixel"

        previous_is_occluded = is_occluded;
    } // end of "for each stixel"

    return;
}



} // end of namespace doppia
