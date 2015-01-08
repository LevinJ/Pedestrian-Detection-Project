#include "GroundPlaneEstimator.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"

#include "image_processing/IrlsLinesDetector.hpp"
#include "image_processing/OpenCvLinesDetector.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <Eigen/Dense>

#include <omp.h>

#include <numeric>
#include <string>
#include <limits>
#include <stdexcept>

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"
#include "helpers/xyz_indices.hpp"

namespace {

using namespace logging;

std::ostream &log_info()
{
    return log(InfoMessage, "GroundPlaneEstimator");
}

std::ostream &log_warning()
{
    return log(WarningMessage, "GroundPlaneEstimator");
}

std::ostream &log_error()
{

    return log(ErrorMessage, "GroundPlaneEstimator");
}

std::ostream &log_debug()
{

    return log(DebugMessage, "GroundPlaneEstimator");
}

} // end of private namespace

namespace doppia {



using namespace std;
using namespace boost;

program_options::options_description GroundPlaneEstimator::get_args_options()
{
    program_options::options_description desc("GroundPlaneEstimator options");

    desc.add_options()

            ("ground_plane_estimator.use_opencv_lines_detection",
             program_options::value<bool>()->default_value(false),
             "use opencv lines detection instead of (faster, better) ad-hoc line estimation")

            ;

    return desc;

}


GroundPlaneEstimator::GroundPlaneEstimator(
    const boost::program_options::variables_map &options,
    const StereoCameraCalibration& camera_calibration,
    const bool cost_volume_is_from_residual_image_)
    :
      BaseGroundPlaneEstimator(options, camera_calibration),
      cost_volume_is_from_residual_image(cost_volume_is_from_residual_image_),
      num_ground_plane_estimation_failures(0)
{

    // create lines detector
    const bool use_opencv_lines_detection =
            get_option_value<bool>(options, "ground_plane_estimator.use_opencv_lines_detection");

    if(use_opencv_lines_detection)
    {
        // FIXME hardcoded values
        const float input_image_threshold = 150;
        const float direction_resolution = (M_PI/180)*1;
        const float origin_resolution = 1;
        const int detection_threshold = 70;

        lines_detector_p.reset(
                    new OpenCvLinesDetector(input_image_threshold,
                                            direction_resolution, origin_resolution,
                                            detection_threshold) );
    }
    else
    {
        lines_detector_p.reset(new IrlsLinesDetector(options));
    }


    current_orientation_filter_phase = -1;
    set_orientation_filter(0);

    v_disparity_x_offset = 0;
    return;
}

GroundPlaneEstimator::~GroundPlaneEstimator()
{
    // nothing to do here
    return;
}


void GroundPlaneEstimator::set_ground_disparity_cost_volume(const boost::shared_ptr<DisparityCostVolume> &cost_volume_p)
{
    this->cost_volume_p = cost_volume_p;
    return;
}




void GroundPlaneEstimator::compute_v_disparity_mask()
{
    const int num_rows = cost_volume_p->rows();
    //const int num_columns = cost_volume_p->columns();
    const int num_disparities = cost_volume_p->disparities();

    if( (v_disparity_mask.rows() == num_rows) and (v_disparity_mask.cols() == num_disparities))
    {
        // lazy computation of the v_disparity_mask
        // the mask is already computed
        return;
    }

    // initialize with ones -
    v_disparity_mask = Eigen::MatrixXi::Constant(num_rows, num_disparities, 1);

    // plus and minus line -
    const line_t &plus_line = get_prior_max_v_disparity_line();
    const line_t &minus_line = get_prior_min_v_disparity_line();
    for(int v = 0; v < v_disparity_mask.rows(); v += 1 )
    {
        for(int d = 0; d < v_disparity_mask.cols(); d += 1 )
        {
            const int plus_v = static_cast<int>(plus_line.direction()(0)*d + plus_line.origin()(0));
            const int minus_v = static_cast<int>(minus_line.direction()(0)*d + minus_line.origin()(0));

            if (v > minus_v or v < plus_v)
            {
                v_disparity_mask(v,d) = 0;
            }
            else
            {
                // v_disparity_mask is set to 1 initially
                //v_disparity_mask(v,d) = 1;
                continue;
            }

        } // end of "for each col"
    } // end of "for each row"


    return;
}

void GroundPlaneEstimator::compute_v_disparity_data()
{

    // sum cost volume over the x axis to obtain the v-disparity image ---

    // lazy allocation
    const int num_rows = cost_volume_p->rows();
    const int num_columns = cost_volume_p->columns();
    const int num_disparities = cost_volume_p->disparities();

    // lazy computation of the v_disparity_mask
    compute_v_disparity_mask();

    // initialize with zeros -
    v_disparity_data = Eigen::MatrixXf::Zero(num_rows, num_disparities);

    typedef DisparityCostVolume::const_data_3d_view_t const_data_3d_view_t;
    typedef DisparityCostVolume::const_data_2d_subarray_t const_data_2d_subarray_t;
    typedef DisparityCostVolume::const_data_1d_subarray_t const_data_1d_subarray_t;
    typedef DisparityCostVolume::range_t range_t;

    // data is organized as y (rows), x (columns), disparity
    DisparityCostVolume::const_data_3d_view_t data = cost_volume_p->get_costs_views();


    float max_v_disparity_data_value = -std::numeric_limits<float>::max();
    float min_v_disparity_data_value = std::numeric_limits<float>::max();

    assert(ground_object_boundary_prior.size() == cost_volume_p->columns());
    const int horizon_row = *min_element(ground_object_boundary_prior.begin(), ground_object_boundary_prior.end());

    const bool use_ground_area_prior = true;

    // FIXME hardcoded values
    const int top_bottom_margin = 10; // [pixels]

    // FIXME hardcoded values
    const float max_cost = 25; // [intensity value]
    //const float max_cost = 125; // [intensity value]

#pragma omp parallel for
    for(int row=0; row < num_rows; row +=1)
    {   
        if(row < horizon_row or
                row < top_bottom_margin or
                row > static_cast<int>(num_rows - top_bottom_margin))
        {
            // we skip the upper part of the image (above the horizon)
            // we skip the top and bottom rows since they contain "black pixels" areas (due to rectification)
            continue;
        }

        const_data_2d_subarray_t column_disparity_slice = data[row];

        const_data_2d_subarray_t::const_iterator column_disparity_slice_it;
        column_disparity_slice_it = column_disparity_slice.begin();
        for(int col=0; col < num_columns; col +=1, ++column_disparity_slice_it)
        {
            if(use_ground_area_prior)
            {
                const int &boundary_row = ground_object_boundary_prior[col];
                if(row < boundary_row)
                {
                    // skip the pixels that are above the ground area
                    continue;
                }
            }

            const_data_1d_subarray_t disparity_slice = *column_disparity_slice_it;
            const_data_1d_subarray_t::const_iterator costs_it = disparity_slice.begin();
            for(int disparity=0; disparity < num_disparities; disparity +=1, ++costs_it)
            {
                const float &cost = *costs_it;
                // we do not count high errors
                v_disparity_data(row, disparity) += std::min(cost, max_cost);
                //v_disparity_data(row, disparity) += cost;
                /*if(cost < max_cost)
                {
                    v_disparity_data(row, disparity) += cost;
                }
                else
                {
                    continue;
                }*/

            } // end of "for each disparity"

        } // end of "for each column"
    } // end of "for each row"


    max_v_disparity_data_value = v_disparity_data.maxCoeff();
    min_v_disparity_data_value = v_disparity_data.minCoeff();

    if(false)
    {
        log_debug() << "max_v_disparity_data_value == " << max_v_disparity_data_value << std::endl;
        log_debug() << "min_v_disparity_data_value == " << min_v_disparity_data_value << std::endl;
    }

    // scale to 0-255 -
    {
        v_disparity_data.array() -= min_v_disparity_data_value;
        v_disparity_data *= 255.f / (max_v_disparity_data_value - min_v_disparity_data_value);
    }

    // set to max_value the upper area and lower area of the v_disparity_data
    v_disparity_data.block(0, 0, horizon_row, num_disparities).setConstant(255.f);
    //v_disparity_data.block(num_rows - top_bottom_margin, 0, top_bottom_margin, num_disparities).setConstant(255.f);

    // use v_disparity_mask --
    v_disparity_data =
            v_disparity_data.array().max((v_disparity_mask.cast<float>().array() - 1)* -255.f);

    return;
}



void high_pass_filter(const boost::gil::gray8c_view_t &src_view, const boost::gil::gray8_view_t &dst_view)
{
    // Implementation based on Toby Vaudrey code
    // http://www.cs.auckland.ac.nz/~tobi/openCV-Examples.html

    using namespace cv;
    using namespace boost;

    // FIXME hardcoded value
    //const int num_iterations = 40;
    //const int num_iterations = 20;
    const int num_iterations = 5;
    //const int num_iterations = 2;

    gil::opencv::ipl_image_wrapper ipl_src = gil::opencv::create_ipl_image(src_view);
    gil::opencv::ipl_image_wrapper ipl_dst = gil::opencv::create_ipl_image(dst_view);

    Mat src(ipl_src.get()), dst(ipl_dst.get());

    Mat float_src(src.size(), CV_32FC1);
    Mat float_residual(src.size(), CV_32FC1);
    Mat float_smooth(src.size(), CV_32FC1);



    vector<Mat> channels;

    const bool process_colors = true;
    if(process_colors)
    {
        split(src, channels);
    }
    else
    {
        channels.resize(1);
        cvtColor(src, channels[0], CV_RGB2GRAY);
    }

    BOOST_FOREACH(Mat &channel, channels)
    {
        // Convert to float image (f)
        channel.convertTo(float_src, float_src.depth());
        normalize(float_src, float_src, 0.f, 1.f, NORM_MINMAX);

        // Smooth image (s)
        // (using 3x3 MEAN filter [CV_BLUR])
        //cvSmooth(float_src, float_smooth, CV_BLUR, 3);
        blur(float_src, float_smooth, Size(3,3));
        //const double sigma_color = 0.3, sigma_space = 3;
        //bilateralFilter(float_src, float_smooth, -1, sigma_color, sigma_space);
        for(int i = 0 ; i < num_iterations ; i+=1)
        {
            blur(float_smooth, float_smooth, Size(3,3));
            //bilateralFilter(float_smooth, float_smooth, -1, sigma_color, sigma_space);
        }

        const bool compute_residual = true;
        if(compute_residual)
        {
            // Compute Residual Image
            // (r = f - s)
            float_residual = float_src - float_smooth;

            // Find max value
            double min_value, max_value;
            minMaxLoc(float_residual, &min_value, &max_value);
            max_value = std::max(std::abs<double>(min_value), max_value);
            // Rescale between 0 to 1
            float_residual *= 1.f / max_value;
            float_residual += 0.5f;

            //normalize(float_residual, float_residual, 0.f, 255.f, NORM_MINMAX);

            float_residual.convertTo(channel, channel.depth(), 255.f);
            //float_residual.convertTo(channel, channel.depth());
        } else {
            float_smooth.convertTo(channel, channel.depth(), 255.f);
        }

    } // end of "for each channel"

    if(channels.size() == 1 and dst.type() == CV_8UC3)
    {
        cvtColor(channels[0], dst, CV_GRAY2RGB);
    }
    else
    {
        merge(channels, dst);
    }

    return;
}


void GroundPlaneEstimator::compute_v_disparity_image()
{

    const int num_rows = cost_volume_p->rows();
    const int cols = cost_volume_p->disparities();

    // FIXME hardcoded values
    //const int horizon_row = rows*0.4;
    const int horizon_row =
            std::max(std::min(
                         *min_element(ground_object_boundary_prior.begin(), ground_object_boundary_prior.end()),
                         num_rows - 2), 0); // num_rows - 2 so we can have a block of height 1
    const int top_bottom_margin = 10; // pixels

    // lazy allocation
    raw_v_disparity_image.recreate(cols, num_rows);
    raw_v_disparity_image_view = boost::gil::view(raw_v_disparity_image);
    raw_v_disparity_image_const_view = boost::gil::const_view(raw_v_disparity_image);

    v_disparity_image.recreate(cols, num_rows);
    v_disparity_image_view = boost::gil::view(v_disparity_image);
    v_disparity_image_const_view = boost::gil::const_view(v_disparity_image);


    Eigen::MatrixXf block = v_disparity_data.block(horizon_row, 0,
                                                   std::max(1, num_rows -horizon_row - top_bottom_margin), cols);

    // do base normalizations --
    const bool normalize_block = false;
    if(normalize_block)
    { // normalize the block
        const float t_min = block.minCoeff();
        const float t_max = block.maxCoeff();
        block.array() -= t_min;
        block *= 255.f / (t_max - t_min);
    }


    const bool normalize_verticaly = false; // was true
    if(normalize_verticaly)
    { // normalize verticaly
        // FIXME dirty hack to get rid of "black areas disparities"
        for(int disparity=0; disparity < block.cols(); disparity +=1)
        {
            const float t_min = block.col(disparity).minCoeff();
            const float t_max = block.col(disparity).maxCoeff();
            block.col(disparity).array() -= t_min;
            block.col(disparity) *= 255.f / (t_max - t_min);
        }
    }

    const bool normalize_horizontaly = true;
    if(normalize_horizontaly)
    { // normalize horizontaly

        for(int row=0; row < block.rows(); row +=1)
        {
            const float t_min = block.row(row).minCoeff();
            const float t_max = block.row(row).maxCoeff();
            block.row(row).array() -= t_min;
            block.row(row) *= 255.f / (t_max - t_min);
        }
    }

    v_disparity_data.block(horizon_row, 0,
                           std::max(1, num_rows -horizon_row -top_bottom_margin), cols) = block;

    // copy data into image -
    for(int row=0; row < num_rows; row +=1)
    {
        for(int disparity=0; disparity < cols; disparity +=1)
        {
            raw_v_disparity_image_view.at(disparity, row)[0] = 255 - v_disparity_data(row, disparity);
        } // end of "for each disparity"
    } // end of "for each row"


    boost::gil::copy_pixels(raw_v_disparity_image_const_view, v_disparity_image_view);

    const bool show_raw_v_disparity_image = false;
    if(show_raw_v_disparity_image)
    {
        boost::gil::opencv::ipl_image_wrapper src_img =
                boost::gil::opencv::create_ipl_image(v_disparity_image_view);
        cv::Mat input_image(src_img.get());
        cv::imshow("raw_v_disparity_image", input_image);
        cv::waitKey(0);
    }


    const bool apply_edges_detector = false;
    if(apply_edges_detector)
    {
        printf("PING 0.0\n");
        boost::gil::opencv::ipl_image_wrapper src_img =
                boost::gil::opencv::create_ipl_image(v_disparity_image_view);
        cv::Mat input_image(src_img.get()), temp_image, output_image;

        printf("PING 0\n");
        input_image.copyTo(temp_image);
        input_image.copyTo(output_image);

        // from the edge.c example in OpenCv source code
        // const float edge_threshold1 = 1, edge_threshold2 = 3;
        printf("PING 1\n");
        //cv::Canny(temp_image, output_image, edge_threshold1, edge_threshold2);
        cv::Sobel(temp_image, output_image, input_image.depth(), 1, 1);
        //cv::normalize(output_image, output_image);

        double min_val, max_val;
        cv::minMaxLoc(output_image, &min_val);
        output_image -= min_val;
        cv::minMaxLoc(output_image, &min_val, &max_val);
        output_image *= 255.0f/max_val;

        //output_image = input_image;
        //cv::imshow("edge_image", output_image);
        //cv::waitKey(0);

        output_image.copyTo(input_image);
    }

    const bool apply_high_pass_filter = false;
    if(apply_high_pass_filter)
    {
        high_pass_filter(v_disparity_image_view, v_disparity_image_view);
    }


    const bool threshold_the_image = true;
    if((not cost_volume_is_from_residual_image) and threshold_the_image)
    {
        boost::gil::opencv::ipl_image_wrapper src_img =
                boost::gil::opencv::create_ipl_image(v_disparity_image_view);
        cv::Mat input_image(src_img.get()), output_image;

        // FIXME harcoded value
        //const int threshold_margin = 2; // [intensity value]
        const int threshold_margin = 15; // [intensity value]

        cv::threshold(input_image, output_image,
                      255 - threshold_margin, 255, cv::THRESH_TOZERO);

        const bool show_thresholded_image = false;
        if(show_thresholded_image)
        {
            cv::imshow("threshold_image", output_image);
            cv::waitKey(0);
        }

        output_image.copyTo(input_image);
    }


    const bool filter_v_disparity_using_prior = true;
    if(filter_v_disparity_using_prior)
    {
        // filter lines that are not similar to our prior orientation

        const line_t prior_line = ground_plane_to_v_disparity_line(get_ground_plane_prior());
        const float orientation = atan2(prior_line.direction()(0), 1) + M_PI/2;
        set_orientation_filter(orientation);

        boost::gil::opencv::ipl_image_wrapper src_img =
                boost::gil::opencv::create_ipl_image(v_disparity_image_view);
        cv::Mat input_image(src_img.get()), output_image;

        cv::filter2D(input_image, output_image, input_image.depth(), orientation_filter_kernel);

        const bool show_filtering = false;
        if(show_filtering)
        {
            cv::imshow("input_image", input_image);
            cv::imshow("output_image", output_image);
            cv::Mat kernel_show;
            cv::resize(orientation_filter_kernel, kernel_show, cv::Size(200, 200));
            kernel_show *= 0.5; kernel_show += 0.5;
            cv::imshow("filter_kernel", kernel_show);

            if(false)
            {
                cv::imwrite("input_v_disparity.png", input_image);
                cv::imwrite("output_v_disparity.png", output_image);
            }

            cv::waitKey(0);
        }

        output_image.copyTo(input_image);

        // we store the x_offset induced by the filtering stage,
        // we use it to compensate latter in the obtained line
        v_disparity_x_offset = (orientation_filter_kernel.cols - 1) / 2;

    } // end of "filter_v_disparity_using_prior"


    const bool save_v_disparity_image = false;
    if(save_v_disparity_image)
    {
        const std::string filename = "v_disparity.png";
        log_info() << "Created image " << filename << std::endl;
        boost::gil::png_write_view(filename, v_disparity_image_view);
    }


    return;
}

bool GroundPlaneEstimator::find_ground_line(AbstractLinesDetector::line_t &ground_line) const
{
    // find the most likely plane (line) in the v-disparity image ---
    AbstractLinesDetector::lines_t found_lines;
    bool found_ground_plane = false;

    IrlsLinesDetector *irls_lines_detector_p = dynamic_cast<IrlsLinesDetector *>(lines_detector_p.get());
    if(irls_lines_detector_p)
    {
        const line_t line_prior = ground_plane_to_v_disparity_line(estimated_ground_plane);
        irls_lines_detector_p->set_initial_estimate(line_prior);
    }

    (*lines_detector_p)(v_disparity_image_const_view, found_lines);

    // given two bounding lines we verify the x=0 line and the y=max_y line
    // this checks bound quite well the desired ground line

    const float max_line_direction = prior_max_v_disparity_line.direction()(0),
            min_line_direction = prior_min_v_disparity_line.direction()(0);

    const float max_line_y0 = prior_max_v_disparity_line.origin()(0),
            min_line_y0 = prior_min_v_disparity_line.origin()(0);

    const float min_y0 = max_line_y0, max_y0 = min_line_y0;


    const float y_intercept = v_disparity_image_const_view.height();
    const float max_x_intercept = (y_intercept - max_line_y0) / max_line_direction,
            min_x_intercept = (y_intercept - min_line_y0) / min_line_direction;

    assert(min_y0 < max_y0);
    assert(min_x_intercept < max_x_intercept);


    AbstractLinesDetector::lines_t::const_iterator lines_it;
    for(lines_it = found_lines.begin(); lines_it != found_lines.end(); ++lines_it)
    {
        AbstractLinesDetector::line_t t_ground_line = *lines_it;

        /*// FIXME just for debugging
        ground_line = t_ground_line;
        found_ground_plane = true;
        break;*/

        {   // since we compute the ground_line over a filtered image,
            // there is an x offset, which in turn generates a origin offset

            // FIXME this seems like a bug !
            //const float origin_offset = -v_disparity_x_offset * t_ground_line.direction()(0);
            const float origin_offset = 0;

            t_ground_line.origin()(0) += origin_offset;
        }

        const float t_y0 = t_ground_line.origin()(0);
        const float t_x_intercept = (y_intercept - t_y0) / t_ground_line.direction()(0);

        if(t_y0 <= max_y0 and t_y0 >= min_y0 and
                t_x_intercept <= max_x_intercept and t_x_intercept >= min_x_intercept)
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
} // end of "GroundPlaneEstimator::find_ground_line"

void GroundPlaneEstimator::estimate_ground_plane()
{
    const bool found_ground_plane = find_ground_line(v_disparity_ground_line);

    // retrieve ground plane parameters --
    if(found_ground_plane == true)
    {
        const float weight = 1.0;
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

        log_warning() << "Did not find a ground plane, keeping previous estimate." << std::endl;
        const float weight = 1.0;
        // we keep previous estimate
        set_ground_plane_estimate(estimated_ground_plane, weight);

        const bool save_failures_v_disparity_image = true;
        if(save_failures_v_disparity_image)
        {
            const std::string filename = boost::str(
                        boost::format("failure_v_disparity_%i.png") % num_ground_plane_estimation_failures );
            log_info() << "Created image " << filename << std::endl;
            boost::gil::png_write_view(filename, v_disparity_image_view);
        }
    }

} // end of "GroundPlaneEstimator::estimate_ground_plane"

void GroundPlaneEstimator::compute()
{ 
    static int num_iterations = 0;
    static double cumulated_time = 0;

    const int num_iterations_for_timing = 50;
    const double start_wall_time = omp_get_wtime();

    compute_v_disparity_data();
    compute_v_disparity_image();
    estimate_ground_plane();


    // timing ---
    cumulated_time += omp_get_wtime() - start_wall_time;
    num_iterations += 1;

    if((num_iterations % num_iterations_for_timing) == 0)
    {
        printf("Average GroundPlaneEstimator::compute speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
        cumulated_time = 0;
    }

    return;
}



const GroundPlaneEstimator::v_disparity_const_view_t &GroundPlaneEstimator::get_raw_v_disparity_view() const
{
    return raw_v_disparity_image_const_view;
}

const GroundPlaneEstimator::v_disparity_const_view_t &GroundPlaneEstimator::get_v_disparity_view() const
{
    return v_disparity_image_const_view;
}

void GroundPlaneEstimator::set_orientation_filter(const float orientation)
{
    // based on code from
    // http://www.eml.ele.cst.nihon-u.ac.jp/~momma/wiki/wiki.cgi/OpenCV/Gabor%20Filter.html

    if(boost::math::isnan(orientation))
    {
        throw std::invalid_argument("GroundPlaneEstimator::set_orientation_filter received a NaN orientation");
    }

    if(orientation == current_orientation_filter_phase)
    {
        // no need to update the filter
        return;
    }
    else
    {
        current_orientation_filter_phase = orientation;
    }

    cv::Mat_<float> &kernel = orientation_filter_kernel;

    const int kernel_size = 7; // 15; // 5; // [pixels]
    const float filter_variance = 30; // a value between 30 and 100 (100 means black and white stripes)
    const float filter_pulsation = 6; // a value between 5 to 7
    const float filter_phase = orientation; // [radians]
    const float filter_psi = M_PI / 2; // [radians]

    assert(kernel_size % 2 == 1);
    kernel = cv::Mat_<float>(kernel_size, kernel_size);

    const int kernel_half_size = (kernel_size - 1) / 2;
    const float variance = filter_variance/10.0f;
    const float w = filter_pulsation/10.0f;
    const float psi = filter_psi;
    double sin_phase, cos_phase;
    sincos(filter_phase, &sin_phase, &cos_phase);

    for (int y = -kernel_half_size; y<=kernel_half_size; y+=1) {
        for (int x = -kernel_half_size; x<=kernel_half_size; x+=1) {
            const float numerator = -((x*x)+(y*y));
            const float denominator = (2*variance);
            const float exp_value = std::exp(numerator/ denominator);
            const float cos_value = std::cos(w*x*cos_phase+w*y*sin_phase+psi);
            const float kernel_value = exp_value*cos_value;
            kernel(y+kernel_half_size, x+kernel_half_size) = kernel_value;
        }
    }

    return;
}

} // end of namespace doppia
