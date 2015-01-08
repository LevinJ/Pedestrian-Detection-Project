
#include "CpuPreprocessor.hpp"

#include "FastReverseMapper.hpp"

#include "helpers/Log.hpp"

#include <Eigen/Dense>

#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/foreach.hpp>

#include <helpers/get_option_value.hpp>

// FIXME should not depend on OpenCv for video input
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h> // opencv 2.4, for cvSmooth
#include <opencv2/highgui/highgui.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "CpuPreprocessor");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "CpuPreprocessor");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "CpuPreprocessor");
}

} // end of anonymous namespace

namespace doppia
{
using namespace std;
using namespace boost;
using namespace Eigen;

program_options::options_description CpuPreprocessor::get_args_options()
{
    program_options::options_description desc("CpuPreprocessor options");

    desc.add_options()
            ("preprocess.residual",
             program_options::value<bool>()->default_value(false),
             "use residual image instead of input image")

            ("preprocess.specular",
             program_options::value<bool>()->default_value(true),
             "transform the RGB image into an UV image where specular reflections are mitigated")
            ;


    return desc;
}



CpuPreprocessor::
CpuPreprocessor(
        const dimensions_t &dimensions,
        const StereoCameraCalibration &stereo_calibration,
        const boost::program_options::variables_map &options)
    : AbstractPreprocessor(dimensions, stereo_calibration, options), post_processing_stereo_calibration(stereo_calibration)
{

    should_compute_residual = get_option_value<bool>(options, "preprocess.residual");
    should_remove_specular_reflection = get_option_value<bool>(options, "preprocess.specular");

    compute_rectification_homographies(dimensions, stereo_calibration);

    left_camera_calibration_p.reset(new CameraCalibration(stereo_calibration.get_left_camera_calibration()));
    right_camera_calibration_p.reset(new CameraCalibration(stereo_calibration.get_right_camera_calibration()));


    const bool use_slow_reverse_mapper = false;

    if(use_slow_reverse_mapper)
    {
        left_reverse_mapper_p.reset(new ReverseMapper(dimensions, left_camera_calibration_p));
        right_reverse_mapper_p.reset(new ReverseMapper(dimensions, right_camera_calibration_p));
    }
    else
    {
        left_reverse_mapper_p.reset(new FastReverseMapper(dimensions, left_camera_calibration_p));
        right_reverse_mapper_p.reset(new FastReverseMapper(dimensions, right_camera_calibration_p));
    }

    if(this->do_undistortion)
    {
        left_reverse_mapper_p->add_undistortion();
        right_reverse_mapper_p->add_undistortion();
    }

    if(this->do_rectification)
    {
        const bool H_is_centered = false;
        left_reverse_mapper_p->add_homography_projection(left_rectification_homography, H_is_centered);
        right_reverse_mapper_p->add_homography_projection(right_rectification_homography, H_is_centered);
    }


    set_post_processing_stereo_calibration();

    return;
}

CpuPreprocessor::~CpuPreprocessor()
{
    // nothing to do here
    return;
}

void CpuPreprocessor::set_post_processing_stereo_calibration()
{
    // this method assumes that compute_rectification_homographies has already been called
    assert(left_rectification_homography != HomographyMatrix::Zero());

    CameraCalibration &left_calibration = post_processing_stereo_calibration.get_left_camera_calibration();
    CameraCalibration &right_calibration = post_processing_stereo_calibration.get_right_camera_calibration();

    if(this->do_undistortion)
    {
        left_calibration.get_radial_distortion_parameters()
                = RadialDistortionParametersVector::Zero();

        right_calibration.get_radial_distortion_parameters()
                = RadialDistortionParametersVector::Zero();

        left_calibration.get_tangential_distortion_parameters()
                = TangentialDistortionParametersVector::Zero();

        right_calibration.get_tangential_distortion_parameters()
                = TangentialDistortionParametersVector::Zero();

    }


    if(this->do_rectification)
    {
        InternalCalibrationMatrix & left_matrix = left_calibration.get_internal_calibration();
        left_matrix = new_left_internal_calibration;

        InternalCalibrationMatrix & right_matrix = right_calibration.get_internal_calibration();
        right_matrix = new_right_internal_calibration;
    }


    return;
}


/// @returns the stereo calibration corresponding to the post-processed images
const StereoCameraCalibration& CpuPreprocessor::get_post_processing_calibration() const
{
    return post_processing_stereo_calibration;
}


void CpuPreprocessor::run(const input_image_view_t& input, const int camera_index, const output_image_view_t &output)
{

    if(camera_index < 0 or camera_index > 1)
    {
        throw std::runtime_error("On a stereo camera, camera_index can only be 0 or 1");
    }

    if (this->do_unbayering)
    {
        //output = unbayerBW(input, umLaroche);
        throw std::runtime_error("Unbayering is not yet implemented");
    }
    else
    {
        boost::gil::copy_pixels(input, output);
    }

    if (this->do_smoothing)
    {
        //smooth(output, 0.7);
        compute_smoothing(output, output);
    }

    if(should_compute_residual)
    {
        compute_residual(output, output);
    }

    if(should_remove_specular_reflection)
    {
        compute_specular_removal(output, output);
    }

    if (this->do_undistortion or this->do_rectification)
    {
        compute_warping(output, camera_index, output);
    }


    return;
}

const point2<float> CpuPreprocessor::run(const point2<int> &point, const int camera_index) const
{

    if(camera_index < 0 or camera_index > 1)
    {
        throw std::runtime_error("On a stereo camera, camera_index can only be 0 or 1");
    }

    point2<float> t_point(point.x, point.y);

    if (this->do_undistortion or this->do_rectification)
    {
        t_point = compute_warping(point, camera_index);
    }

    return t_point;
}

/**
Smooth given image with a Gaussian

 @param im image to smooth
 @param sigma sigma of Gaussian
 @param fsize size of filter mask
*/
void CpuPreprocessor::compute_smoothing(const input_image_view_t &src, const output_image_view_t &dst)
{
    // see codebase/lib/graphics/ImageSmooth.hpp

    // FIXME hardcoded value
    //const int num_iterations = 20;
    //const int num_iterations = 5;
    const int num_iterations = 2;
    //const int num_iterations = 1;

    gil::opencv::ipl_image_wrapper ipl_src = gil::opencv::create_ipl_image(src);
    gil::opencv::ipl_image_wrapper ipl_dst = gil::opencv::create_ipl_image(dst);

    // Smooth image (s)
    // (using 3x3 MEAN filter [CV_BLUR])
    cvSmooth(ipl_src.get(), ipl_dst.get(), CV_BLUR, 3);
    for(int i = 0 ; i < num_iterations ; i+=1)
    {
        cvSmooth(ipl_dst.get(), ipl_dst.get(), CV_BLUR,3);
    }


    return;
}

/// @returns focal point of camera
Vector3f get_focal_point(const Pose &pose)
{
    return -pose.R.transpose() * pose.t;
}


void CpuPreprocessor::compute_rectification_homographies(
        const dimensions_t &dimensions,
        const StereoCameraCalibration &stereo_calibration)
{
    // rectification code based on http://profs.sci.univr.it/~fusiello/demo/rect


    // in this function 1 refers to left and 2 refers to right

    const CameraCalibration &left_camera_calibration = stereo_calibration.get_left_camera_calibration();
    const CameraCalibration &right_camera_calibration = stereo_calibration.get_right_camera_calibration();

    const InternalCalibrationMatrix K_left = left_camera_calibration.get_internal_calibration();
    const RotationMatrix R_left = left_camera_calibration.get_pose().rotation;

    const InternalCalibrationMatrix K_right = right_camera_calibration.get_internal_calibration();
    const RotationMatrix R_right = right_camera_calibration.get_pose().rotation;

    // optical centers (unchanged)
    Vector3f focal_center1 = get_focal_point(left_camera_calibration.get_pose());
    Vector3f focal_center2 = get_focal_point(right_camera_calibration.get_pose());

    const Matrix3f
            KR_left_inverse = (K_left * R_left).inverse(),
            KR_right_inverse = (K_right * R_right).inverse();

    //log_debug() << "Q1 == " << Q1 << std::endl;
    //log_debug() << "Q2 == " << Q2 << std::endl;

    // new x axis (direction of the baseline)
    Vector3f v1 = (focal_center2 - focal_center1).normalized(); // Normalise by dividing through by the magnitude.
    // new y axes (old y)
    Vector3f v2 = (R_left.row(1) + R_right.row(1))*0.5;
    v2.normalize();
    // new z axes (orthogonal to baseline and y)
    Vector3f v3 = v1.cross(v2);
    v3.normalize();
    v2 = v3.cross(v1);
    v2.normalize();

    // new extrinsic parameters
    RotationMatrix R_new;
    R_new.row(0) = v1;
    R_new.row(1) = v2;
    R_new.row(2) = v3;

    log_debug() << "v1 == " << v1.transpose() << std::endl;
    log_debug() << "R1 set with v1, v2, v3 ==\n" << R_new << std::endl;


    // find best offset --
    float x_offset_left = 0, x_offset_right = 0;

    // new intrinsic parameters (arbitrary)
    InternalCalibrationMatrix K_new_left, K_new_right;
    K_new_left = (K_left + K_right) * 0.5;
    K_new_right = K_new_left;
    K_new_left(0, 2) += x_offset_left; // [pixels]
    K_new_right(0, 2) -= x_offset_right; // [pixels]


    Matrix3f
            KR_new_left = ((K_new_left * R_new) * KR_left_inverse).inverse(),
            KR_new_right = ((K_new_right * R_new) * KR_right_inverse).inverse();


    //log_debug() << "KR_new_left before centering == " << KR_new_left << std::endl;
    //log_debug() << "KR_new_right before centering == " << KR_new_right << std::endl;


    const bool find_best_offset_for_each_camera = true;
    //const bool find_best_offset_for_each_camera = false; // not fast stixels not updated for that

    if(find_best_offset_for_each_camera)
    {
        // we search the image offset such as most of the original image
        // appears in the rectified one.
        // to do this we check where the left and right corners are projected to.

        // iterate a few times until convergence
        for(int i=0; i < 50; i+=1)
        {
            // new intrinsic parameters (arbitrary)
            K_new_left = (K_left + K_right) * 0.5;
            K_new_right = K_new_left;
            K_new_left(0, 2) += x_offset_left; // [pixels]
            K_new_right(0, 2) -= x_offset_right; // [pixels]

            KR_new_left = ((K_new_left * R_new) * KR_left_inverse).inverse();
            KR_new_right = ((K_new_right * R_new) * KR_right_inverse).inverse();

            const point2<float>
                    top_left(0, 0),
                    top_right(dimensions.x, 0),
                    bottom_left(0, dimensions.y),
                    bottom_right(dimensions.x, dimensions.y);

            const float
                    left_margin_a =
                    (compute_rectification(top_left, KR_new_left).x
                     + compute_rectification(bottom_left, KR_new_left).x
                     )*0.5,
                    left_margin_b = -(dimensions.x -
                                      (compute_rectification(top_right, KR_new_left).x
                                       + compute_rectification(bottom_right, KR_new_left).x
                                       )*0.5),
                    right_margin_a = dimensions.x -
                    ((compute_rectification(top_right, KR_new_right).x
                      + compute_rectification(bottom_right, KR_new_right).x
                      )*0.5),
                    right_margin_b =-(
                        (compute_rectification(top_left, KR_new_right).x
                         + compute_rectification(bottom_left, KR_new_right).x
                         )*0.5);


            const float
                    left_step = (left_margin_a +  left_margin_b)*0.5,
                    right_step = (right_margin_a + right_margin_b)*0.5;
            // left_step =  min(left_margin_a,  left_margin_b),
            // right_step = min(right_margin_a, right_margin_b);

            if((left_step == 0) and (right_step == 0))
            {
                break;
            }

            const float step_scaling = 0.25;
            x_offset_left += left_step * step_scaling;
            x_offset_right += right_step * step_scaling;

            if(false)
                //if(true)
            {
                printf("left margin_a == %.3f, left margin_b == %.3f, x_offset_left == %.3f\n",
                       left_margin_a, left_margin_b, x_offset_left);

                printf("right margin_a== %.3f, right margin_b == %.3f, x_offset_right == %.3f\n",
                       right_margin_a, right_margin_b, x_offset_right);
            }
        } // end of "for a few iterations"

        // we round anyway since later on disparities will be integers
        x_offset_left = boost::math::iround(x_offset_left);
        x_offset_right = boost::math::iround(x_offset_right);

        // new intrinsic parameters (arbitrary)
        K_new_left = (K_left + K_right) * 0.5;
        K_new_right = K_new_left;
        K_new_left(0, 2) += x_offset_left; // [pixels]
        K_new_right(0, 2) -= x_offset_right; // [pixels]

        KR_new_left = ((K_new_left * R_new) * KR_left_inverse).inverse();
        KR_new_right = ((K_new_right * R_new) * KR_right_inverse).inverse();

        log_debug() << "Final x_offset_left after centering == " << x_offset_left << std::endl;
        log_debug() << "Final x_offset_right after centering == " << x_offset_right << std::endl;

        //printf("Final x_offset_left == %.3f, x_offset_right == %.3f\n",
        //       x_offset_left, x_offset_right);

    } // end "find the best offset"


    // do centering, both cameras will have the same image center
    if(not find_best_offset_for_each_camera)
    {
        // we run a few iterations until convergence
        for (int i = 0; i < 50; i+=1)
        {
            // new intrinsic parameters (arbitrary)
            K_new_left = (K_new_left + K_new_right) * 0.5;
            K_new_right = K_new_left;

            K_new_left(0, 2) += (KR_new_left(0, 2) + KR_new_right(0, 2)) / 4;
            K_new_right(0, 2) += (KR_new_left(0, 2) + KR_new_right(0, 2)) / 4;

            KR_new_left = ((K_new_left * R_new) * KR_left_inverse).inverse();
            KR_new_right = ((K_new_right * R_new) * KR_right_inverse).inverse();
        }
    } // end of "if do centering"

    //log_debug() << "KR_new_left after centering == " << KR_new_left << std::endl;
    //log_debug() << "KR_new_right after centering == " << KR_new_right << std::endl;

    new_left_internal_calibration = K_new_left;
    new_right_internal_calibration = K_new_right;

    left_rectification_homography = KR_new_left;
    right_rectification_homography = KR_new_right;

    left_rectification_inverse_homography = left_rectification_homography.inverse();
    right_rectification_inverse_homography = right_rectification_homography.inverse();

    //log_debug() <<
    //               "left_rectification_inverse_homography == " <<
    //               left_rectification_inverse_homography << std::endl;
    return;
}


void CpuPreprocessor::compute_rectification(const input_image_view_t &src, const int camera_index, const output_image_view_t &dst)
{

    if(camera_index == 0)
    {
        compute_rectification(src, left_rectification_homography, dst);

    }
    else
    {
        compute_rectification(src, right_rectification_homography, dst);
    }

    return;
}

// this function shares a lot with void StixelWorldEstimator::project_left_to_right_image()
void CpuPreprocessor::compute_rectification(const input_image_view_t &src, const HomographyMatrix &H, const output_image_view_t &dst)
{
    assert(src.dimensions() == dst.dimensions());

    using namespace boost::gil;
    t_img.recreate(src.dimensions()); // lazy allocation
    typedef AbstractVideoInput::input_image_t::view_t t_img_view_t;
    t_img_view_t t_img_view(gil::view(t_img));

    typedef output_image_view_t::value_type output_value_type_t;
    const output_value_type_t default_pixel_value(0,0,0);


#pragma omp parallel for
    for (int y = 0; y < dst.height(); y+=1)
    {
        point2<float> t_point;
        t_img_view_t::x_iterator row_iterator = t_img_view.row_begin(y);
        for (int x = 0; x < dst.width(); x+=1, ++row_iterator)
        {
            t_point.x = H(0,0) * x + H(0,1) * y + H(0,2);
            t_point.y = H(1,0) * x + H(1,1) * y + H(1,2);
            const float w = H(2,0) * x + H(2,1) * y + H(2,2);
            t_point.x /= w;
            t_point.y /= w;

            //output_value_type_t &output_pixel = (*t_img_view.xy_at(x,y));
            output_value_type_t &output_pixel = *row_iterator;
            //const bool result_was_set = sample(bilinear_sampler(), src, t_point, output_pixel);
            const bool result_was_set = sample(nearest_neighbor_sampler(), src, t_point, output_pixel);

            if (result_was_set == false)
            { // t_point is out of the src range
                output_pixel = default_pixel_value;
            }
        } // for each x value
    } // for each y value

    boost::gil::copy_pixels(gil::const_view(t_img), dst);

    return;
}


const point2<float> CpuPreprocessor::compute_rectification(const point2<float> &point, const int camera_index) const
{
    if(camera_index == 0)
    {
        return compute_rectification(point, left_rectification_inverse_homography);

    }
    else
    {
        return compute_rectification(point, right_rectification_inverse_homography);
    }
}


const point2<float> CpuPreprocessor::compute_rectification(const point2<float> &point, const HomographyMatrix &H) const
{
    point2<float> rectified_point;

    rectified_point.x = H(0,0) * point.x + H(0,1) * point.y + H(0,2);
    rectified_point.y = H(1,0) * point.x + H(1,1) * point.y + H(1,2);
    const float w = H(2,0) * point.x + H(2,1) * point.y + H(2,2);
    rectified_point.x /= w;
    rectified_point.y /= w;

    return rectified_point;
}

void CpuPreprocessor::compute_warping(const input_image_view_t &src, const int camera_index, const output_image_view_t &dst)
{
    if(camera_index == 0)
    {
        left_reverse_mapper_p->warp(src, dst);
    }
    else
    {
        right_reverse_mapper_p->warp(src, dst);
    }

    return;
}

const point2<float> CpuPreprocessor::compute_warping(const point2<int> &point, const int camera_index) const
{
    if(camera_index == 0)
    {
        return left_reverse_mapper_p->warp(point);
    }
    else
    {
        return right_reverse_mapper_p->warp(point);
    }
}

/*
void CpuPreprocessor::compute_harris_corners()
{


    if ((m_pstages & PREPROCESS_HARRIS) and harris != NULL)
    {
        Image< float > gim(im.width(), im.height());

        for (int y = 0; y < im.height(); y++)
            for (int x = 0; x < im.width(); x++)
                gim(x, y) = im(x, y);

        Image< float > gx, gy, gxy;
        gradient(gim, gx, gy);

        gxy = gx * gy;
        gx *= gx;
        gy *= gy;

        smooth(gx, 1.4);
        smooth(gy, 1.4);
        smooth(gxy, 1.4);

        harris->allocate(im.width(), im.height());
        for (int y = 0; y < gim.height(); y++)
            for (int x = 0; x < gim.width(); x++)
            {
                (*harris)(x,y) = gx(x,y)*gy(x,y) - gxy(x,y)*gxy(x,y) -
                                 0.04*(gx(x,y) + gy(x,y))*(gx(x,y) + gy(x,y));
                if ((*harris)(x, y) < 0 or x < 3 or y < 3 or x > gim.width() - 4 or y > gim.height() - 4)
                    (*harris)(x, y) = 0;
            }
    }

    return;
}*/


void CpuPreprocessor::compute_residual(const input_image_view_t &src_view, const output_image_view_t &dst_view)
{
    // Implementation based on Toby Vaudrey code
    // http://www.cs.auckland.ac.nz/~tobi/openCV-Examples.html

    using namespace cv;

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

    if(channels.size() == 1)
    {
        cvtColor(channels[0], dst, CV_GRAY2RGB);
    }
    else
    {
        merge(channels, dst);
    }

    return;
}


// Based on the method from
// Kuk-Jin Yoon and In So Kweon 2006
struct specular_reflection_removal
{

    void operator()( boost::gil::rgb8_pixel_t& pixel )
    {
        using namespace boost::gil;
        gil::bits8 min_value =
                std::min(
                    std::min( pixel[0], pixel[1]), pixel[2]);

        gil::at_c<0>(pixel) -= min_value;
        gil::at_c<1>(pixel) -= min_value;
        gil::at_c<2>(pixel) -= min_value;
        return;
    }

};


void CpuPreprocessor::compute_specular_removal(const input_image_view_t &src, const output_image_view_t &dst)
{

    if(reinterpret_cast<const void *>(&src) != reinterpret_cast<const void *>(&dst))
    {
        boost::gil::copy_pixels(src, dst);
    }

    boost::gil::for_each_pixel(dst, specular_reflection_removal() );

    return;
}


} // end of namespace doppia
