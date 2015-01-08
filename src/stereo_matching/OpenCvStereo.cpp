#include "OpenCvStereo.hpp"

#include "helpers/for_each.hpp"
#include "helpers/get_option_value.hpp"

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/static_assert.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_view_factory.hpp>
#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#if defined(USE_GPU)
#include <opencv2/gpu/gpu.hpp>
#endif

#include <omp.h>

#include <climits>
#include <algorithm>

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
namespace cv
{

StereoGC::StereoGC()
{
    state = NULL;
}

StereoGC::StereoGC(const int numberOfDisparities, const int maxIters)
{
    init(numberOfDisparities, maxIters);
    return;
}

void StereoGC::init(const int numberOfDisparities, const int maxIters)
{
    state = cvCreateStereoGCState(numberOfDisparities, maxIters);
    //state->SADWindowSize = _SADWindowSize;
    return;
}

void StereoGC::operator()( const Mat& left, const Mat& right, Mat& disparity )
{
    assert(state != NULL);
    Mat& disparity_left = disparity;
    disparity_left.create(left.size(), CV_16S);
    Mat disparity_right(left.size(), CV_16S);


    const CvMat left_cvmat(left);
    const CvMat right_cvmat(right);

    CvMat disparity_left_cvmat(disparity_left);
    CvMat disparity_right_cvmat(disparity_right);
    /*
    const CvArr *left_p = &static_cast<CvMat>(&left);
    const CvArr *_right = &static_cast<CvMat>(&right);
    CvArr* _dispLeft = &static_cast<CvMat>(&disparity_left);
    CvArr* _dispRight = &static_cast<CvMat>(&disparity_right);
    */

    const CvArr *left_p = &left_cvmat;
    const CvArr *_right = &right_cvmat;
    CvArr* _dispLeft = &disparity_left_cvmat;
    CvArr* _dispRight = &disparity_right_cvmat;


    printf("Going to call cvFindStereoCorrespondenceGC\n");
    cvFindStereoCorrespondenceGC(left_p, _right, _dispLeft, _dispRight, state, 0);
    return;
}


} // end of namespace cv


namespace doppia
{

using namespace std;
using namespace boost;
using namespace boost::gil;

BOOST_STATIC_ASSERT(CV_MAJOR_VERSION == 2);

program_options::options_description OpenCvStereo::get_args_options()
{
    program_options::options_description desc("OpenCvStereo options");

    desc.add_options()

            ("gc_max_iterations",
             program_options::value<int>()->default_value(3), "maximum number of interations for the graph cut algorithm. Default value is 3");


    return desc;
}

OpenCvStereo::OpenCvStereo(const program_options::variables_map &options) :
    AbstractStereoBlockMatcher(options)
{

    if ((max_disparity % 16) != 0)
    {
        std::runtime_error("OpenCvStereo requires max_disparity to be positive and multiple of 16");
    }

    string method;

    if(options.count("method") > 0)
    {
        method = get_option_value<std::string>(options, "method");
    }
    else if(options.count("stereo.method") > 0)
    {
        method = get_option_value<std::string>(options, "stereo.method");
    }
    else if(options.count("cost_volume.method") > 0)
    {
        method = get_option_value<std::string>(options, "cost_volume.method");
    }
    else
    {
        throw std::invalid_argument("OpenCvStereo failed to deduce the stereo matching method to use");
    }

    if(method.compare("opencv_sad") == 0 or
            method.compare("opencv_bm") == 0 or
            method.compare("sad") == 0 )
    {
        stereo_algorithm = BlockMatchingAlgorithm;
    }
    else if(method.compare("opencv_gc") == 0 or
            method.compare("gc") == 0 )
    {
        stereo_algorithm = GraphCutAlgorithm;
    }
    else if(method.compare("opencv_csbp") == 0 or
            method.compare("opencv_gpu_csbp") == 0 or
            method.compare("csbp") == 0 )
    {
        stereo_algorithm = CsbpAlgorithm;

#if defined(USE_GPU)
        // if opencv was not compiled with gpu support, this constructor will raise an exception
        stereo_gpu_csbp_p.reset(new cv::gpu::StereoConstantSpaceBP(max_disparity));
#else
        throw std::runtime_error("This executable was compiled without GPU support");
#endif
    }
    else if(method.compare("opencv_bp") == 0)
    {
        stereo_algorithm = BeliefPropagationAlgorithm;
#if defined(USE_GPU)
        stereo_gpu_bp_p.reset(new cv::gpu::StereoBeliefPropagation(max_disparity));
#else
        throw std::runtime_error("This executable was compiled without GPU support");
#endif
    }
    else
    {
        printf("Received unknown method '%s'\n", method.c_str());
        throw std::invalid_argument("OpenCvStereo failed to retrieve a known stereo matching method");
    }

    //stereo_bm.ndisp = max_disparity;
    //stereo_gc.ndisp = max_disparity;
    gc_max_iterations = get_option_value<int>(options, "gc_max_iterations");

    return;
}


OpenCvStereo::~OpenCvStereo()
{

    // nothing to do here
    return;
}

void OpenCvStereo::set_rectified_images_pair( gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right)
{
    this->AbstractStereoMatcher::set_rectified_images_pair(left, right);


    return;
}


void OpenCvStereo::compute_disparity_map(
        gray8c_view_t &left_view, gray8c_view_t &right_view,
        bool left_right_are_inverted)
{    
    compute_disparity_map_impl(left_view, right_view, left_right_are_inverted);
    return;
}

void OpenCvStereo::compute_disparity_map(
        rgb8c_view_t  &left_view, rgb8c_view_t &right_view,
        bool left_right_are_inverted)
{
    compute_disparity_map_impl(left_view, right_view, left_right_are_inverted);
    return;
}


template<typename ImgView>
void OpenCvStereo::compute_disparity_map_impl(
        ImgView &left_view, ImgView &right_view,
        bool left_right_are_inverted)
{

    if (left_right_are_inverted)
    {
        throw std::runtime_error("OpenCvStereo does not implement right to left matching yet");
    }

    // num_disparities should positive and multiple of 16
    const int num_disparities = max_disparity;

    // will violate the constness of the view
    opencv::ipl_image_wrapper imgLeft = opencv::create_ipl_image(left_view);
    opencv::ipl_image_wrapper imgRight = opencv::create_ipl_image(right_view);

    // copy and transform the format
    const cv::Mat left_mat = imgLeft.get(), right_mat = imgRight.get();
    cv::Mat disparity_out(left_mat.size(), CV_8U);

    static int num_iterations = 0;
    static double cumulated_time = 0;
    const int num_iterations_for_timing = 50;

    switch(stereo_algorithm)
    {
    case BlockMatchingAlgorithm:
    {
        const int preset = cv::StereoBM::BASIC_PRESET;
        //const int preset = cv::StereoBM::NARROW_PRESET;
        //const int preset = cv::StereoBM::FISH_EYE_PRESET;
        const int sad_window_size = this->window_width;
        stereo_bm.init(preset, num_disparities, sad_window_size);

        const double start_wall_time = omp_get_wtime();
        stereo_bm(left_mat, right_mat, disparity_out);
        cumulated_time += omp_get_wtime() - start_wall_time;

        disparity_out = disparity_out / 16.0;
        // after dividing 16 the ocluded values are marked as -1
        // when copying this pixels -1 will become 255 (via overflow),
        // which is exactly the value we wanted
    }
        break;

    case GraphCutAlgorithm:
    {
        stereo_gc.init(num_disparities, gc_max_iterations);

        const double start_wall_time = omp_get_wtime();
        stereo_gc(left_mat, right_mat, disparity_out);
        cumulated_time += omp_get_wtime() - start_wall_time;
        printf("stereo_gc finished\n");

        disparity_out *= -1;
    }
        break;

    case CsbpAlgorithm:
    {
#if defined(USE_GPU)
        assert(stereo_gpu_csbp_p);

        cv::gpu::GpuMat left_gpu_mat(left_mat), right_gpu_mat(right_mat);
        cv::gpu::GpuMat disparity_gpu_out(left_mat.size(), CV_8U);

        const double start_wall_time = omp_get_wtime();
        (*stereo_gpu_csbp_p)(left_gpu_mat, right_gpu_mat, disparity_gpu_out);
        cumulated_time += omp_get_wtime() - start_wall_time;

        disparity_gpu_out.download(disparity_out);

        //cv::imshow("gpu disparity", disparity_out);
        //cv::waitKey(3);
#else
        throw std::runtime_error("This executable was compiled without GPU support");
#endif
    }
        break;

    case BeliefPropagationAlgorithm:
    {
#if defined(USE_GPU)
        assert(stereo_gpu_csbp_p);

        cv::gpu::GpuMat left_gpu_mat(left_mat), right_gpu_mat(right_mat);
        cv::gpu::GpuMat disparity_gpu_out(left_mat.size(), CV_8U);

        const double start_wall_time = omp_get_wtime();
        (*stereo_gpu_bp_p)(left_gpu_mat, right_gpu_mat, disparity_gpu_out);
        cumulated_time += omp_get_wtime() - start_wall_time;

        disparity_gpu_out.download(disparity_out);

        //cv::imshow("gpu disparity", disparity_out);
        //cv::waitKey(3);
#else
        throw std::runtime_error("This executable was compiled without GPU support");
#endif
    }
        break;


    default:
        throw std::runtime_error("OpenCvStereo::compute_disparity_map_impl received an unknown algorithm type");
        break;
    }

    if(disparity_out.type() == CV_16SC1)
    {
        gil::gray16sc_view_t disparity_out_view = interleaved_view(disparity_out.cols, disparity_out.rows,
                                                                   reinterpret_cast<gil::gray16sc_pixel_t*>(disparity_out.data), disparity_out.step);
        //copy_and_convert_pixels(disparity_out_view, disparity_map_view);
        copy_pixels(disparity_out_view, disparity_map_view);
    }
    else if(disparity_out.type() == CV_8UC1)
    {
        gil::gray8c_view_t disparity_out_view = interleaved_view(disparity_out.cols, disparity_out.rows,
                                                                 reinterpret_cast<gil::gray8c_pixel_t*>(disparity_out.data), disparity_out.step);
        //copy_and_convert_pixels(disparity_out_view, disparity_map_view);
        copy_pixels(disparity_out_view, disparity_map_view);
    }
    else
    {
        printf("disparity_out.type()  == %i\n", disparity_out.type());
        //throw std::runtime_error("OpenCvStereo::compute_disparity_map opencv stereo method did not return the expected type");
    }


    num_iterations += 1;
    if((num_iterations % num_iterations_for_timing) == 0)
    {
        printf("Average OpenCvStereo::compute_disparity_map_impl speed  %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
        cumulated_time = 0;
    }

    return;
}

} // end of namespace doppia


