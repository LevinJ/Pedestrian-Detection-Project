#include "GpuIntegralChannelsForPedestrians.hpp"

#if not defined(USE_GPU)
#error "Should use -DUSE_GPU to compile this file"
#endif

#include "gpu/integral_channels.cu.hpp"
#include "gpu/shrinking.cu.hpp"

#include "helpers/ModuleLog.hpp"
#include "helpers/gpu/cuda_safe_call.hpp"
#include "helpers/fill_multi_array.hpp"
#include "helpers/get_option_value.hpp"

#include <boost/type_traits/is_same.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <cudatemplates/hostmemoryreference.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememoryreference.hpp>
#include <cudatemplates/stream.hpp>

#include <opencv2/highgui/highgui.hpp> // for debugging purposes

#include <numeric>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "GpuIntegralChannels");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "GpuIntegralChannels");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "GpuIntegralChannels");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "GpuIntegralChannels");
}

} // end of anonymous namespace


namespace doppia {

using namespace cv;
using namespace cv::gpu;

typedef boost::gil::rgb8_pixel_t pixel_t;

// this can only be defined in the *.cu files
// Cuda::Array2D<pixel_t>::Texture input_image_texture;

typedef GpuIntegralChannelsForPedestrians::filter_shared_pointer_t filter_shared_pointer_t;

filter_shared_pointer_t
create_pre_smoothing_gpu_filter()
{
    //const int binomial_filter_degree = 0;
    const int binomial_filter_degree = 1;
    //const int binomial_filter_degree = 2;

    const bool copy_data = true;
    const cv::Mat binomial_kernel_1d = cv::Mat(get_binomial_kernel_1d(binomial_filter_degree), copy_data);
    filter_shared_pointer_t pre_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC4, CV_8UC4, binomial_kernel_1d, binomial_kernel_1d);

    // using the 2d kernel is significantly faster than using the linear separable kernel
    // (at least for binomial_filter_degree == 1)
    // on OpenCv 2.3 svn trunk @ yildun, when detecting over 52 scales,
    // we get 0.89 Hz with 1d kernel and 0.98 Hz with 2d kernel
    // ( 2.45 Hz versus 2.55 Hz on jabbah)
    const bool use_2d_kernel = true;
    if(use_2d_kernel)
    {
        const cv::Mat binomial_kernel_2d =  binomial_kernel_1d * binomial_kernel_1d.t();

        const cv::Point default_anchor = cv::Point(-1,-1);
        const int border_type = BORDER_DEFAULT;
        //const int border_type = BORDER_REPLICATE;
        //const int border_type = BORDER_REFLECT;
        pre_smoothing_filter_p =
                cv::gpu::createLinearFilter_GPU(CV_8UC4, CV_8UC4, binomial_kernel_2d, default_anchor, border_type);
    }

    return pre_smoothing_filter_p;
}


GpuIntegralChannelsForPedestrians::GpuIntegralChannelsForPedestrians(
        const boost::program_options::variables_map &options,
        const bool use_presmoothing_)
    : AbstractGpuIntegralChannelsComputer(get_shrinking_factor()),
      num_hog_angle_bins(get_option_value<int>(options, "channels.num_hog_angle_bins")),
      use_presmoothing(use_presmoothing_)
{

    if ((num_hog_angle_bins != 2)
            and (num_hog_angle_bins != 6)
            and (num_hog_angle_bins != 18))
    {
        log_error() << "Received channels.num_hog_angle_bins == " << num_hog_angle_bins
                    << " but only values 2, 6, and 18 are currently supported"
                    << std::endl;
        throw std::invalid_argument("Requested a value channels.num_hog_angle_bins "
                                    "not currently supported by GpuIntegralChannelsForPedestrians");
    }

    if(use_presmoothing)
    {
        pre_smoothing_filter_p = create_pre_smoothing_gpu_filter();
    }
    return;
}


GpuIntegralChannelsForPedestrians::GpuIntegralChannelsForPedestrians(const size_t num_hog_angle_bins_,
                                                                     const bool use_presmoothing_)
    : AbstractGpuIntegralChannelsComputer(get_shrinking_factor()),
      num_hog_angle_bins(num_hog_angle_bins_),
      use_presmoothing(use_presmoothing_)
{

    if ((num_hog_angle_bins != 2)
            and (num_hog_angle_bins != 6)
            and (num_hog_angle_bins != 18))
    {
        log_error() << "Received channels.num_hog_angle_bins == " << num_hog_angle_bins
                    << " but only values 2, 6, and 18 are currently supported"
                    << std::endl;
        throw std::invalid_argument("Requested a value channels.num_hog_angle_bins "
                                    "not currently supported by GpuIntegralChannelsForPedestrians");
    }

    if(use_presmoothing)
    {
        pre_smoothing_filter_p = create_pre_smoothing_gpu_filter();
    }

    return;

}


GpuIntegralChannelsForPedestrians::~GpuIntegralChannelsForPedestrians()
{
    // nothing to do here
    return;
}


int GpuIntegralChannelsForPedestrians::get_shrinking_factor()
{
    return IntegralChannelsForPedestrians::get_shrinking_factor();
}


size_t GpuIntegralChannelsForPedestrians::get_num_channels() const
{
    return num_hog_angle_bins + 4;
}



void GpuIntegralChannelsForPedestrians::compute_v0()
{
    // v0 is mainly based on OpenCv's GpuMat

    // smooth the input image --
    compute_smoothed_image_v0();

    // compute the HOG channels --
    compute_hog_channels_v0();

    // compute the LUV channels --
    compute_luv_channels_v0();

    // resize and compute integral images for each channel --
    resize_and_integrate_gpu_channels_v0();

    return;
}


void GpuIntegralChannelsForPedestrians::compute_smoothed_image_v0()
{
    // smooth the input image --
    smoothed_input_gpu_mat.create(input_gpu_mat.size(), input_gpu_mat.type());

    if(use_presmoothing and pre_smoothing_filter_p)
    {
        // we need to force the filter to be applied to the borders
        // (by default it omits them)
        const Rect region_of_interest = Rect(0,0, input_gpu_mat.cols, input_gpu_mat.rows);
        pre_smoothing_filter_p->apply(input_gpu_mat, smoothed_input_gpu_mat, region_of_interest);
    }
    else
    {
        input_gpu_mat.copyTo(smoothed_input_gpu_mat); // simple copy, no filtering
    }

    return;
}


void GpuIntegralChannelsForPedestrians::compute_hog_channels_v0()
{
    if(false)
    {
        printf("input_size.x == %zi, input_size.y == %zi\n",
               input_size.x, input_size.y);
        printf("channels.size == [%zi, %zi, %zi]\n",
               gpu_channels.size[0], gpu_channels.size[1], gpu_channels.size[2]);
        printf("channels.stride == [%zi, %zi, %zi]\n",
               gpu_channels.stride[0], gpu_channels.stride[1], gpu_channels.stride[2]);

        throw std::runtime_error("Stopped everything so you can inspect the printed vaules");
    }

    cv::gpu::cvtColor(smoothed_input_gpu_mat, hog_input_gpu_mat, CV_RGBA2GRAY);

    if(hog_input_gpu_mat.type() != CV_8UC1)
    {
        printf("compute_hog_channels(...) input_image.type() == %i\n", hog_input_gpu_mat.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::invalid_argument("doppia::integral_channels::compute_hog_channels expects an input image of type CV_8UC1");
    }

    // compute the HOG channels  --
    switch(num_hog_angle_bins)
    {
    case 6:
        //doppia::integral_channels::compute_hog6_channels(hog_input_gpu_mat, gpu_channels);
        doppia::integral_channels::compute_hog_channels(hog_input_gpu_mat, gpu_channels);
        break;
    default:
        throw std::invalid_argument("GpuIntegralChannelsForPedestrians::compute_hog_channels_v0 "
                                    "called with an unsupported value for num_hog_angle_bins");
        break;
    }

    return;
}


void GpuIntegralChannelsForPedestrians::compute_luv_channels_v0()
{
    // compute the LUV channels --

    const bool use_opencv = false;

    if(use_opencv)
    {

        if (num_hog_angle_bins != 6)
        {
            throw std::runtime_error("This part of code does not (yet) handle num_hog_angle_bins != 6");
        }

        // CV_RGB2HSV and CV_RGB2Luv seem to work fine even when the input is RGBA
        //cv::gpu::cvtColor(smoothed_input_gpu_mat, luv_gpu_mat, CV_RGB2Luv);

        // warning doing HSV until LUV is actually implemented
        cv::gpu::cvtColor(smoothed_input_gpu_mat, luv_gpu_mat, CV_RGB2HSV);

        // split the LUV image into the L,U and V channels
        std::vector<GpuMat> destinations(3);

        destinations[0] = get_slice(gpu_channels, 7);
        destinations[1] = get_slice(gpu_channels, 8);
        destinations[2] = get_slice(gpu_channels, 9);

        cv::gpu::split(luv_gpu_mat, destinations);

        if(false)
        {
            cv::Mat test_image;
            luv_gpu_mat.download(test_image);
            cv::imwrite("debug_image.png", test_image);
            throw std::runtime_error("Stopped everything so you can inspect debug_image.png");
        }

    }
    else
    {
        if(smoothed_input_gpu_mat.type() != CV_8UC4)
        {
            throw std::invalid_argument("doppia::integral_channels::compute_luv_channels expects an RGBA image as input");
        }

        switch(num_hog_angle_bins)
        {
        case 6:
            //doppia::integral_channels::compute_luv_channels_after_hog6(smoothed_input_gpu_mat, gpu_channels);
            doppia::integral_channels::compute_luv_channels(smoothed_input_gpu_mat, gpu_channels);
            break;
        default:
            throw std::invalid_argument("GpuIntegralChannelsForPedestrians::compute_luv_channels_v0 "
                                        "called with an unsupported value for num_hog_angle_bins");
            break;
        }

    }
    return;
}


void GpuIntegralChannelsForPedestrians::shrink_gpu_channel_v0(
        GpuMat &gpu_feature_channel, GpuMat &shrunk_gpu_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{
    // we assume that gpu_shrunk_channel is already properly allocated
    assert(shrunk_gpu_channel.empty() == false);

    if(shrinking_factor == 4)
    {
        cv::gpu::pyrDown(gpu_feature_channel, one_half, channel_stream);
        cv::gpu::pyrDown(one_half, shrunk_gpu_channel, channel_stream); // shrunk_channel == one_fourth
    }
    else if (shrinking_factor == 2)
    {
        cv::gpu::pyrDown(gpu_feature_channel, one_half, channel_stream);
        shrunk_gpu_channel = one_half;
    }
    else
    {
        shrunk_gpu_channel = gpu_feature_channel;
    }

    return;
}


void GpuIntegralChannelsForPedestrians::shrink_gpu_channel_v1(
        GpuMat &gpu_feature_channel, GpuMat &shrunk_gpu_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{

    // FIXME need to add tests to verify that _v1 and _v0 output the same results

    // lazy allocation
    //shrunk_gpu_hannel.create(shrunk_gpu_channel_size.y, shrunk_gpu_channel_size.x, gpu_feature_channel.type());
    // we assume that shrunk_gpu_channel is already properly allocated
    //assert(shrunk_gpu_channel.empty() == false);

    shrink_gpu_channel_buffer_a.create(gpu_feature_channel.rows, gpu_feature_channel.cols, gpu_feature_channel.type());
    shrink_gpu_channel_buffer_b.create((gpu_feature_channel.rows + 1) / 2, (gpu_feature_channel.cols+1) / 2, gpu_feature_channel.type());

    static const float pyrDown_kernel_1d_values[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625},
            pyrDown4_kernel_1d_values[9] = {
        0.01483945, 0.04981729, 0.11832251, 0.198829, 0.23638351, 0.198829, 0.11832251, 0.04981729, 0.01483945 };

    static const vector<float>
            t_vec(&pyrDown_kernel_1d_values[0], &pyrDown_kernel_1d_values[5]),
            t_vec4(&pyrDown4_kernel_1d_values[0], &pyrDown4_kernel_1d_values[9]);

    static const cv::Mat
            pyrDown_kernel_1d = cv::Mat(t_vec, true),
            pyrDown4_kernel_1d = cv::Mat(t_vec4, true);
    //pyrDown_kernel_2d =  pyrDown_kernel_1d * pyrDown_kernel_1d.t();
    static const cv::Rect dummy_roi = cv::Rect(0,0,-1,-1);
    static filter_shared_pointer_t pyrDown_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown_kernel_1d, pyrDown_kernel_1d);
    //cv::gpu::createLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown_kernel_2d);

    static filter_shared_pointer_t pyrDown4_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown4_kernel_1d, pyrDown4_kernel_1d);
    //cv::gpu::createLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown4_kernel_2d);

    if(shrinking_factor == 4)
    {

        const bool use_one_filter = true;

        if(use_one_filter)
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown4_smoothing_filter_p->apply(gpu_feature_channel, shrink_gpu_channel_buffer_a,
                                               dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_a, shrunk_gpu_channel,
                            cv::Size(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        else
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown_smoothing_filter_p->apply(gpu_feature_channel, shrink_gpu_channel_buffer_a,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_a, one_half,
                            shrink_gpu_channel_buffer_b.size(), 0, 0, cv::INTER_NEAREST, channel_stream);

            pyrDown_smoothing_filter_p->apply(one_half, shrink_gpu_channel_buffer_b,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_b, shrunk_gpu_channel,
                            cv::Size(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        // shrunk_gpu_channel == one_fourth
    }
    else
    {
        throw std::runtime_error("shrink_channel_v1 only supports shrinking factor == 4");
    }

    return;
}

void GpuIntegralChannelsForPedestrians::shrink_gpu_channel_v2(
        GpuMat &gpu_feature_channel, GpuMat &shrunk_gpu_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{

    // _v2 is based on the realization that for detection purposes we do not want to do a nice Gaussian filter + subsampling,
    // actually we want to do an average pooling, so simply averaging over a box filter + subsampling is both correct and faster


    shrink_gpu_channel_buffer_a.create(gpu_feature_channel.rows, gpu_feature_channel.cols, gpu_feature_channel.type());
    shrink_gpu_channel_buffer_b.create((gpu_feature_channel.rows + 1) / 2, (gpu_feature_channel.cols+1) / 2, gpu_feature_channel.type());

    static const cv::Rect dummy_roi = cv::Rect(0,0,-1,-1);
    static filter_shared_pointer_t pyrDown_smoothing_filter_p =
            cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(2,2));

    static filter_shared_pointer_t pyrDown4_smoothing_filter_p =
            cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(4, 4));

    if(shrinking_factor == 4)
    {

        // FIXME based on the Nvidia forum answer, using the smoothing filter here is useless
        // http://forums.nvidia.com/index.php?showtopic=210066

        const bool use_one_filter = false;
        if(use_one_filter)
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown4_smoothing_filter_p->apply(gpu_feature_channel, shrink_gpu_channel_buffer_a,
                                               dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_a, shrunk_gpu_channel,
                            cv::Size(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        else
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown_smoothing_filter_p->apply(gpu_feature_channel, shrink_gpu_channel_buffer_a,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_a, one_half,
                            shrink_gpu_channel_buffer_b.size(), 0, 0, cv::INTER_NEAREST, channel_stream);

            pyrDown_smoothing_filter_p->apply(one_half, shrink_gpu_channel_buffer_b,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_gpu_channel_buffer_b, shrunk_gpu_channel,
                            cv::Size(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        // shrunk_gpu_channel == one_fourth
    }
    else
    {
        throw std::runtime_error("shrink_channel_v2 only supports shrinking factor == 4");
    }

    return;
}

/// Helper method to handle OpenCv's GPU data
template <typename DstType>
Cuda::DeviceMemoryReference2D<DstType> gpumat_to_device_reference_2d(cv::gpu::GpuMat &src)
{
    Cuda::Layout<DstType, 2> layout(Cuda::Size<2>(src.cols, src.rows));
    layout.setPitch(src.step); // step is in bytes
    return Cuda::DeviceMemoryReference2D<DstType>(layout, reinterpret_cast<DstType *>(src.data));
}


void GpuIntegralChannelsForPedestrians::shrink_gpu_channel_v3(
        GpuMat &gpu_feature_channel, GpuMat &shrunk_gpu_channel,
        const int shrinking_factor, cv::gpu::Stream &/*channel_stream*/)
{
    // the channel_stream argument is currently ignored

    assert(gpu_feature_channel.type() == CV_8UC1);
    assert(shrunk_gpu_channel.type() == CV_8UC1);

    // lazy allocation
    shrunk_gpu_channel.create(shrunk_gpu_channel_size.y, shrunk_gpu_channel_size.x, gpu_feature_channel.type());

    Cuda::DeviceMemoryReference2D<uint8_t>
            gpu_feature_channel_reference = gpumat_to_device_reference_2d<uint8_t>(gpu_feature_channel),
            shrunk_gpu_channel_reference = gpumat_to_device_reference_2d<uint8_t>(shrunk_gpu_channel);

    doppia::integral_channels::shrink_channel(gpu_feature_channel_reference, shrunk_gpu_channel_reference, shrinking_factor);

    return;
}

void GpuIntegralChannelsForPedestrians::resize_and_integrate_gpu_channels_v0()
{
    // using streams here seems to have zero effect on the computation speed (at Jabbah and the Europa laptop)
    // we keep "just in case" we run over a 2 GPUs system
    cv::gpu::Stream stream_a, stream_b;

    // shrink and compute integral over the channels ---
    for(size_t channel_index = 0; channel_index < gpu_channels.size[2]; channel_index +=1)
    {
        // we do not wait last channel completion to launch the new computation
        cv::gpu::Stream *channel_stream_p = &stream_a;
        cv::gpu::GpuMat *integral_channel_buffer_p = &gpu_integral_channel_buffer_a;
        if((channel_index % 2) == 1)
        {
            channel_stream_p = &stream_b;
            integral_channel_buffer_p = &gpu_integral_channel_buffer_b;
        }

        cv::gpu::Stream &channel_stream = *channel_stream_p;

        gpu::GpuMat feature_channel = get_slice(gpu_channels, channel_index);

        // v0 is about twice as fast as v1
        // v2 is significantly slower than v0 (1.7 Hz versus 2.3 Hz on detection rate)
        // v3 is brutaly faster, 4.5 Hz versus 3.8 Hz on Jabbah
        //shrink_channel_v0(feature_channel, shrunk_channel, resizing_factor, channel_stream);
        //shrink_channel_v1(feature_channel, shrunk_channel, resizing_factor, channel_stream);
        //shrink_channel_v2(feature_channel, shrunk_channel, resizing_factor, channel_stream);
        shrink_gpu_channel_v3(feature_channel, shrunk_gpu_channel, resizing_factor, channel_stream);


        const bool check_channels_sizes = false;
        if(check_channels_sizes)
        {
            if((shrunk_gpu_channel.rows != shrunk_gpu_channel_size.y)
                    or (shrunk_gpu_channel.cols != shrunk_gpu_channel_size.x))
            {
                printf("shrunk_channel size == (%i, %i)\n",
                       shrunk_gpu_channel.cols, shrunk_gpu_channel.rows);
                printf("shrunk_channel size == (%zi, %zi) (expected value for shrunk_channel size)\n",
                       shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y);
                throw std::runtime_error("shrunk_channel size != expected shrunk channel size");
            }

            if((gpu_integral_channels.size[0] != static_cast<size_t>(shrunk_gpu_channel_size.x + 1))
                    or (gpu_integral_channels.size[1] != static_cast<size_t>(shrunk_gpu_channel_size.y + 1)))
            {
                printf("integral channel size == (%zi, %zi)\n",
                       gpu_integral_channels.size[0], gpu_integral_channels.size[1]);
                printf("shrunk_channel size == (%zi, %zi) (expected value for shrunk_channel size)\n",
                       shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y);
                throw std::runtime_error("integral channel size != shrunk channel size + 1");
            }
        } // end of "if check channels sizes"


        // set mini test or not --
        const bool set_test_integral_image = false;
        if(set_test_integral_image)
        { // dummy test integral image, used for debugging only
            // mimics the equivalent cpu snippet in IntegralChannelsForPedestrians::compute_v0()
            cv::Mat channel_test_values(shrunk_gpu_channel.size(), shrunk_gpu_channel.type());

            if(shrunk_gpu_channel.type() != CV_8UC1)
            {
                throw std::runtime_error("shrunk_channel.type() has an unexpected value");
            }

            for(int row=0; row < channel_test_values.rows; row+=1)
            {
                for(int col=0; col < channel_test_values.cols; col+=1)
                {
                    const float row_scale = 100.0f/(channel_test_values.rows);
                    const float col_scale = 10.0f/(channel_test_values.cols);
                    channel_test_values.at<boost::uint8_t>(row,col) = \
                            static_cast<boost::uint8_t>(min(255.0f, row_scale*row + col_scale*col + channel_index));
                } // end of "for each col"
            } // end of "for each row"

            shrunk_gpu_channel.upload(channel_test_values);
        } // end of set_test_integral_image


        // compute integral images for shrunk_channel --
        gpu::GpuMat integral_channel = get_slice(gpu_integral_channels, channel_index);

        // sum will have CV_32S type, but will contain unsigned int values
        cv::gpu::integralBuffered(shrunk_gpu_channel, integral_channel, *integral_channel_buffer_p, channel_stream);

        cuda_safe_call( cudaGetLastError() );

        if(false and (channel_index == gpu_channels.size[2] - 1))
        {

            cv::Mat test_image;
            shrunk_gpu_channel.download(test_image);
            cv::imwrite("debug_image.png", test_image);

            cv::Mat integral_image, test_image2;

            if(true and channel_index > 0)
            {
                // we take the previous channel, to check if it was not erased
                gpu::GpuMat integral_channel2(get_slice(gpu_integral_channels, channel_index - 1));

                printf("integral_channel2 cols, rows, step, type, elemSize == %i, %i, %zi, %i, %zi\n",
                       integral_channel2.cols, integral_channel2.rows,
                       integral_channel2.step, integral_channel2.type(), integral_channel2.elemSize() );

                printf("integral_channel cols, rows, step, type, elemSize == %i, %i, %zi, %i, %zi\n",
                       integral_channel.cols, integral_channel.rows,
                       integral_channel.step, integral_channel.type(), integral_channel.elemSize() );

                printf("CV_8UC1 == %i, CV_32UC1 == NOT DEFINED,  CV_32SC1 == %i,  CV_32SC2 == %i, CV_USRTYPE1 == %i\n",
                       CV_8UC1, CV_32SC1, CV_32SC2, CV_USRTYPE1);

                integral_channel2.download(integral_image); // copy from GPU to CPU
            }
            else
            {
                integral_channel.download(integral_image); // copy from GPU to CPU
            }

            test_image2 = cv::Mat(shrunk_gpu_channel.size(), cv::DataType<float>::type);
            for(int y=0; y < test_image2.rows; y+=1)
            {
                for(int x = 0; x < test_image2.cols; x += 1)
                {
                    const uint32_t
                            a = integral_image.at<boost::uint32_t>(y,x),
                            b = integral_image.at<boost::uint32_t>(y+0,x+1),
                            c = integral_image.at<boost::uint32_t>(y+1,x+1),
                            d = integral_image.at<boost::uint32_t>(y+1,x+0);
                    test_image2.at<float>(y,x) = a +c -b -d;
                } // end of "for each column"
            } // end of "for each row"

            cv::imwrite("debug_image2.png", test_image2);

            throw std::runtime_error("Stopped everything so you can inspect debug_image.png and debug_image2.png");
        }

    } // end of "for each channel"

    stream_a.waitForCompletion();
    stream_b.waitForCompletion();

    return;
}


void GpuIntegralChannelsForPedestrians::resize_and_integrate_gpu_channels_v1()
{
    if(not boost::is_same<gpu_integral_channels_t, gpu_3d_integral_channels_t>::value)
    {
        throw std::runtime_error("resize_and_integrate_channels_v1 should only be used with gpu_3d_integral_channels_t, "
                                 "please check your code");
    }

    // first we shrink all the channels ---
    {
        doppia::integral_channels::shrink_channels(gpu_channels, shrunk_gpu_channels, resizing_factor);
    }

    // second, we compute the integral of each channel ---
    {
        // using streams here seems to have zero effect on the computation speed (at Jabbah and the Europa laptop)
        // we keep "just in case" we run over a 2 GPUs system
        cv::gpu::Stream stream_a, stream_b;

        // shrink the channels ---
        for(size_t channel_index = 0; channel_index < shrunk_gpu_channels.size[2]; channel_index +=1)
        {
            // we do not wait last channel completion to launch the new computation
            cv::gpu::Stream *channel_stream_p = &stream_a;
            cv::gpu::GpuMat *integral_channel_buffer_p = &gpu_integral_channel_buffer_a;
            if((channel_index % 2) == 1)
            {
                channel_stream_p = &stream_b;
                integral_channel_buffer_p = &gpu_integral_channel_buffer_b;
            }

            cv::gpu::Stream &channel_stream = *channel_stream_p;

            gpu::GpuMat shrunk_channel = get_slice(shrunk_gpu_channels, channel_index);

            // compute integral images for shrunk_channel --
            gpu::GpuMat integral_channel = get_slice(gpu_integral_channels, channel_index);

            //printf("DEPTH: %i, NUM. CHANNELS: %i, IS CONTINUOUS: %i, ROWS: %i, COLS: %i\n",
            //       integral_channel.rows, integral_channel.cols);

            // sum will have CV_32S type, but will contain unsigned int values
            cv::gpu::integralBuffered(shrunk_channel, integral_channel, *integral_channel_buffer_p, channel_stream);

            cuda_safe_call( cudaGetLastError() );
        } // end of "for each channel"

        stream_a.waitForCompletion();
        stream_b.waitForCompletion();
    }

    return;
}


void GpuIntegralChannelsForPedestrians::resize_and_integrate_gpu_channels_v2()
{
    if(not boost::is_same<gpu_integral_channels_t, gpu_2d_integral_channels_t>::value)
    {
        throw std::runtime_error("resize_and_integrate_channels_v2 should only be used with gpu_2d_integral_channels_t, "
                                 "please check your code");
    }

    // first we shrink all the channels ---
    {
        doppia::integral_channels::shrink_channels(gpu_channels, shrunk_gpu_channels, resizing_factor);

        const bool save_shrunk_channels = false;
        if(save_shrunk_channels)
        {
            const gpu_channels_t& gpu_channels = shrunk_gpu_channels;
            const Cuda::Size<3> &data_size = gpu_channels.getLayout().size;

            typedef GpuIntegralChannelsForPedestrians::channels_t  channels_t;
            channels_t cpu_shrunk_channels;
            // resize the CPU memory storage
            // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
            cpu_shrunk_channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

            // create cudatemplates reference
            Cuda::HostMemoryReference3D<channels_t::element>
                    cpu_channels_reference(data_size, cpu_shrunk_channels.origin());

            // copy from GPU to CPU --
            Cuda::copy(cpu_channels_reference, gpu_channels);

            const size_t
                    gpu_channels_width = cpu_shrunk_channels.shape()[2],
                    gpu_channels_height = cpu_shrunk_channels.shape()[0]*cpu_shrunk_channels.shape()[1];

            const boost::gil::gray8c_view_t shrunk_channels_view =
                    boost::gil::interleaved_view(
                        gpu_channels_width, gpu_channels_height,
                        reinterpret_cast<boost::gil::gray8c_pixel_t *>(cpu_shrunk_channels.data()),
                        sizeof(gpu_channels_t::Type)*gpu_channels_width);


            boost::gil::png_write_view("gpu_shrunk_channels_v2.png", shrunk_channels_view);

            throw std::runtime_error("GpuIntegralChannelsForPedestrians::resize_and_integrate_channels_v2 "
                                     "Stopping everything so you can inspect "
                                     "the created file gpu_shrunk_channels_v2.png");
        } // end of "save the shrunk channels"
    }

    {
        // second, we compute the integral of all channels in one shot ---
        const size_t
                shrunk_channels_area =  shrunk_gpu_channels.size[0]*shrunk_gpu_channels.size[1]*shrunk_gpu_channels.size[2],
                max_sum = shrunk_channels_area * std::numeric_limits<gpu_channels_t::Type>::max();

        // images of 1024*1024 over
        if(max_sum > std::numeric_limits<boost::uint32_t>::max())
        {
            printf("max_sum/max_uint32 value =~= %.4f \n",
                   static_cast<float>(max_sum) / std::numeric_limits<boost::uint32_t>::max());
            throw std::runtime_error("Using resize_and_integrate_channels_v2 will create an overflow, "
                                     "use resize_and_integrate_channels_v1 for this image size");
        }

        // size0 == x/cols, size1 == y/rows, size2 == num_channels
        // GpuMat(rows, cols, type)
        const gpu::GpuMat shrunk_channels_gpu_mat(
                    shrunk_gpu_channels.size[1]*shrunk_gpu_channels.size[2], shrunk_gpu_channels.size[0],
                cv::DataType<gpu_channels_t::Type>::type,
                shrunk_gpu_channels.getBuffer(),
                shrunk_gpu_channels.getPitch());

        gpu::GpuMat integral_channels_gpu_mat(
                    gpu_integral_channels.size[1], gpu_integral_channels.size[0],
                cv::DataType<gpu_integral_channels_t::Type>::type,
                gpu_integral_channels.getBuffer(),
                gpu_integral_channels.getPitch());

        // sum will have CV_32S type, but will contain unsigned int values
        cv::gpu::integralBuffered(shrunk_channels_gpu_mat, integral_channels_gpu_mat, gpu_integral_channel_buffer_a);

        cuda_safe_call( cudaGetLastError() );
    }

    return;
}


void GpuIntegralChannelsForPedestrians::compute_v1()
{
    // v1 is mainly based on v0,
    // but merges a few steps into single "bigger kernels" calls

    // smooth the input image --
    compute_smoothed_image_v0();

    // compute the HOG and LUV channels --
    compute_hog_and_luv_channels_v1();

    // resize and compute integral images for each channel --
    // with v1 we obtain 4.65 Hz versus 4.55 Hz with v0
    //resize_and_integrate_channels_v0();
    if(boost::is_same<gpu_integral_channels_t, gpu_3d_integral_channels_t>::value)
    {
        resize_and_integrate_gpu_channels_v1();
    }
    else if(boost::is_same<gpu_integral_channels_t, gpu_2d_integral_channels_t>::value)
    {
        resize_and_integrate_gpu_channels_v2();
    }
    else
    {
        throw std::invalid_argument("Received an unknown gpu_integral_channels_t");
    }

    return;
}


void GpuIntegralChannelsForPedestrians::compute_hog_and_luv_channels_v1()
{

    cv::gpu::cvtColor(smoothed_input_gpu_mat, hog_input_gpu_mat, CV_RGBA2GRAY);

    if(hog_input_gpu_mat.type() != CV_8UC1)
    {
        printf("compute_hog_and_luv_channels(...) input_gray_image.type() == %i\n", hog_input_gpu_mat.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::invalid_argument("doppia::integral_channels::compute_hog_and_luv_channels expects an input gray image of type CV_8UC1");
    }

    if(smoothed_input_gpu_mat.type() != CV_8UC4)
    {
        throw std::invalid_argument("doppia::integral_channels::compute_hog_luv_channels expects to have an RGBA image as an input");
    }

    switch(num_hog_angle_bins)
    {
    case 6:
        //doppia::integral_channels::compute_hog6_and_luv_channels(hog_input_gpu_mat, smoothed_input_gpu_mat, gpu_channels);
        doppia::integral_channels::compute_hog_and_luv_channels(hog_input_gpu_mat, smoothed_input_gpu_mat, gpu_channels);
        break;
    default:
        throw std::invalid_argument("GpuIntegralChannelsForPedestrians::compute_hog_and_luv_channels_v1 "
                                    "called with an unsupported value for num_hog_angle_bins");
        break;
    }

    return;
}


void GpuIntegralChannelsForPedestrians::save_channels_to_file()
{

    const string filename = "gpu_hog6_luv_integral_channels.png";
    save_integral_channels_to_file(get_integral_channels(), filename);
    log_info() << "Created image " << filename << std::endl;
    return;
}


void GpuIntegralChannelsForPedestrians::compute_channels_at_canonical_scales(const GpuMat &input_image,
                                                                             const std::string image_file_name)
{
    // nothing to do here, just defined to avoid exception raised from
    // AbstractGpuIntegralChannelsComputer::compute_channels_at_canonical_scales(...)
    return;
}


void GpuIntegralChannelsForPedestrians::compute()
{
    //compute_v0();
    compute_v1();


    //save_channels_to_file();
    //throw std::runtime_error("STOP TO INSPECT INTEGRAL CHANNELS");
    return;
}


/// helper function to access the computed channels on cpu
/// this is quite slow (large data transfer between GPU and CPU)
/// this method should be used for debugging only
const GpuIntegralChannelsForPedestrians::channels_t &GpuIntegralChannelsForPedestrians::get_channels()
{
    const gpu_channels_t& gpu_channels = get_gpu_channels();

    const Cuda::Size<3> &data_size = gpu_channels.getLayout().size;

    // resize the CPU memory storage --
    // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
    channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference3D<channels_t::element>
            channels_memory_reference(data_size, channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(channels_memory_reference, gpu_channels);

    return channels;
}


const AbstractChannelsComputer::input_channels_t &GpuIntegralChannelsForPedestrians::get_input_channels_uint8()
{
    return get_channels();
}


const AbstractIntegralChannelsComputer::channels_t &GpuIntegralChannelsForPedestrians::get_input_channels_uint16()
{
    //throw std::runtime_error("GpuIntegralChannelsForPedestrians does not implement get_input_channels_uint16");
    //static AbstractIntegralChannelsComputer::channels_t empty;
    //return empty;

    // copy the content inside the channels_uint16
    const channels_t &computed_channels = get_channels();

    uint8_to_uint16_channels(computed_channels, channels_uint16);
    return channels_uint16;
}


} // end of namespace doppia
