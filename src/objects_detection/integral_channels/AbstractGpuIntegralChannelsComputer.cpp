#include "AbstractGpuIntegralChannelsComputer.hpp"
#include <cudatemplates/devicememoryreference.hpp>

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include "helpers/ModuleLog.hpp"

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

AbstractGpuIntegralChannelsComputer::AbstractGpuIntegralChannelsComputer(const int resizing_factor_)
    : AbstractIntegralChannelsComputer(),
      resizing_factor(resizing_factor_)
{
    return;
}

AbstractGpuIntegralChannelsComputer::~AbstractGpuIntegralChannelsComputer()
{
    // nothing to do here
    return;
}


AbstractGpuIntegralChannelsComputer::gpu_integral_channels_t&
AbstractGpuIntegralChannelsComputer::get_gpu_integral_channels()
{
    return gpu_integral_channels;
}


void AbstractGpuIntegralChannelsComputer::set_canonical_scales_for_stuff(const boost::program_options::variables_map &options)
{

    float
            min_canonical_scale_for_stuff = get_option_value<float>(options, "pixel_labeler.min_scale"),
            max_canonical_scale_for_stuff = get_option_value<float>(options, "pixel_labeler.max_scale");
    int num_scales_for_stuff = get_option_value<int>(options, "pixel_labeler.num_scales");

    float scale_logarithmic_step = 0;
    if(num_scales_for_stuff > 1)
    {
        scale_logarithmic_step = (log(max_canonical_scale_for_stuff) - log(min_canonical_scale_for_stuff)) / (num_scales_for_stuff -1);
    }

    canonical_scales_for_stuff.push_back(min_canonical_scale_for_stuff);
    for(int i=1; i<num_scales_for_stuff; i+= 1)
    {
        canonical_scales_for_stuff.push_back(std::min(max_canonical_scale_for_stuff, expf(logf(canonical_scales_for_stuff[i-1]) + scale_logarithmic_step)));
    }

    return;
}


void AbstractGpuIntegralChannelsComputer::set_canonical_scales_for_things(const boost::program_options::variables_map &options)
{

    float
            min_canonical_scale_for_things = get_option_value<float>(options, "objects_detector.min_scale"),
            max_canonical_scale_for_things = get_option_value<float>(options, "objects_detector.max_scale");
    int num_scales_for_things = get_option_value<int>(options, "objects_detector.num_scales");

    float scale_logarithmic_step = 0;
    if(num_scales_for_things > 1)
    {
        scale_logarithmic_step = (log(max_canonical_scale_for_things) - log(min_canonical_scale_for_things)) / (num_scales_for_things -1);
    }

    canonical_scales_for_things.push_back(min_canonical_scale_for_things);
    for(int i=1; i<num_scales_for_things; i+= 1)
    {
        canonical_scales_for_things.push_back(std::min(max_canonical_scale_for_things, expf(logf(canonical_scales_for_things[i-1]) + scale_logarithmic_step)));
        //printf("scale %i: %f\n", i, canonical_scales_for_things[i]);
    }

    return;
}

void AbstractGpuIntegralChannelsComputer::set_input_image(const AbstractChannelsComputer::input_image_view_t &input_view)
{
    // transfer image into GPU --
    boost::gil::opencv::ipl_image_wrapper input_ipl =
            boost::gil::opencv::create_ipl_image(input_view);

    cv::Mat input_mat(input_ipl.get());

    const bool use_cuda_write_combined = true;
    if(use_cuda_write_combined)
    {
        if((input_rgb8_gpu_mem.rows != input_mat.rows) or (input_rgb8_gpu_mem.cols != input_mat.cols))
        {
            // lazy allocate the cuda memory
            // using WRITE_COMBINED, in theory allows for 40% speed-up in the upload,
            // (but reading this memory from host will be _very slow_)
            // tests on the laptop show no speed improvement (maybe faster on desktop ?)
            input_rgb8_gpu_mem.create(input_mat.size(), input_mat.type(), cv::gpu::CudaMem::ALLOC_WRITE_COMBINED);
            input_rgb8_gpu_mem_mat = input_rgb8_gpu_mem.createMatHeader();
        }

        input_mat.copyTo(input_rgb8_gpu_mem_mat); // copy to write_combined host memory
        input_rgb8_gpu_mat.upload(input_rgb8_gpu_mem_mat);  // fast transfer from CPU to GPU
    }
    else
    {
        input_rgb8_gpu_mat.upload(input_mat);  // from CPU to GPU
    }

    // most tasks in GPU are optimized for CV_8UC1 and CV_8UC4, so we set the input as such
    cv::gpu::cvtColor(input_rgb8_gpu_mat, input_gpu_mat, CV_RGB2RGBA); // GPU type conversion

    if(input_gpu_mat.type() != CV_8UC4)
    {
        throw std::runtime_error("cv::gpu::cvtColor did not work as expected");
    }

    allocate_gpu_channels(input_view.width(), input_view.height());

    return;
}


void AbstractGpuIntegralChannelsComputer::set_gpu_image(const cv::gpu::GpuMat &input_image)
{
    if(input_image.type() != CV_8UC4)
    {
        printf("GpuIntegralChannelsForPedestrians::set_image input_image.type() == %i\n",
               input_image.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::runtime_error("OpenCv gpu module handles color images as CV_8UC4, received an unexpected GpuMat type");
    }

    const bool input_size_changed = (input_gpu_mat.cols != input_image.cols) or (input_gpu_mat.rows != input_image.rows);
    input_gpu_mat = input_image;

    if(input_size_changed)
    {
        allocate_gpu_channels(input_image.cols, input_image.rows);
    }
    else
    {
        // no need to reallocated, only need to reset a few matrices

        // we reset the content of the channels to zero
        // this is particularly important for the HOG channels where not all the pixels will be set
        gpu_channels.initMem(0);
    }
    return;
}

template <typename T>
void allocate_gpu_integral_channels(const size_t /*shrunk_channel_size_x*/,
                                const size_t /*shrunk_channel_size_y*/,
                                const size_t /*num_channels*/,
                                T &/*gpu_integral_channels*/)
{
    throw std::runtime_error("Called alloc_integral_channels<> with an unhandled integral channel type");
    return;
}


template <>
void allocate_gpu_integral_channels<AbstractGpuIntegralChannelsComputer::gpu_3d_integral_channels_t>(
        const size_t shrunk_channel_size_x,
        const size_t shrunk_channel_size_y,
        const size_t num_channels,
        AbstractGpuIntegralChannelsComputer::gpu_3d_integral_channels_t &gpu_integral_channels)
{
    gpu_integral_channels.realloc(shrunk_channel_size_x+1, shrunk_channel_size_y+1, num_channels);
    // size0 == x/cols, size1 == y/rows, size2 == num_channels
    return;
}


template <>
void allocate_gpu_integral_channels<AbstractGpuIntegralChannelsComputer::gpu_2d_integral_channels_t>(
        const size_t shrunk_channel_size_x,
        const size_t shrunk_channel_size_y,
        const size_t num_channels,
        AbstractGpuIntegralChannelsComputer::gpu_2d_integral_channels_t &gpu_integral_channels)
{
    gpu_integral_channels.alloc(shrunk_channel_size_x+1, (shrunk_channel_size_y*num_channels) + 1);
    // size0 == x/cols, size1 == y/rows, size2 == num_channels

    gpu_integral_channels.height = shrunk_channel_size_y;
    return;
}


void AbstractGpuIntegralChannelsComputer::allocate_gpu_channels(const size_t image_width, const size_t image_height)
{
    input_size.x = image_width;
    input_size.y = image_height;

    //channel_size = input_image.dimensions() / resizing_factor;
    // +resizing_factor/2 to round-up
    if(resizing_factor == 4)
    {
        shrunk_gpu_channel_size.x = (( (image_width+1) / 2) + 1) / 2;
        shrunk_gpu_channel_size.y = (( (image_height+1) / 2) + 1) / 2;
    }
    else if(resizing_factor == 2)
    {
        shrunk_gpu_channel_size.x = (image_width+1) / 2;
        shrunk_gpu_channel_size.y = (image_height+1) / 2;
    }
    else
    {
        shrunk_gpu_channel_size = input_size;
    }

    static int num_calls = 0;
    if(num_calls < 100)
    { // we only log the first N calls
        log_debug() << "Input image dimensions (" << image_width << ", " << image_height << ")" << std::endl;
        log_debug() << "Shrunk Channel size (" << shrunk_gpu_channel_size.x << ", " << shrunk_gpu_channel_size.y << ")" << std::endl;
        num_calls += 1;
    }

    if(shrunk_gpu_channel_size.x == 0 or shrunk_gpu_channel_size.y == 0)
    {
        log_error() << "Input image dimensions (" << image_width << ", " << image_height << ")" << std::endl;
        throw std::runtime_error("Input image for GpuIntegralChannelsForPedestrians::set_image "
                                 "was too small");
    }

    const size_t num_channels = get_num_channels();

    // FIXME should allocate only once (for the largest size)
    // and then reuse the allocated memory with smaller images
    // e.g. allocate only for "bigger images", and reuse memory for smaller images
    // Cuda::DeviceMemoryPitched3D<> for allocations and Cuda::DeviceMemoryReference3D<> on top, or similar..

    // allocate the channel images, the first dimension goes last
    gpu_channels.realloc(input_size.x, input_size.y, num_channels);
    // if not using shrunk_channels, this allocation should be commented out (since it takes time)
    shrunk_gpu_channels.realloc(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y, num_channels);

    allocate_gpu_integral_channels(shrunk_gpu_channel_size.x, shrunk_gpu_channel_size.y, num_channels,
                               gpu_integral_channels);

    // we reset the content of the channels to zero
    // this is particularly important for the HOG channels where not all the pixels will be set
    gpu_channels.initMem(0);

    const bool print_slice_size = false;
    if(print_slice_size)
    {
        printf("channels.getSlice(0).size[0] == %zi\n", gpu_channels.getSlice(0).size[0]);
        printf("channels.getSlice(0).size[1] == %zi\n", gpu_channels.getSlice(0).size[1]);
        printf("channels.getSlice(9).size[0] == %zi\n", gpu_channels.getSlice(9).size[0]);
        printf("channels.getSlice(9).size[1] == %zi\n", gpu_channels.getSlice(9).size[1]);
        printf("input_size.y == %zi, input_size.x == %zi\n", input_size.y, input_size.x);
        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }

    return;
}


void AbstractGpuIntegralChannelsComputer::compute_channels_at_canonical_scales(const cv::gpu::GpuMat &input_image,
                                                                               const std::string image_file_name)
{
    throw std::runtime_error("AbstractGpuIntegralChannelsComputer::compute_channels_at_canonical_scales(): "
                             "trying to compute canonical channels with a computer that doesn't override this method.");
    return;
}


void AbstractGpuIntegralChannelsComputer::set_current_scale(const float current_scale_)
{
    current_scale = current_scale_;
    return;
}


void AbstractGpuIntegralChannelsComputer::get_channel_matrix(const cv::Mat &integral_channel, cv::Mat &channel) const
{

    const size_t
            channel_size_x = integral_channel.cols - 1,
            channel_size_y = integral_channel.rows - 1;

    // reconstruct "non integral image" from integral image
    channel = cv::Mat(channel_size_y, channel_size_x, CV_32SC1);

    for(size_t y=0; y < channel_size_y; y+=1)
    {
        for(size_t x = 0; x < channel_size_x; x += 1)
        {
            const uint32_t
                    a = integral_channel.at<uint32_t>(y,x),
                    b = integral_channel.at<uint32_t>(y+0,x+1),
                    c = integral_channel.at<uint32_t>(y+1,x+1),
                    d = integral_channel.at<uint32_t>(y+1,x+0);
            channel.at<uint32_t>(y,x) = a+c-b-d;
        } // end of "for each column"
    } // end of "for each row"

    return;
}


AbstractGpuIntegralChannelsComputer::gpu_channels_t& AbstractGpuIntegralChannelsComputer::get_gpu_channels()
{
    return gpu_channels;
}


AbstractGpuIntegralChannelsComputer::gpu_channels_t& AbstractGpuIntegralChannelsComputer::get_shrunk_gpu_channels()
{
    return shrunk_gpu_channels;
}


/// helper function to access the integral channels on cpu
/// this is quite slow (large data transfer between GPU and CPU)
/// this method should be used for debugging only
const AbstractGpuIntegralChannelsComputer::integral_channels_t &AbstractGpuIntegralChannelsComputer::get_integral_channels()
{
    //const gpu_integral_channels_t& gpu_integral_channels = get_gpu_integral_channels();
    integral_channels_gpu_to_cpu(gpu_integral_channels, integral_channels);
    return integral_channels;
}



void AbstractGpuIntegralChannelsComputer::integral_channels_gpu_to_cpu(
        const AbstractGpuIntegralChannelsComputer::gpu_3d_integral_channels_t &gpu_integral_channels,
        AbstractGpuIntegralChannelsComputer::integral_channels_t &integral_channels)
{ // special code for the32d case


    const Cuda::Size<3> &data_size = gpu_integral_channels.getLayout().size;

    // resize the CPU memory storage --
    // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
    integral_channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference3D<AbstractGpuIntegralChannelsComputer::integral_channels_t::element>
            integral_channels_host_reference(data_size, integral_channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(integral_channels_host_reference, gpu_integral_channels);

    const bool print_sizes = false;
    if(print_sizes)
    {
        printf("gpu_integral_channels layout size == [%zi, %zi, %zi]\n",
               data_size[0], data_size[1], data_size[2]);

        const Cuda::Size<3> &data_stride = gpu_integral_channels.getLayout().stride;
        printf("gpu_integral_channels layout stride == [%zi, %zi, %zi]\n",
               data_stride[0], data_stride[1], data_stride[2]);

        printf("integral_channels shape == [%zi, %zi, %zi]\n",
               integral_channels.shape()[0],
                integral_channels.shape()[1],
                integral_channels.shape()[2]);

        printf("integral_channels strides == [%zi, %zi, %zi]\n",
               integral_channels.strides()[0],
                integral_channels.strides()[1],
                integral_channels.strides()[2]);

        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }

return;
}



void AbstractGpuIntegralChannelsComputer::integral_channels_gpu_to_cpu(
        const AbstractGpuIntegralChannelsComputer::gpu_2d_integral_channels_t &gpu_integral_channels,
        AbstractGpuIntegralChannelsComputer::integral_channels_t &integral_channels)
{ // special code for the 2d case

    const Cuda::Size<2> &data_size = gpu_integral_channels.getLayout().size;
    const size_t
            width_plus_one = data_size[0],
            height = gpu_integral_channels.height,
            num_channels = (data_size[1] - 1)/height;
    // the 2d integral channel height is = (channel_height*num_channels) + 1;
    // the +1 is due to the discrete integral.

    if((data_size[1] % height) != 1)
    {
        printf("data_size == %zu, %zu; height == %zu; data_size[1] %% height == %zu =?= 1\n",
               data_size[0], data_size[1], height, data_size[1] % height);
        throw std::runtime_error("integral_channels_gpu_to_cpu "
                                 "received 2d gpu integral channels with unexpected dimensions");
    }

    // resize the CPU memory storage --
    typedef boost::multi_array<integral_channels_t::element, 2>
            cpu_2d_integral_channels_t;
    cpu_2d_integral_channels_t long_integral_channels;
    // Cuda::DeviceMemoryPitched2D store the size indices in reverse order with respect to boost::multi_array
    long_integral_channels.resize(boost::extents[data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference2D<cpu_2d_integral_channels_t::element>
            integral_channels_host_reference(data_size, long_integral_channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(integral_channels_host_reference, gpu_integral_channels);


    // we need to create the full size integral channels, and copy the small one into it
    integral_channels.resize(boost::extents[num_channels][height+1][width_plus_one]);

    for(size_t channel_index = 0; channel_index < num_channels; channel_index += 1)
    {
        typedef integral_channels_t::reference integral_channel_ref_t;
        typedef integral_channels_t::const_reference integral_channel_const_ref_t;
        typedef boost::multi_array_types::index_range range_t;

        // ranges are non-inclusive for the max value, i.e. [start_index, bound_index)
        const range_t vertical_range(channel_index*height, (channel_index+1)*height + 1);
        integral_channel_const_ref_t::array_view<2>::type
                const_integral_channel = long_integral_channels[ boost::indices[vertical_range][range_t()] ];

        integral_channel_ref_t::array_view<2>::type small_view =
                integral_channels[ boost::indices[channel_index][range_t(0, height + 1)][range_t()] ];

        // we copy the core content -
        small_view = const_integral_channel;

    } // end of "for each channel"


    const bool print_sizes = false;
    if(print_sizes)
    {
        printf("gpu_integral_channels layout size == [%zu, %zu]\n",
               data_size[0], data_size[1]);

        const Cuda::Size<2> &data_stride = gpu_integral_channels.getLayout().stride;
        printf("gpu_integral_channels layout stride == [%zu, %zu]\n",
               data_stride[0], data_stride[1]);

        printf("long_integral_channels shape == [%zu, %zu]; strides == [%zu, %zu]\n",
                long_integral_channels.shape()[0],
                long_integral_channels.shape()[1],
                long_integral_channels.strides()[0],
                long_integral_channels.strides()[1]);

        printf("integral_channels shape == [%zu, %zu, %zu]\n",
               integral_channels.shape()[0],
                integral_channels.shape()[1],
                integral_channels.shape()[2]);

        printf("integral_channels strides == [%zu, %zu, %zu]\n",
               integral_channels.strides()[0],
                integral_channels.strides()[1],
                integral_channels.strides()[2]);

        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }

    return;
}




}
