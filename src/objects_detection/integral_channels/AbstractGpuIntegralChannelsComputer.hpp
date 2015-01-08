#ifndef ABSTRACTGPUINTEGRALCHANNELSCOMPUTER_HPP
#define ABSTRACTGPUINTEGRALCHANNELSCOMPUTER_HPP

#include "AbstractIntegralChannelsComputer.hpp"

#include "../gpu/DeviceMemoryPitched2DWithHeight.hpp"
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememoryreference.hpp>
#include <cudatemplates/copy.hpp>


#include <opencv2/core/version.hpp>
#include <opencv2/gpu/gpu.hpp>
//#if CV_MINOR_VERSION <= 3
//#include <opencv2/gpu/gpu.hpp> // opencv 2.3
//#else
//#include <opencv2/core/gpumat.hpp> // opencv 2.4
//#endif

#include <boost/cstdint.hpp>
#include <boost/program_options/variables_map.hpp>

#include "helpers/get_option_value.hpp"

namespace doppia {

class AbstractGpuIntegralChannelsComputer : public AbstractIntegralChannelsComputer
{
public:

    // the computed channels are uint8, after shrinking they are uint12
    // on CPU we use uint16, however opencv's GPU code only supports int16,
    // which is compatible with uint12
    // using gpu_channels_t::element == uint8 means loosing 4 bits,
    // this however generates only a small loss in the detections performance
    //typedef Cuda::DeviceMemoryPitched3D<boost::uint16_t> gpu_channels_t;
    //typedef Cuda::DeviceMemoryPitched3D<boost::int16_t> gpu_channels_t;
    typedef Cuda::DeviceMemoryPitched3D<boost::uint8_t> gpu_channels_t;
    typedef Cuda::DeviceMemoryPitched3D<boost::uint32_t> gpu_3d_integral_channels_t;
    //typedef Cuda::DeviceMemoryPitched2D<boost::uint32_t> gpu_2d_integral_channels_t;
    typedef doppia::DeviceMemoryPitched2DWithHeight<boost::uint32_t> gpu_2d_integral_channels_t;
    //typedef gpu_3d_integral_channels_t gpu_integral_channels_t; // 818.6 Hz on Kochab //TODO: multiple definitions
    typedef gpu_2d_integral_channels_t gpu_integral_channels_t; // 1007.7 Hz on Kochab

public:
    AbstractGpuIntegralChannelsComputer(const int resizing_factor_);
    virtual ~AbstractGpuIntegralChannelsComputer();

    /// keep a reference to an existing GPU image
    /// we assume that the given gpu image will not change during the compute calls
    virtual void set_gpu_image(const cv::gpu::GpuMat &input_image);

    /// returns a reference to the GPU 3d structure holding the integral channels
    /// returns a non-const reference because cuda structures do not play nice with const-ness
    virtual gpu_integral_channels_t& get_gpu_integral_channels();
    virtual gpu_channels_t& get_gpu_channels();
    virtual gpu_channels_t& get_shrunk_gpu_channels();

    virtual void allocate_gpu_channels(const size_t image_width, const size_t image_height);
    template<typename ChannelsType>
    cv::gpu::GpuMat get_slice(ChannelsType &channels, const size_t slice_index);
    gpu_channels_t gpu_channels;


    void integral_channels_gpu_to_cpu(const gpu_2d_integral_channels_t &gpu_integral_channels,
                                      integral_channels_t &integral_channels);

    void integral_channels_gpu_to_cpu(const gpu_3d_integral_channels_t &gpu_integral_channels,
                                      integral_channels_t &integral_channels);


    /// helper function to access the integral channels on cpu
    /// this is quite slow (large data transfer between GPU and CPU)
    /// this method should be used for debugging only
    virtual const integral_channels_t &get_integral_channels();

    void set_canonical_scales_for_stuff(const boost::program_options::variables_map &options);
    void set_canonical_scales_for_things(const boost::program_options::variables_map &options);

    virtual void set_current_scale(const float current_scale_);

    virtual void compute_channels_at_canonical_scales(const cv::gpu::GpuMat &input_image,
                                                      const std::string image_file_name = "");

protected:
    virtual void set_input_image(const AbstractChannelsComputer::input_image_view_t &input_view);
    void get_channel_matrix(const cv::Mat &integral_channel, cv::Mat &channel) const;

protected:

    /// how much we shrink the channel images ?
    const int resizing_factor;

    input_image_view_t::point_t
    input_size,
    shrunk_gpu_channel_size;

    /// integral channels are computed over the shrunk channels
    cv::gpu::GpuMat input_gpu_mat,
    input_rgb8_gpu_mat;
    cv::Mat input_rgb8_gpu_mem_mat;  // allows for faster data upload
    cv::gpu::CudaMem input_rgb8_gpu_mem; // allows for faster data upload

    gpu_integral_channels_t gpu_integral_channels;
    gpu_channels_t shrunk_gpu_channels;

    /// cpu copy of the integral channels
    integral_channels_t integral_channels;

    std::vector<float> canonical_scales_for_stuff;
    std::vector<float> canonical_scales_for_things;
    float current_scale;


};

} // end of namespace doppia


namespace cv
{ // to define get_slice(...) we need to fix OpenCv

//template<> class DataDepth<boost::uint32_t> { public: enum { value = CV_32S, fmt=(int)'i' }; };

template<> class DataType<boost::uint32_t>
{
public:
    typedef boost::uint32_t value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};


} // end of namespace cv


namespace doppia {

template<typename ChannelsType> inline
cv::gpu::GpuMat AbstractGpuIntegralChannelsComputer::get_slice(ChannelsType &channels, const size_t slice_index)
{
    // GpuMat(rows, cols, ...)
    return cv::gpu::GpuMat(channels.getSlice(slice_index).size[1], channels.getSlice(slice_index).size[0],
                           cv::DataType<typename ChannelsType::Type>::type,
                           channels.getSlice(slice_index).getBuffer(),
                           channels.getSlice(slice_index).getPitch() );
}


} // end of namespace doppia

#endif // ABSTRACTGPUINTEGRALCHANNELSCOMPUTER_HPP
