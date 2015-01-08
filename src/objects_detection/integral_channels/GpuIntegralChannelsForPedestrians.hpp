#ifndef DOPPIA_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP
#define DOPPIA_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP

#include "AbstractGpuIntegralChannelsComputer.hpp"
#include "IntegralChannelsForPedestrians.hpp"

//#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/scoped_ptr.hpp>

namespace doppia {

/// This is the GPU mirror of the IntegralChannelsForPedestrians
/// IntegralChannelsForPedestrians and GpuIntegralChannelsForPedestrians do not have a common basis class, because
/// altought they provide similar functions, their are aimed to be used in different pipelines.
/// In particular GpuIntegralChannelsForPedestrians does not aim at provide CPU access to the integral channels,
/// but GPU access to a following up GPU detection algorithm. CPU transfer of the integral channels is only
/// provided for debugging purposes
class GpuIntegralChannelsForPedestrians: public AbstractGpuIntegralChannelsComputer
{

public:
    typedef IntegralChannelsForPedestrians::input_image_t input_image_t;
    typedef IntegralChannelsForPedestrians::input_image_view_t input_image_view_t;

    typedef boost::gil::rgb8_pixel_t pixel_t;

    typedef AbstractGpuIntegralChannelsComputer::gpu_channels_t gpu_channels_t;
    typedef AbstractGpuIntegralChannelsComputer::gpu_integral_channels_t gpu_integral_channels_t; // 1007.7 Hz on Kochab

    //typedef IntegralChannelsForPedestrians::channels_t channels_t;
    //typedef boost::multi_array<gpu_channels_t::Type, 3> channels_t;
    typedef AbstractChannelsComputer::input_channels_t channels_t;

    typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;
    typedef IntegralChannelsForPedestrians::integral_channels_view_t integral_channels_view_t;
    typedef IntegralChannelsForPedestrians::integral_channels_const_view_t integral_channels_const_view_t;

    /// preprocessing filter
    typedef cv::gpu::FilterEngine_GPU filter_t;
    typedef cv::Ptr<filter_t> filter_shared_pointer_t;

public:

    //static boost::program_options::options_description get_options_description();

    /// main constructor to use
    GpuIntegralChannelsForPedestrians(const boost::program_options::variables_map &options,
                                      const bool use_presmoothing_ = true);

    /// helper constructor used in HogTwoSixEighteenLuvChannels
    GpuIntegralChannelsForPedestrians(const size_t num_hog_angle_bins_ = 6,
                                      const bool use_presmoothing_ = true);

    ~GpuIntegralChannelsForPedestrians();

    /// how much we shrink the channel images ?
    static int get_shrinking_factor();

    size_t get_num_channels() const;

    /// helper function, just for debugging
    void save_channels_to_file();

    void compute_channels_at_canonical_scales(const cv::gpu::GpuMat &input_image,
                                              const std::string image_file_name = "");

public:

    void compute();
    void compute_v0();
    void compute_v1();

    /// helper function to access the compute channels on cpu
    /// this is quite slow (large data transfer between GPU and CPU)
    /// this method should be used for debugging only
    const channels_t &get_channels();

    const AbstractIntegralChannelsComputer::input_channels_t &get_input_channels_uint8();
    const AbstractIntegralChannelsComputer::channels_t &get_input_channels_uint16();

protected:

    const int num_hog_angle_bins;
    const bool use_presmoothing;

    /// cpu copy of the computed channels
    channels_t channels;

protected:

    void compute_smoothed_image_v0();
    void compute_hog_channels_v0();
    void compute_luv_channels_v0();
    void compute_hog_and_luv_channels_v1();

    void resize_and_integrate_gpu_channels_v0();
    void resize_and_integrate_gpu_channels_v1();
    void resize_and_integrate_gpu_channels_v2();

    void shrink_gpu_channel_v0(cv::gpu::GpuMat &gpu_feature_channel, cv::gpu::GpuMat &shrunk_gpu_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);
    void shrink_gpu_channel_v1(cv::gpu::GpuMat &gpu_feature_channel, cv::gpu::GpuMat &shrunk_gpu_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);
    void shrink_gpu_channel_v2(cv::gpu::GpuMat &gpu_feature_channel, cv::gpu::GpuMat &shrunk_gpu_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);
    void shrink_gpu_channel_v3(cv::gpu::GpuMat &gpu_feature_channel, cv::gpu::GpuMat &shrunk_gpu_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);

    cv::gpu::GpuMat
    smoothed_input_gpu_mat,
    hog_input_gpu_mat,
    luv_gpu_mat;

    /// helper variable to avoid doing memory re-allocations for each channel shrinking
    cv::gpu::GpuMat shrink_gpu_channel_buffer_a, shrink_gpu_channel_buffer_b;

    /// helper variable to avoid doing memory re-allocations for each channel index integral computation
    cv::gpu::GpuMat gpu_integral_channel_buffer_a, gpu_integral_channel_buffer_b;

    /// the size of one_half and shrunk_channel is the same of all channels,
    /// thus we can reuse the same matrices
    cv::gpu::GpuMat one_half, shrunk_gpu_channel;

    filter_shared_pointer_t pre_smoothing_filter_p;

};


} // end of namespace doppia


#endif // DOPPIA_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP
