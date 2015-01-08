#ifndef DOPPIA_GPUVERYFASTINTEGRALCHANNELSDETECTOR_HPP
#define DOPPIA_GPUVERYFASTINTEGRALCHANNELSDETECTOR_HPP

#include "GpuIntegralChannelsDetector.hpp"

#include "BaseVeryFastIntegralChannelsDetector.hpp"

#include "gpu/integral_channels_detector.cu.hpp" // for gpu_scale_data_t

namespace doppia {

class MultiScalesIntegralChannelsModel;


/// This is the GPU version of VeryFastIntegralChannelsDetector
/// @see VeryFastIntegralChannelsDetector
class GpuVeryFastIntegralChannelsDetector:
        public GpuIntegralChannelsDetector, public BaseVeryFastIntegralChannelsDetector
{
public:

    typedef doppia::objects_detection::gpu_scales_data_t gpu_scales_data_t;
    typedef doppia::objects_detection::gpu_scale_datum_t gpu_scale_datum_t;

    typedef doppia::objects_detection::gpu_stixel_t gpu_stixel_t;
    typedef doppia::objects_detection::gpu_stixels_t gpu_stixels_t;

    typedef doppia::objects_detection::gpu_half_window_widths_t gpu_half_window_widths_t;

    GpuVeryFastIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
            const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~GpuVeryFastIntegralChannelsDetector();

    /// This function is identical to GpuIntegralChannelsDetector::set_image except it loads
    /// additional per-scale information into the GPU
    void set_image(const boost::gil::rgb8c_view_t &input_view,
                   const std::string &image_file_path = std::string());

    void set_ground_plane_corridor(const ground_plane_corridor_t &corridor);
    void set_stixels(const stixels_t &stixels);

protected:

    gpu_scales_data_t gpu_scales_data;
    gpu_scale_datum_t::search_range_t max_search_range;
    int stixels_max_search_range_height;

    /// we store in GPU memory the stixels data
    gpu_stixels_t gpu_stixels;

    /// helper variable for the gpu_stixels computation
    float scale_logarithmic_step, log_min_detection_window_scale, object_to_detection_window_height_ratio;

    /// we store here detection_window_width/2, for each scale (used to find the correct stixel)
    gpu_half_window_widths_t gpu_half_window_widths;

    /// updates the GPU data about the scales
    void set_gpu_scales_data();

    /// updates the GPU data about the stixels
    void set_gpu_stixels();


    /// Helper function, obtain the scale index corresponding to a given detection height
    void cpu_stixel_to_gpu_stixel(const Stixel &cpu_stixel, gpu_stixel_t &gpu_stixel);

    /// Helper function, obtain the scale index corresponding to a given detection height
    size_t get_scale_index_from_height(const float height);


    void set_gpu_half_window_widths();

    /// we implement a specific GPU computation, where all the scales are handled on the GPU side
    void compute();

    /// this is the GpuVeryFastIntegralChannelsDetector implementation
    void compute_v2();

    /// helper class for testing
    friend class VeryFastDetectorScaleStatisticsApplication;

    /// for debugging
    friend class ObjectsDetectionGui;
};

} // end of namespace doppia

#endif // DOPPIA_GPUVERYFASTINTEGRALCHANNELSDETECTOR_HPP
