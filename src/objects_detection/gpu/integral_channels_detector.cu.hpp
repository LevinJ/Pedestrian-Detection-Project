#ifndef INTEGRAL_CHANNELS_DETECTOR_CU_HPP
#define INTEGRAL_CHANNELS_DETECTOR_CU_HPP

#include "objects_detection/ScaleData.hpp"
#include "objects_detection/SoftCascadeOverIntegralChannelsModel.hpp"
#include "objects_detection/cascade_stages/SoftCascadeOverIntegralChannelsFastFractionalStage.hpp"

#include "DeviceMemoryPitched2DWithHeight.hpp"

#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememorylinear.hpp>

#include <opencv2/core/version.hpp>
#if CV_MINOR_VERSION <= 3
#include <opencv2/gpu/devmem2d.hpp> // opencv 2.3
#else
#include <opencv2/core/devmem2d.hpp> // opencv 2.4
#endif

#include <boost/cstdint.hpp>

namespace doppia {
namespace objects_detection {

#if CV_MINOR_VERSION < 3
typedef cv::gpu::DevMem2Df dev_mem_2d_float_t;
typedef cv::gpu::PtrElemStepf dev_mem_ptr_step_float_t;
#else
// we avoid the deprecation warnings
typedef cv::gpu::PtrStepSz<float> dev_mem_2d_float_t;
typedef cv::gpu::PtrStep<float> dev_mem_ptr_step_float_t;
#endif


typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t cascade_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::stump_stage_t stump_cascade_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stage_t fractional_cascade_stage_t;

typedef SoftCascadeOverIntegralChannelsModel::two_stumps_stage_t two_stumps_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::three_stumps_stage_t three_stumps_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::four_stumps_stage_t four_stumps_stage_t;

typedef Cuda::DeviceMemoryPitched3D<boost::uint32_t> gpu_3d_integral_channels_t;
//typedef Cuda::DeviceMemoryPitched2D<boost::uint32_t> gpu_2d_integral_channels_t;
typedef doppia::DeviceMemoryPitched2DWithHeight<boost::uint32_t> gpu_2d_integral_channels_t;

// 2d integral channels are faster than the 3d ones (~1000 Hz versus ~800 Hz on Kochab)
//typedef gpu_3d_integral_channels_t gpu_integral_channels_t;
typedef gpu_2d_integral_channels_t gpu_integral_channels_t;

//typedef Cuda::DeviceMemoryPitched2D<cascade_stage_t> gpu_detection_cascade_per_scale_t;
typedef Cuda::DeviceMemoryLinear2D<cascade_stage_t> gpu_detection_cascade_per_scale_t;

//typedef Cuda::DeviceMemoryPitched2D<stump_cascade_stage_t> gpu_detection_stump_cascade_per_scale_t;
typedef Cuda::DeviceMemoryLinear2D<stump_cascade_stage_t> gpu_detection_stump_cascade_per_scale_t;

//typedef Cuda::DeviceMemoryPitched2D<fractional_cascade_stage_t> gpu_detection_cascade_per_scale_t;
typedef Cuda::DeviceMemoryLinear2D<fractional_cascade_stage_t> gpu_fractional_detection_cascade_per_scale_t;

typedef Cuda::DeviceMemoryLinear2D<two_stumps_stage_t> gpu_detection_two_stumps_cascade_per_scale_t;
typedef Cuda::DeviceMemoryLinear2D<three_stumps_stage_t> gpu_detection_three_stumps_cascade_per_scale_t;
typedef Cuda::DeviceMemoryLinear2D<four_stumps_stage_t> gpu_detection_four_stumps_cascade_per_scale_t;


/// Small structure used to store the scale specific information on GPU
struct __align__(16) gpu_scale_datum_t {

    typedef boost::int16_t coordinate_t;
    typedef doppia::geometry::point_xy<coordinate_t> point_t;
    typedef doppia::geometry::box<point_t> search_range_t;

    typedef geometry::point_xy<boost::uint8_t> stride_t;

    search_range_t search_range; ///< scaled search area
    stride_t stride; ///< scaled x/y stride
};

typedef Cuda::DeviceMemoryLinear1D<gpu_scale_datum_t> gpu_scales_data_t;


/// Small structure that stores the information of one specific stixels
/// GPU stixels are always of width one
struct __align__(16) gpu_stixel_t {

    typedef boost::int16_t coordinate_t;
    coordinate_t min_y, max_y;

    /// min_scale_index is inclusive, max_scale_index is non-inclusive
    size_t min_scale_index, reference_scale_index, max_scale_index;
};

typedef Cuda::DeviceMemoryLinear1D<gpu_stixel_t> gpu_stixels_t;

/// in this datatype we will store detection_window_width/2, which is required for stixels usage
typedef Cuda::DeviceMemoryLinear1D<boost::int16_t> gpu_half_window_widths_t;

/// Small structure used to store on the GPU the detections
struct gpu_detection_t {
    /// scale_index is synonym of search_range_index (each scale has a different search range)
    boost::uint16_t scale_index, x, y;
    float score;
};

typedef Cuda::DeviceMemoryLinear1D<gpu_detection_t> gpu_detections_t;


/// gpu types are passed as non const references, because cudatemplates code does not play nice with const-ness
/// this method fills in the detections score image, that is then thresholded to retreive the actual detections
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const bool use_the_model_cascade,
                                objects_detection::dev_mem_2d_float_t &detection_scores);

/// Stump version of integral_channels_detector
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
                                const bool use_the_model_cascade,
                                dev_mem_2d_float_t& detection_scores);

/// StumpSet version of integral_channels_detector
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_detection_three_stumps_cascade_per_scale_t &detection_cascade_per_scale,
                                const bool use_the_model_cascade,
                                dev_mem_2d_float_t& detection_scores);


// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// this method directly adds elements into the gpu_detections vector
/// @warning will skip all detections once the vector is full
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections);


/// this method directly adds elements into the gpu_detections vector
/// stumps detector at a specific scale
/// @warning will skip all detections once the vector is full
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections);


/// this method directly adds elements into the gpu_detections vector
/// fractional version
/// @warning will skip all detections once the vector is full
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections);


/// this method directly adds elements into the gpu_detections vector
/// stumps detector at a specific scale
/// @warning will skip all detections once the vector is full
void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_three_stumps_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections);

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// compute detections at all scales in one call
/// this method directly adds elements into the gpu_detections vector
/// @warning will skip all detections once the vector is full
void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections);


/// compute detections at all scales in one call
/// three stumps version
/// this method directly adds elements into the gpu_detections vector
/// @warning will skip all detections once the vector is full
void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_detection_three_stumps_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections);


/// compute detections at all scales in one call (fractional version)
/// this method directly adds elements into the gpu_detections vector
/// @warning will skip all detections once the vector is full
void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections);

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// this method directly adds elements into the gpu_detections vector
/// this version takes into account the ground plane and stixel constraints
/// @warning will skip all detections once the vector is full
void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        const int num_scales_to_evaluate,
        gpu_scales_data_t &scales_data,
        gpu_stixels_t &stixels,
        gpu_half_window_widths_t &gpu_half_window_widths,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections);


} // end of namespace objects_detection
} // end of namespace doppia

#endif // INTEGRAL_CHANNELS_DETECTOR_CU_HPP
