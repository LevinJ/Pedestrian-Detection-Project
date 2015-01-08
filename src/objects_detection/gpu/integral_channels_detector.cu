#include "integral_channels_detector.cu.hpp"

#include "helpers/gpu/cuda_safe_call.hpp"

#include "cudatemplates/array.hpp"
#include "cudatemplates/symbol.hpp"
#include "cudatemplates/copy.hpp"

#include <cudatemplates/hostmemoryheap.hpp>

#include <boost/cstdint.hpp>

#include <stdexcept>



namespace {

/// small helper function that computes "static_cast<int>(ceil(static_cast<float>(total)/grain))", but faster
static inline int div_up(const int total, const int grain)
{
    return (total + grain - 1) / grain;
}

} // end of anonymous namespace


namespace doppia {
namespace objects_detection {

using namespace cv::gpu;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 1>::Texture gpu_integral_channels_1d_texture_t;
gpu_integral_channels_1d_texture_t integral_channels_1d_texture;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 2>::Texture gpu_integral_channels_2d_texture_t;
gpu_integral_channels_2d_texture_t integral_channels_2d_texture;

// global variable to switch from using 1d textures to using 2d textures
// On visics-gt680r 1d texture runs at ~4.8 Hz; 2d texture runs at ~4.4 Hz
//const bool use_2d_texture = false;
const bool use_2d_texture = true;

/// this method will do zero memory checks, the user is responsible of avoiding out of memory accesses
inline
__device__
float get_feature_value_global_memory(const IntegralChannelsFeature &feature,
                                      const int x, const int y,
                                      const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride,
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    const gpu_integral_channels_t::Type
            a = integral_channels.data[top_left_index],
            b = integral_channels.data[top_right_index],
            c = integral_channels.data[bottom_right_index],
            d = integral_channels.data[bottom_left_index];

    const float feature_value = a +c -b -d;

    return feature_value;
}


inline
__device__
float get_feature_value_tex1d(const IntegralChannelsFeature &feature,
                              const int x, const int y,
                              const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride;
    const size_t
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_1d_texture_t &tex = integral_channels_1d_texture;
#define tex integral_channels_1d_texture
    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    // tex1Dfetch should be used to access linear memory (not text1D)
    const gpu_integral_channels_t::Type 
    //const float
            a = tex1Dfetch(tex, top_left_index),
            b = tex1Dfetch(tex, top_right_index),
            c = tex1Dfetch(tex, bottom_right_index),
            d = tex1Dfetch(tex, bottom_left_index);
#undef tex


    const float feature_value = a +c -b -d;

    return feature_value;
}



/// This is a dumb compability code, IntegralChannelsFractionalFeature is meant to be used with
/// get_feature_value_tex2d
inline
__device__
float get_feature_value_tex1d(const IntegralChannelsFractionalFeature &feature,
                              const int x, const int y,
                              const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFractionalFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride;
    const size_t
            top_left_index =
            (x + size_t(box.min_corner().x())) + ((y + size_t(box.min_corner().y()))*row_stride) + channel_offset,
            top_right_index =
            (x + size_t(box.max_corner().x())) + ((y + size_t(box.min_corner().y()))*row_stride) + channel_offset,
            bottom_left_index =
            (x + size_t(box.min_corner().x())) + ((y + size_t(box.max_corner().y()))*row_stride) + channel_offset,
            bottom_right_index =
            (x + size_t(box.max_corner().x())) + ((y + size_t(box.max_corner().y()))*row_stride) + channel_offset;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_1d_texture_t &tex = integral_channels_1d_texture;
#define tex integral_channels_1d_texture
    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    // tex1Dfetch should be used to access linear memory (not text1D)
    const gpu_integral_channels_t::Type 
    //const float
            a = tex1Dfetch(tex, top_left_index),
            b = tex1Dfetch(tex, top_right_index),
            c = tex1Dfetch(tex, bottom_right_index),
            d = tex1Dfetch(tex, bottom_left_index);
#undef tex
    const float feature_value = a +c -b -d;

    return feature_value;
}



template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_3d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    const size_t integral_channels_height = integral_channels.size[1];
    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    const gpu_integral_channels_t::Type 
    //const float
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_2d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    const size_t integral_channels_height = integral_channels.height; // magic trick !

    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    //const float
    const gpu_integral_channels_t::Type 
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType, bool should_use_2d_texture>
inline
__device__
float get_feature_value(const FeatureType &feature,
                        const int x, const int y,
                        const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    // default implementation (hopefully optimized by the compiler)
    if (should_use_2d_texture)
    {
        return get_feature_value_tex2d(feature, x, y, integral_channels);
    }
    else
    {
        //return get_feature_value_global_memory(feature, x, y, integral_channels);
        return get_feature_value_tex1d(feature, x, y, integral_channels);
    }

    //return 0;
}


template <>
inline
__device__
float get_feature_value<IntegralChannelsFractionalFeature, true>(
        const IntegralChannelsFractionalFeature &feature,
        const int x, const int y,
        const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    // should_use_2d_texture == true
    return get_feature_value_tex2d(feature, x, y, integral_channels);
}


inline
__device__
bool evaluate_decision_stump(const DecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    if(feature_value >= stump.feature_threshold)
    {
        return stump.larger_than_threshold;
    }
    else
    {
        return not stump.larger_than_threshold;
    }
}


inline
__device__
bool evaluate_decision_stump(const SimpleDecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold);
}


inline
__device__
float evaluate_decision_stump(const DecisionStumpWithWeights &stump,
                              const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold)? stump.weight_true_leaf : stump.weight_false_leaf;
}

inline
__device__
bool evaluate_decision_stump(const SimpleFractionalDecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold);
}


inline
__device__
float evaluate_decision_stump(const FractionalDecisionStumpWithWeights &stump,
                              const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold)? stump.weight_true_leaf : stump.weight_false_leaf;
}


template<typename CascadeStageType>
inline
__device__
void update_detection_score(
        const int x, const int y,
        const CascadeStageType &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_score)
{
    const typename CascadeStageType::weak_classifier_t &weak_classifier = stage.weak_classifier;
    typedef typename CascadeStageType::weak_classifier_t::feature_t feature_t;

    // level 1 nodes evaluation returns a boolean value,
    // level 2 nodes evaluation returns directly the float value to add to the score

    const float level1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(
                weak_classifier.level1_node.feature, x, y, integral_channels);

    // On preliminary versions,
    // evaluating the level2 features inside the if/else
    // runs slightly faster than evaluating all of them beforehand; 4.35 Hz vs 4.55 Hz)
    // on the fastest version (50 Hz or more) evaluating all three features is best

    const bool use_if_else = false;
    if(not use_if_else)
    { // this version is faster

        const float level2_true_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_true_node.feature, x, y, integral_channels);

        const float level2_false_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_false_node.feature, x, y, integral_channels);

        current_score +=
                (evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value)) ?
                    evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value) :
                    evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
    }
    else
    {
        if(evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value))
        {
            const float level2_true_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_true_node.feature, x, y, integral_channels);

            current_score += evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value);
        }
        else
        {
            const float level2_false_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_false_node.feature, x, y, integral_channels);

            current_score +=evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
        }

    }

    return;
}



template<>
inline
__device__
void update_detection_score<SoftCascadeOverIntegralChannelsStumpStage>(
        const int x, const int y,
        const SoftCascadeOverIntegralChannelsStumpStage &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_detection_score)
{
    const typename SoftCascadeOverIntegralChannelsStumpStage::weak_classifier_t &weak_classifier = stage.weak_classifier;
    typedef SoftCascadeOverIntegralChannelsStumpStage::weak_classifier_t::feature_t feature_t;

    const float feature_value =
            get_feature_value<feature_t, use_2d_texture>(
                weak_classifier.feature, x, y, integral_channels);

    current_detection_score += evaluate_decision_stump(weak_classifier, feature_value);

    return;
}



template<>
inline
__device__
void update_detection_score<two_stumps_stage_t>(
        const int x, const int y,
        const two_stumps_stage_t &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_detection_score)
{
    typedef two_stumps_stage_t::stump_t::feature_t feature_t;

    const float
            stump_0_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[0].feature, x, y, integral_channels),
            stump_1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[1].feature, x, y, integral_channels);

    const int weight_index =
            evaluate_decision_stump(stage.stumps[0], stump_0_feature_value) * 1
            + evaluate_decision_stump(stage.stumps[1], stump_1_feature_value) * 2;

    current_detection_score += stage.weights[weight_index];

    return;
}


template<>
inline
__device__
void update_detection_score<three_stumps_stage_t>(
        const int x, const int y,
        const three_stumps_stage_t &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_detection_score)
{
    typedef three_stumps_stage_t::stump_t::feature_t feature_t;

    const float
            stump_0_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[0].feature, x, y, integral_channels),
            stump_1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[1].feature, x, y, integral_channels),
            stump_2_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[2].feature, x, y, integral_channels);

    const int weight_index =
            evaluate_decision_stump(stage.stumps[0], stump_0_feature_value) * 1
            + evaluate_decision_stump(stage.stumps[1], stump_1_feature_value) * 2
            + evaluate_decision_stump(stage.stumps[2], stump_2_feature_value) * 4;

    current_detection_score += stage.weights[weight_index];

    return;
}


template<>
inline
__device__
void update_detection_score<four_stumps_stage_t>(
        const int x, const int y,
        const four_stumps_stage_t &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_detection_score)
{
    typedef four_stumps_stage_t::stump_t::feature_t feature_t;

    const float
            stump_0_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[0].feature, x, y, integral_channels),
            stump_1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[1].feature, x, y, integral_channels),
            stump_2_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[2].feature, x, y, integral_channels),
            stump_3_feature_value =
            get_feature_value<feature_t, use_2d_texture>(stage.stumps[3].feature, x, y, integral_channels);


    const int weight_index =
            evaluate_decision_stump(stage.stumps[0], stump_0_feature_value) * 1
            + evaluate_decision_stump(stage.stumps[1], stump_1_feature_value) * 2
            + evaluate_decision_stump(stage.stumps[2], stump_2_feature_value) * 4
            + evaluate_decision_stump(stage.stumps[3], stump_3_feature_value) * 8;

    current_detection_score += stage.weights[weight_index];

    return;
}

#if defined(BOOTSTRAPPING_LIB)
const bool use_hardcoded_cascade = true;
//const int hardcoded_cascade_start_stage = 100; // this is probably good enough
const int hardcoded_cascade_start_stage = 100; // to be on the safe side
const float hardcoded_cascade_threshold = -5;
//const int hardcoded_cascade_start_stage = 500; // this is conservative
#else
// FIXME these should be templated options, selected at runtime
//const bool use_hardcoded_cascade = true;
const bool use_hardcoded_cascade = true;
//const int hardcoded_cascade_start_stage = 100;
//const int hardcoded_cascade_start_stage = 250; // same as during bootstrapping
const int hardcoded_cascade_start_stage = 100;
//const int hardcoded_cascade_start_stage = 1000;

// will break if (detection_score < hardcoded_cascade_threshold)
//const float hardcoded_cascade_threshold = 0;
//const float hardcoded_cascade_threshold = -1;
const float hardcoded_cascade_threshold = -0.03;
#endif

/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector

template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const int search_range_width, const int search_range_height,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        dev_mem_ptr_step_float_t detection_scores)
{
    const int
            x = blockIdx.x * blockDim.x + threadIdx.x,
            //y = blockIdx.y;
            y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x >= search_range_width) or ( y >= search_range_height))
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0; //detection_scores_row[x];

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index = 0; stage_index < cascade_length; stage_index += 1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        // (printfs in cuda code requires at least -arch=sm_20) (we use -arch=sm_21)
        // should not print too much from the GPU, to avoid opencv's timeouts
        /*if(print_cascade_scores
                and (y == 52) and (x == 4)
                //and (y == 203) and (x == 101)
                and (stage_index < 10))
        {

            const DetectionCascadeStageType::weak_classifier_t &weak_classifier = stage.weak_classifier;
            const float level1_feature_value =
                    get_feature_value_tex1d(weak_classifier.level1_node.feature,
                                            x, y, integral_channels);

            const float level2_true_feature_value =
                    get_feature_value_tex1d(weak_classifier.level2_true_node.feature,
                                            x, y, integral_channels);

            const float level2_false_feature_value =
                    get_feature_value_tex1d(weak_classifier.level2_false_node.feature,
                                            x, y, integral_channels);

            printf("Cascade score at (%i, %i),\tstage %i == %2.3f,\tthreshold == %.3f,\t"
                   "level1_feature == %.3f,\tlevel2_true_feature == %4.3f,\tlevel2_false_feature == %4.3f\n",
                   x, y, stage_index,
                   detection_score, stage.cascade_threshold,
                   level1_feature_value, level2_true_feature_value, level2_false_feature_value);

            printf("level1 threshold %.3f, level2_true threshold %.3f, level2_false threshold %.3f\n",
                   weak_classifier.level1_node.feature_threshold,
                   weak_classifier.level2_true_node.feature_threshold,
                   weak_classifier.level2_false_node.feature_threshold);
        }*/



        if((not use_the_model_cascade)
                and use_hardcoded_cascade
                and (stage_index > hardcoded_cascade_start_stage)
                and (detection_score < hardcoded_cascade_threshold))
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            break;
        }

        if(use_the_model_cascade and detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    float* detection_scores_row = detection_scores.ptr(y);
    detection_scores_row[x] = detection_score; // save the updated score
    //detection_scores_row[x] = cascade_length; // just for debugging
    return;
}


/// type int because atomicAdd does not support size_t
__device__ int num_gpu_detections[1];
Cuda::Symbol<int, 1> num_gpu_detections_symbol(Cuda::Size<1>(1), num_gpu_detections);
int num_detections_int;
Cuda::HostMemoryReference1D<int> num_detections_host_ref(1, &num_detections_int);


void move_num_detections_from_cpu_to_gpu(size_t &num_detections)
{ // move num_detections from CPU to GPU --
    num_detections_int = static_cast<int>(num_detections);
    Cuda::copy(num_gpu_detections_symbol, num_detections_host_ref);
    return;
}

void move_num_detections_from_gpu_to_cpu(size_t &num_detections)
{ // move (updated) num_detections from GPU to CPU
    Cuda::copy(num_detections_host_ref, num_gpu_detections_symbol);
    if(num_detections_int < static_cast<int>(num_detections))
    {
        throw std::runtime_error("Something went terribly wrong when updating the number of gpu detections");
    }
    num_detections = static_cast<size_t>(num_detections_int);
    return;
}



template<typename ScaleType>
inline
__device__
void add_detection(
        gpu_detections_t::KernelData &gpu_detections,
        const int x, const int y, const ScaleType scale_index,
        const float detection_score)
{
    gpu_detection_t detection;
    detection.scale_index = static_cast<boost::int16_t>(scale_index);
    detection.x = static_cast<boost::int16_t>(x);
    detection.y = static_cast<boost::int16_t>(y);
    detection.score = detection_score;

    const size_t detection_index = atomicAdd(num_gpu_detections, 1);
    if(detection_index < gpu_detections.size[0])
    {
        // copy the detection into the global memory
        gpu_detections.data[detection_index] = detection;
    }
    else
    {
        // we drop out of range detections
    }

    return;
}


/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const gpu_scale_datum_t scale_datum,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{

    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    //const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            //delta_y = blockIdx.y;
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;


    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0;

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index = 0; stage_index < cascade_length; stage_index += 1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        if((not use_the_model_cascade)
                and use_hardcoded_cascade
                and (stage_index > hardcoded_cascade_start_stage)
                and (detection_score < hardcoded_cascade_threshold))
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            break;
        }

        if(use_the_model_cascade and detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    // >= to be consistent with Markus's code
    if(detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);
    }

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 1d binding
void bind_integral_channels_to_1d_texture(gpu_integral_channels_t &integral_channels)
{

    //integral_channels_texture.filterMode = cudaFilterModeLinear; // linear interpolation of the values
    integral_channels_1d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 1d data,
    // and then have ad-hoc texture access in the kernel
    // (if interpolation is needed, will need to do a different 2d data hack
    const cudaChannelFormatDesc texture_channel_description = \
            cudaCreateChannelDesc<gpu_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
            or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // FIXME add this 3D to 1D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture(0, integral_channels_1d_texture, integral_channels.getBuffer(),
                               texture_channel_description, integral_channels.getBytes()));

    cuda_safe_call( cudaGetLastError() );

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_3d_integral_channels_t &integral_channels)
{ // 3d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_2d_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 2d data,
    // and then have ad-hoc texture access in the kernel
    const cudaChannelFormatDesc texture_channel_description = cudaCreateChannelDesc<gpu_3d_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
            or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc seems to have failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // Layout.size is width, height, num_channels
    const size_t
            integral_channels_width = integral_channels.getLayout().size[0],
            integral_channels_height = integral_channels.getLayout().size[1],
            num_integral_channels = integral_channels.getLayout().size[2],
            //channel_stride = integral_channels.getLayout().stride[1],
            row_stride = integral_channels.getLayout().stride[0],
            pitch_in_bytes = row_stride * sizeof(gpu_3d_integral_channels_t::Type);

    if(false)
    {
        printf("image width/height == %zi, %zi; row_stride == %zi\n",
               integral_channels.getLayout().size[0], integral_channels.getLayout().size[1],
               integral_channels.getLayout().stride[0]);

        printf("integral_channels size / channel_stride == %.3f\n",
               integral_channels.getLayout().stride[2] / float(integral_channels.getLayout().stride[1]) );
    }

    // FIXME add this 3D to 2D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture2D(0, integral_channels_2d_texture, integral_channels.getBuffer(),
                                 texture_channel_description,
                                 integral_channels_width, integral_channels_height*num_integral_channels,
                                 pitch_in_bytes));

    cuda_safe_call( cudaGetLastError() );
    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_2d_integral_channels_t &integral_channels)
{ // 2d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values

    // integral_channels_height == (channel_height*num_channels) + 1
    // 2d to 2d binding
    integral_channels.bindTexture(integral_channels_2d_texture);

    cuda_safe_call( cudaGetLastError() );
    return;
}



void bind_integral_channels_texture(gpu_integral_channels_t &integral_channels)
{
    if(use_2d_texture)
    {
        bind_integral_channels_to_2d_texture(integral_channels);
    }
    else
    {
        bind_integral_channels_to_1d_texture(integral_channels);
    }

    return;
}


void unbind_integral_channels_texture()
{

    if(use_2d_texture)
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_2d_texture) );
    }
    else
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_1d_texture) );
    }
    cuda_safe_call( cudaGetLastError() );

    return;
}


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_impl(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        dev_mem_2d_float_t& detection_scores)
{

    if((search_range.min_x != 0) or (search_range.min_y != 0))
    {
        throw std::runtime_error("integral_channels_detector(...) expect search_range.min_x/y values to be zero"
                                 "(use_stixels/use_ground_plane should be false)");
    }

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;
    const int width = search_range.max_x, height = search_range.max_y;

    if((width <= 0) or (height <= 0))
    { // nothing to be done
        throw std::invalid_argument("integral_channels_detector_impl (with detection_scores), "
                                    "should be calied with valid search ranges");
    }


    if((width > detection_scores.cols) or (height > detection_scores.rows))
    {
        printf("search_range.max_x/y == (%i, %i)\n", search_range.max_x, search_range.max_y);
        printf("detection_scores.cols/rows == (%i, %i)\n", detection_scores.cols, detection_scores.rows);
        throw std::runtime_error("integral_channels_detector(...) expects "
                                 "detections_scores to be larger than the search_range.max_x/y values");
    }


    if(false)
    {
        printf("detection_cascade_per_scale.size[0] == %zi\n", detection_cascade_per_scale.size[0]);
        printf("detection_cascade_per_scale.stride[0] == %zi\n", detection_cascade_per_scale.stride[0]);
        printf("detection_cascade_per_scale.size[1] == %zi\n", detection_cascade_per_scale.size[1]);
    }


    //const int nthreads = 256; dim3 block_dimensions(nthreads, 1);
    //dim3 block_dimensions(16, 16);
    //dim3 block_dimensions(32, 32);
    //const int nthreads = 320; // we optimize for images of width 640 pixel
    dim3 block_dimensions(32, 10);
    dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));

    // bind the integral_channels_texture
    bind_integral_channels_texture(integral_channels);

    if(use_the_model_cascade)
    {
        integral_channels_detector_kernel
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       detection_scores);
    }
    else
    {
        integral_channels_detector_kernel
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       detection_scores);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    unbind_integral_channels_texture();
    return;
}


void integral_channels_detector(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        dev_mem_2d_float_t& detection_scores)
{
    integral_channels_detector_impl(integral_channels,
                                    search_range_index,
                                    search_range,
                                    detection_cascade_per_scale,
                                    use_the_model_cascade,
                                    detection_scores);
    return;
}


void integral_channels_detector(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        dev_mem_2d_float_t& detection_scores)
{
    integral_channels_detector_impl(integral_channels,
                                    search_range_index,
                                    search_range,
                                    detection_cascade_per_scale,
                                    use_the_model_cascade,
                                    detection_scores);
    return;
}


void integral_channels_detector(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        gpu_detection_three_stumps_cascade_per_scale_t &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        dev_mem_2d_float_t& detection_scores)
{
    integral_channels_detector_impl(integral_channels,
                                    search_range_index,
                                    search_range,
                                    detection_cascade_per_scale,
                                    use_the_model_cascade,
                                    detection_scores);
    return;
}
// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// this method directly adds elements into the gpu_detections vector
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_impl(gpu_integral_channels_t &integral_channels,
                                     const size_t search_range_index,
                                     const doppia::ScaleData &scale_data,
                                     GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
                                     const float score_threshold,
                                     const bool use_the_model_cascade,
                                     gpu_detections_t& gpu_detections,
                                     size_t &num_detections)
{

    const doppia::DetectorSearchRange &search_range = scale_data.scaled_search_range;

    gpu_scale_datum_t scale_datum;
    {
        scale_datum.search_range.min_corner().x( search_range.min_x );
        scale_datum.search_range.min_corner().y( search_range.min_y );
        scale_datum.search_range.max_corner().x( search_range.max_x );
        scale_datum.search_range.max_corner().y( search_range.max_y );

        scale_datum.stride.x( scale_data.stride.x() );
        scale_datum.stride.y( scale_data.stride.y() );
    }


    //if((search_range.detection_window_scale/search_range.range_scaling) == 1)
    if(false)
    {
        printf("Occlusion type == %i\n", search_range.detector_occlusion_type);
        printf("integral_channels_detector_impl search range min (x,y) == (%.3f, %.3f), max (x,y) == (%.3f, %.3f)\n",
               (search_range.min_x/search_range.range_scaling),
               (search_range.min_y/search_range.range_scaling),
               (search_range.max_x/search_range.range_scaling),
               (search_range.max_y/search_range.range_scaling));

        //throw std::runtime_error("Stopping everything so you can debug");
    }

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    //const int nthreads = 320; // we optimize for images of width 640 pixel
    //dim3 block_dimensions(32, 10);
    //const int block_x = std::max(4, width/5), block_y = std::max(1, 256/block_x);

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            block_x = 16,
            block_y = num_threads / block_x;
    dim3 block_dimensions(block_x, block_y);

    const int
            width = search_range.max_x - search_range.min_x,
            height = search_range.max_y - search_range.min_y;

    if((width <= 0) or (height <= 0))
    { // nothing to be done
        // num_detections is left unchanged
        return;
    }

    dim3 grid_dimensions(div_up(width, block_dimensions.x),
                         div_up(height, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    if(use_the_model_cascade)
    {
        integral_channels_detector_kernel
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (scale_datum,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       score_threshold,
                                                       gpu_detections);
    }
    else
    {
        integral_channels_detector_kernel
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (scale_datum,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       score_threshold,
                                                       gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, scale_data, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, scale_data, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, scale_data, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_three_stumps_cascade_per_scale_t  &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, scale_data, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


/// This kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is based on integral_channels_detector_kernel,
/// but does detections for all scales instead of a single scale
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v0(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int max_search_range_width, const int max_search_range_height,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y;

    if((delta_x >= max_search_range_width) or (delta_y >= max_search_range_height))
    {
        // out of area of interest
        return;
    }

    const int
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];

    for(int scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        // when using softcascades, __syncthread here is a significant slow down, so not using it

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        // we order the if conditions putting most likely ones first
        if( (y > search_range.max_corner().y())
                or (y < search_range.min_corner().y())
                or (x < search_range.min_corner().x())
                or (x > search_range.max_corner().x()) )
        {
            // current pixel is out of this scale search range, we skip computations
            continue;
        }

        //bool should_skip_scale = false;

        // retrieve current score value
        float detection_score = 0;

        const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        for(int stage_index = 0; stage_index < cascade_length; stage_index += 1)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            update_detection_score(x, y, stage, integral_channels, detection_score);

            if((not use_the_model_cascade)
                    and use_hardcoded_cascade
                    and (stage_index > hardcoded_cascade_start_stage)
                    and (detection_score < hardcoded_cascade_threshold))
            {
                // this is not an object of the class we are looking for
                // do an early stop of this pixel

                // FIXME this is an experiment
                // since this is a really bad detection, we also skip one scale
                //scale_index += 1;
                //scale_index += 3;
                //scale_index += 10;
                //should_skip_scale = true;
                break;
            }

            if(use_the_model_cascade
                    //and scale_index > 100 // force evaluation of the first 100 stages
                    and detection_score < stage.cascade_threshold)
            {
                // this is not an object of the class we are looking for
                // do an early stop of this pixel
                detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

                // FIXME this is an experiment
                // since this is a really bad detection, we also skip one scale
                //if(stage_index < hardcoded_cascade_start_stage)
                {
                    //scale_index += 1;
                    //scale_index += 3;
                }

                const bool use_hardcoded_scale_skipping_from_cascade = false;
                if(use_hardcoded_scale_skipping_from_cascade)
                {

                    // These are the "good enough" values (7.6% curve in INRIA, slightly worse than best results)
                    // these values give a slowdown instead of the desired speed-up
                    /*                    if(stage_index < 10)
                    {
                        scale_index += 5;
                    }
                    else  if(stage_index < 20)
                    {
                        scale_index += 2;
                    }
                    else if(stage_index < 75)
                    {
                        scale_index += 1;
                    }*/

                    // This provide overlap with FPDW, speed up to 17 Hz from 15 Hz
                    if(stage_index < 20)
                    {
                        scale_index += 10;
                    }
                    else if(stage_index < 100)
                    {
                        scale_index += 4;
                    }
                    else if(stage_index < 200)
                    {
                        scale_index += 2;
                    }
                    else if(stage_index < 300)
                    {
                        scale_index += 1;
                    }

                }

                break;
            }


        } // end of "for each stage"

        if(detection_score > max_detection_score)
        {
            max_detection_score = detection_score;
            max_detection_scale_index = scale_index;
        }

        const bool use_hardcoded_scale_skipping = false;
        if(use_hardcoded_scale_skipping)
        {
            // these values are only valid when not using a soft-cascade

            // these are the magic numbers for the INRIA dataset,
            // when using 2011_11_03_1800_full_multiscales_model.proto.bin
            if(detection_score < -0.3)
            {
                scale_index += 11;
            }
            else if(detection_score < -0.2)
            {
                scale_index += 4;
            }
            else if(detection_score < -0.1)
            {
                scale_index += 2;
            }
            else if(detection_score < 0)
            {
                scale_index += 1;
            }
            else
            {
                // no scale jump
            }

        }

        /*
        // we only skip the scale if all the pixels in the warp agree
        if(__all(should_skip_scale))
        {
            // FIXME this is an experiment
            //scale_index += 1;
            scale_index += 3;
            //scale_index += 10;
        }
*/

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel




/// This kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is based on integral_channels_detector_kernel,
/// but does detections for all scales instead of a single scale
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
// _v1 is slower than _v0
void integral_channels_detector_over_all_scales_kernel_v1(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int max_search_range_width, const int max_search_range_height,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            thread_id = blockDim.x*threadIdx.y + threadIdx.x;


    const int num_threads = 192;
    __shared__  DetectionCascadeStageType cascade_stages[num_threads];

    const int
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = -1;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];
    for(int scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        // FIXME _sync, will not work as it is when skipping the scales
        // should create next_scale_to_evaluate variable; if (scale_index < next_scale_to_evaluate) continue;
        __syncthreads(); // all threads should move from one scale to the next at the same peace
        // (adding this syncthreads allowed to move from 1.53 Hz to 1.59 Hz)

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        const bool inside_search_range = ( (y >= search_range.min_corner().y())
                                           and (y < search_range.max_corner().y())
                                           and (x >= search_range.min_corner().x())
                                           and (x < search_range.max_corner().x()) );

        bool should_update_score = inside_search_range;
        //bool should_skip_scale = false;

        // retrieve current score value
        float detection_score = 0;

        if(inside_search_range == false)
        {
            detection_score = -1E10; // we set to a very negative value
        }

        int max_loaded_stage = 0, stage_index_modulo = 0;

        for(int stage_index = 0; stage_index < cascade_length; stage_index += 1, stage_index_modulo += 1)
        {

            if(stage_index >= max_loaded_stage)
            { // time to reload the next batch of cascade stages
                //__syncthreads();

                const int thread_stage_index = max_loaded_stage + thread_id;
                if(thread_stage_index < cascade_length)
                {
                    cascade_stages[thread_id] = detection_cascade_per_scale.data[scale_offset + thread_stage_index];
                }

                __syncthreads();
                max_loaded_stage += num_threads;
                stage_index_modulo = 0;
            }


            if(should_update_score)
            {
                // copying the stage is faster than referring to the shared memory
                const DetectionCascadeStageType stage = cascade_stages[stage_index_modulo];
                //const DetectionCascadeStageType stage = detection_cascade_per_scale.data[scale_offset + stage_index];

                update_detection_score(x, y, stage, integral_channels, detection_score);


                if(use_the_model_cascade and detection_score < stage.cascade_threshold)
                {
                    // this is not an object of the class we are looking for
                    // do an early stop of this pixel
                    detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

                    // FIXME this is an experiment
                    // since this is a really bad detection, we also skip one scale
                    //if(stage_index < hardcoded_cascade_start_stage)
                    {
                        //scale_index += 1;
                        //scale_index += 3;
                    }

                    should_update_score = false;
                } // end of "if crossed detection cascade"

            } // end of "should update score"

        } // end of "for each stage"

        if(detection_score > max_detection_score)
        {
            max_detection_score = detection_score;
            max_detection_scale_index = scale_index;
        }

        const bool use_hardcoded_scale_skipping = false;
        if(use_hardcoded_scale_skipping)
        {
            // these values are only valid when not using a soft-cascade

            // these are the magic numbers for the INRIA dataset,
            // when using 2011_11_03_1800_full_multiscales_model.proto.bin
            if(detection_score < -0.3)
            {
                scale_index += 11;
            }
            else if(detection_score < -0.2)
            {
                scale_index += 4;
            }
            else if(detection_score < -0.1)
            {
                scale_index += 2;
            }
            else if(detection_score < 0)
            {
                scale_index += 1;
            }
            else
            {
                // no scale jump
            }

        }

        /*
        // we only skip the scale if all the pixels in the warp agree
        if(__all(should_skip_scale))
        {
            // FIXME this is an experiment
            //scale_index += 1;
            scale_index += 3;
            //scale_index += 10;
        }
*/

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_scale_index >= 0 and max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v1


/// _v2 is significantly faster than _v0
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v2(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int /*max_search_range_width*/, const int /*max_search_range_height*/,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];

    for(int scale_index = 0; scale_index < num_scales; scale_index +=1)
    {
        // when using softcascades, __syncthreads here is a significant slow down, so not using it

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        // we order the if conditions putting most likelly ones first
        if( (y > search_range.max_corner().y())
                or (y < search_range.min_corner().y())
                or (x < search_range.min_corner().x())
                or (x > search_range.max_corner().x()) )
        {
            // current pixel is out of this scale search range, we skip computations
            // (nothing to do here)
        }
        else
        { // inside search range

            // retrieve current score value
            float detection_score = 0;

            const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

            for(int stage_index = 0; stage_index < cascade_length; stage_index += 1)
            {
                const int index = scale_offset + stage_index;

                // we copy the cascade stage from global memory to thread memory
                // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
                const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

                if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
                {
                    update_detection_score(x, y, stage, integral_channels, detection_score);
                }
                else
                {
                    // detection score is below cascade threshold,
                    // we are not interested on this object
                    detection_score = -1E5; // set to a value lower than score_threshold
                    break;
                }

            } // end of "for each stage"

            if(detection_score > max_detection_score)
            {
                max_detection_score = detection_score;
                max_detection_scale_index = scale_index;
            }

        } // end of "inside search range or not"

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v2


/// this function will evaluate the detection score of a specific window,
/// it is assumed that border checks have already been done
/// this method will directly call add_detection(...) if relevant
/// used in integral_channels_detector_over_all_scales_kernel_v3 and superior
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
inline
__device__
void compute_specific_detection(
        const int x, const int y, const int scale_index,
        const float score_threshold,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData &detection_cascade_per_scale,
        gpu_detections_t::KernelData &gpu_detections)
{
    const int
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    // retrieve current score value
    float detection_score = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    int stage_index = 0;
    for(; stage_index < cascade_length; stage_index += 1)
    {
        const int index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        if(use_the_model_cascade and (detection_score <= stage.cascade_threshold))
        {
            // detection score is below cascade threshold,
            // we are not interested on this object
            break;
        }

        update_detection_score(x, y, stage, integral_channels, detection_score);

    } // end of "for each stage"

    // >= to be consistent with Markus's code
    if((detection_score >= score_threshold) and (stage_index >= cascade_length))
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);

    } // end of "if detection score is high enough"

    return;
}


/// _v3 should only be used with integral_channels_detector_over_all_scales_impl_v1
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v3(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
    // (in current code, we ignore the gpu_scale_datum_t stride value)

    const int
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v3



/// _v4 should only be used with integral_channels_detector_over_all_scales_impl_v2
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v4_xy_stride(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (blockIdx.y * blockDim.y + threadIdx.y)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v4


//#define USE_GENERATED_CODE

// this file was created using the json data that came from
// void BaseVeryFastIntegralChannelsDetector::compute_scaled_detection_cascades()
#if defined(USE_GENERATED_CODE)
//#include "integral_channels_detector_over_all_scales_kernel_v4_generated.cuda_include"
#include "generated_integral_channels_detector.all_scales_kernel.cuda_include"
#endif


/// _v5 should only be used with integral_channels_detector_over_all_scales_impl_v3_coalesced
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v5_coalesced(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const int max_search_range_height,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{

    const int
            y_id = blockIdx.y * blockDim.y + threadIdx.y,
            //y_id_modulo = y_id % max_search_range_height,
            //scale_index = (y_id - y_id_modulo) / max_search_range_height,
            // we exploit the behaviour of integer division to avoid doing a costly modulo operation
            scale_index = y_id / max_search_range_height,
            num_scales = scales_data.size[0];


    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }


    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;


    const int
            y_id_modulo = y_id - (scale_index*max_search_range_height),
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (y_id_modulo)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range
        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);
    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v5_coalesced




/// this method directly adds elements into the gpu_detections vector
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v0(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;
    const int
            max_search_range_min_x = max_search_range.min_corner().x(),
            max_search_range_min_y = max_search_range.min_corner().y(),
            max_search_range_width = max_search_range.max_corner().x() - max_search_range_min_x,
            max_search_range_height = max_search_range.max_corner().y() - max_search_range_min_y;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            block_y = 2, block_x = num_threads / block_y; // slightly faster than block_y = 4

    dim3 block_dimensions(block_x, block_y);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // _v1 is slower than _v0; we should use _v0
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v2
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          max_search_range_min_x, max_search_range_min_y,
                                                          max_search_range_width, max_search_range_height,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v2
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          max_search_range_min_x, max_search_range_min_y,
                                                          max_search_range_width, max_search_range_height,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
}


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v1(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            block_z = 8,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    dim3 block_dimensions(block_x, block_y, block_z);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y),
                         div_up(num_scales, block_z));


    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v3
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v3
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v1


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v2_xy_stride(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //block_z = 8,
            block_z = 1,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4
            block_y = 16, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    // FIXME should use the stride information when setting the block sizes ?

    dim3 block_dimensions(block_x, block_y, block_z);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y),
                         div_up(num_scales, block_z));

    if(false)
    {
        printf("integral_channels_detector_over_all_scales_impl_v2_xy_stride block dimensions "
               "x, y, z == %i, %i, %i\n",
               block_dimensions.x, block_dimensions.y, block_dimensions.z);

        printf("integral_channels_detector_over_all_scales_impl_v2_xy_stride grid dimensions "
               "x, y, z == %i, %i, %i\n",
               grid_dimensions.x, grid_dimensions.y, grid_dimensions.z);
    }

    if((max_search_range_width <= 0) or (max_search_range_height <= 0))
    { // nothing to be done
        // num_detections is left unchanged
        return;
    }

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
#if defined(USE_GENERATED_CODE)
    static bool first_call = true;
    if(first_call)
    {
        printf("Using integral_channels_detector_over_all_scales_kernel_v4_generated\n");
        first_call = false;
    }

    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v4_generated
                <true>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          score_threshold,
                                                          gpu_detections);



    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v4_generated
                <false>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          score_threshold,
                                                          gpu_detections);
    }
#else
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v4_xy_stride
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v4_xy_stride
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
#endif // end of "use generated detector files or not"

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v2_xy_stride






/// in _v2 blocks span x,y, scale
/// in _v3 blocks span only x,y (scales are handled via an repetitions in the 2d domain)
/// this way all the accesses in the block are coalesced
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v3_coalesced(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &/*max_search_range*/,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //num_threads = 160, // to fit the 640 horizontal pixels
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            //block_z = 8,
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y*block_z); // slightly faster than block_y = 4
            // in this version, block_y = 16 (or 32) is critical for the obtained speed
            block_y = 16, block_x = num_threads / (block_y);


    dim3 block_dimensions(block_x, block_y);

    // we map the scales to repetitions on the y axis
    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height*num_scales, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v5_coalesced
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          max_search_range_height,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v5_coalesced
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          max_search_range_height,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v3_coalesced




/// this function will evaluate the detection score of a specific window,
/// it is assumed that border checks have already been done
/// this method will directly call add_detection(...) if relevant
/// used in integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread
/// will modify should_evaluate_neighbour based on the hints from the score progress
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
inline
__device__
void compute_specific_detection(
        const int x, const int y, const int scale_index,
        const float score_threshold, bool &should_evaluate_neighbour,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData &detection_cascade_per_scale,
        gpu_detections_t::KernelData &gpu_detections)
{
    const int
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    // retrieve current score value
    float detection_score = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    int stage_index = 0;
    for(; stage_index < cascade_length; stage_index += 1)
    {
        const int index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
        {
            update_detection_score(x, y, stage, integral_channels, detection_score);
        }
        else
        {
            // detection score is below cascade threshold,
            // we are not interested on this object
            detection_score = -1E5; // set to a value lower than score_threshold
            break;
        }

    } // end of "for each stage"

    // if we got far in the cascade, then we know that this was "close to be a detection"
    should_evaluate_neighbour = stage_index > (cascade_length / 3);

    // >= to be consistent with Markus's code
    if(detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);

    } // end of "if detection score is high enough"

    return;
}


template <bool use_the_model_cascade,
          bool use_excitation,
          typename DetectionCascadeStageType>
inline
__device__
void compute_specific_detection_score_for_two_pixels(
        const int x, const int y, const int scale_index,
        const float score_threshold,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData &detection_cascade_per_scale,
        gpu_detections_t::KernelData &gpu_detections)
{

    if(use_excitation)
    { // we only evaluate the second pixel, if the first pixel was "close to be a detection"

        bool should_evaluate_neighbour = false;
        // first pixel
        compute_specific_detection<use_the_model_cascade,DetectionCascadeStageType>
                (x, y, scale_index, score_threshold, should_evaluate_neighbour,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

        if(should_evaluate_neighbour)
        {
            // second pixel
            compute_specific_detection<use_the_model_cascade,DetectionCascadeStageType>
                    (x, y+1, scale_index, score_threshold,
                     integral_channels, detection_cascade_per_scale, gpu_detections);
        }

    }
    else
    { // we always evaluate both pixels

        // first pixel
        compute_specific_detection<use_the_model_cascade,DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

        // second pixel
        compute_specific_detection<use_the_model_cascade,DetectionCascadeStageType>
                (x, y+1, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    }

    return;
}

/// _v6 should only be used with integral_channels_detector_over_all_scales_impl_v4_two_pixels_per_thread
/// each thread process two pixels in the vertical direction
template <bool use_the_model_cascade,
          bool use_excitation,
          typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];


    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;


    const int
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (blockIdx.y * blockDim.y + threadIdx.y)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection_score_for_two_pixels
                <use_the_model_cascade, use_excitation, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread



template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v4_two_pixels_per_thread(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //block_z = 8,
            block_z = 1,
            //num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            //num_threads = 128, // ~?? Hz
            num_threads = 160, // ~13.10 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4
            block_y = 16, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    // FIXME should use the stride information when setting the block sizes ?

    dim3 block_dimensions(block_x, block_y, block_z);

    // each pixel will process two pixels in the vertical dimension
    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y) / 2,
                         div_up(num_scales, block_z));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
    // v5 is like v4, but uses two pixels per thread


    //const bool use_excitation = false; // speed == 13.10 [Hz]
    const bool use_excitation = true; // speed == 19.32 [Hz], quality ??

    if(use_excitation)
    {
        printf("Will use naive excitation, for two pixels\n");
    }

    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread
                <true, use_excitation, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread
                <false, use_excitation, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v4_two_pixels_per_thread


// we use unsigned int instead of float, because of atomicMax supported types
// we choose unsigned int instead of int to make the scores initialization simpler (0 is the lowest value)
// we assumes scores are in the range -10, +10 to do the uint to float mapping
typedef unsigned int inhibition_checkpoints_score_t;

// we will have only 5 checkpoints, this is hardcoded
typedef Cuda::DeviceMemoryPitched3D<inhibition_checkpoints_score_t> inhibition_scores_map_t;

inhibition_scores_map_t inhibition_scores_map;

/// _v7 should only be used with integral_channels_detector_over_all_scales_impl_v5_scales_inhibition
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v7_scales_inhibition(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections,
        inhibition_scores_map_t::KernelData inhibition_score_checkpoints_map)
{
    const int
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (blockIdx.y * blockDim.y + threadIdx.y)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        const int
                cascade_length = detection_cascade_per_scale.size[0],
                scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        const int checkpoint_zero_index = (y*inhibition_score_checkpoints_map.stride[1])
                + (x*inhibition_score_checkpoints_map.stride[0]);
        inhibition_checkpoints_score_t *inhibition_score_checkpoints =
                &inhibition_score_checkpoints_map.data[checkpoint_zero_index];


        // retrieve current score value
        float detection_score = 0;

        // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
        // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

        const int supremum_checkpoint_stage_index = 32 + 1;

        for(int stage_index = 0; stage_index < supremum_checkpoint_stage_index; stage_index += 1)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
            {
                update_detection_score(x, y, stage, integral_channels, detection_score);

                // check for inhibition checkpoint scores, and updates if needed
                {

                    // to be useful the inhibition needs to be able to abort things before the soft cascade does
                    // since most candidate windows have very few evaluations,
                    // the scales inhibition check points are on the very early stages.

                    // FIXME is the compiler smart enough to put the ifs/elses outside of the for loop ?
                    // (or we should do that by hand/templates ?)
                    inhibition_checkpoints_score_t *score_checkpoint_p = NULL;
                    /*if(stage_index == 2)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[0];
                    }
                    else */if(stage_index == 4)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[1];
                    }
                    else if(stage_index == 8)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[2];
                    }
                    /*else if(stage_index == 16)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[3];
                    }*/
                    else if(stage_index == 32)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[4];
                    }

                    if(score_checkpoint_p != NULL)
                    {
                        // we convert from uint to float
                        const float score_checkpoint = (*score_checkpoint_p - 1e8f)/1e7f;

                        if(detection_score <= score_checkpoint)
                        {

                            // detection score at current checkpoint is below the desired score,
                            // this means that at the same position, but different scale someone has already
                            // found a much stronger detection
                            detection_score = -1E5; // set to a value lower than score_threshold
                            break;
                        }
                        else
                        { // detection_score > score_checkpoint

                            // the current detection has the highest score up to now,
                            // we will update the score map

                            // FIXME this is an horrible heuristic
                            //const float new_checkpoint_float_score = detection_score - 0; // quality drop
                            //const float new_checkpoint_float_score = detection_score - 0.015; // quality 6.6 %
                            //const float new_checkpoint_float_score = detection_score - 0.005; // quality 7.1%
                            const float new_checkpoint_float_score = detection_score - 0.01; // quality 6.6 %

                            // we convert from float to int
                            const inhibition_checkpoints_score_t new_checkpoint_score =
                                    static_cast<inhibition_checkpoints_score_t>((new_checkpoint_float_score * 1e7f) + 1e8f);

                            const inhibition_checkpoints_score_t old =
                                    atomicMax(score_checkpoint_p, new_checkpoint_score);

                            //const bool print_checkpoints = true;
                            const bool print_checkpoints = false;
                            if(print_checkpoints)
                            {
                                if(x == 10 and y == 10)
                                {
                                    printf("At (x, y, scale_index) == (%i, %i, %i), stage %i, old checkpoint score %i, new checkpoint score %i\n",
                                           x, y, scale_index, stage_index, old, new_checkpoint_score);
                                }
                            }
                        }

                    } // end of "if at checkpoint stage"

                } // end of "inhibition checkpoint scores handling"

            }
            else
            {
                // detection score is below cascade threshold,
                // we are not interested on this object
                detection_score = -1E5; // set to a value lower than score_threshold
                break;
            }

        } // end of "for each stage where we may check for inhibition across scales"


        for(int stage_index=supremum_checkpoint_stage_index; stage_index < cascade_length; stage_index += 1)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
            {
                update_detection_score(x, y, stage, integral_channels, detection_score);
            }
            else
            {
                // detection score is below cascade threshold,
                // we are not interested on this object
                detection_score = -1E5; // set to a value lower than score_threshold
                break;
            }

        } // end of "for each stage"


        // >= to be consistent with Markus's code
        if(detection_score >= score_threshold)
        {
            // we got a detection
            add_detection(gpu_detections, x, y, scale_index, detection_score);

        } // end of "if detection score is high enough"


    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v7_scales_inhibition


/// _v8 should only be used with integral_channels_detector_over_all_scales_impl_v6_coalesced_scales_inhibition
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v8_coalesced_scales_inhibition(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const int max_search_range_height,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections,
        inhibition_scores_map_t::KernelData inhibition_score_checkpoints_map)
{


    const int
            y_id = blockIdx.y * blockDim.y + threadIdx.y,
            //y_id_modulo = y_id % max_search_range_height,
            //scale_index = (y_id - y_id_modulo) / max_search_range_height,
            // we exploit the behaviour of integer division to avoid doing a costly modulo operation
            scale_index = y_id / max_search_range_height,
            num_scales = scales_data.size[0];


    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }


    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;


    const int
            y_id_modulo = y_id - (scale_index*max_search_range_height),
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (y_id_modulo)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        const int
                cascade_length = detection_cascade_per_scale.size[0],
                scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        const int checkpoint_zero_index = (y*inhibition_score_checkpoints_map.stride[1])
                + (x*inhibition_score_checkpoints_map.stride[0]);
        inhibition_checkpoints_score_t *inhibition_score_checkpoints =
                &inhibition_score_checkpoints_map.data[checkpoint_zero_index];


        // retrieve current score value
        float detection_score = 0;

        // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
        // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

        for(int stage_index = 0; stage_index < cascade_length; stage_index += 1)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
            {
                update_detection_score(x, y, stage, integral_channels, detection_score);

                // check for inhibition checkpoint scores, and updates if needed
                {

                    // to be useful the inhibition needs to be able to abort things before the soft cascade does
                    // since most candidate windows have very few evaluations,
                    // the scales inhibition check points are on the very early stages.

                    // FIXME is the compiler smart enough to put the ifs/elses outside of the for loop ?
                    // (or we should do that by hand/templates ?)
                    inhibition_checkpoints_score_t *score_checkpoint_p = NULL;
                    if(stage_index == 2)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[0];
                    }
                    else if(stage_index == 4)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[1];
                    }
                    else if(stage_index == 8)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[2];
                    }
                    else if(stage_index == 16)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[3];
                    }
                    else if(stage_index == 32)
                    {
                        score_checkpoint_p = &inhibition_score_checkpoints[4];
                    }

                    if(score_checkpoint_p != NULL)
                    {
                        // we convert from uint to float
                        const float score_checkpoint = (*score_checkpoint_p - 1e8f)/1e7f;

                        if(detection_score <= score_checkpoint)
                        {

                            // detection score at current checkpoint is below the desired score,
                            // this means that at the same position, but different scale someone has already
                            // found a much stronger detection
                            detection_score = -1E5; // set to a value lower than score_threshold
                            break;
                        }
                        else
                        { // detection_score > score_checkpoint

                            // the current detection has the highest score up to now,
                            // we will update the score map

                            // FIXME this is an horrible heuristic
                            //const float new_checkpoint_float_score = detection_score - 0.015;
                            const float new_checkpoint_float_score = detection_score - 0;

                            // we convert from float to int
                            const inhibition_checkpoints_score_t new_checkpoint_score =
                                    static_cast<inhibition_checkpoints_score_t>((new_checkpoint_float_score * 1e7f) + 1e8f);

                            const inhibition_checkpoints_score_t old =
                                    atomicMax(score_checkpoint_p, new_checkpoint_score);

                            //const bool print_checkpoints = true;
                            const bool print_checkpoints = false;
                            if(print_checkpoints)
                            {
                                if(x == 10 and y == 10)
                                {
                                    printf("At (x, y, scale_index) == (%i, %i, %i), stage %i, old checkpoint score %i, new checkpoint score %i\n",
                                           x, y, scale_index, stage_index, old, new_checkpoint_score);
                                }
                            }
                        }

                    } // end of "if at checkpoint stage"

                } // end of "inhibition checkpoint scores handling"

            }
            else
            {
                // detection score is below cascade threshold,
                // we are not interested on this object
                detection_score = -1E5; // set to a value lower than score_threshold
                break;
            }

        } // end of "for each stage"

        // >= to be consistent with Markus's code
        if(detection_score >= score_threshold)
        {
            // we got a detection
            add_detection(gpu_detections, x, y, scale_index, detection_score);

        } // end of "if detection score is high enough"


    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v8_coalesced_scales_inhibition




void initalize_inhibition_scores_map(const gpu_integral_channels_t &integral_channels)
{
    // (re)-allocate the inhibition_scores_map
    if((inhibition_scores_map.size[1] != integral_channels.size[0]) or
            (inhibition_scores_map.size[2] != integral_channels.size[1]))
    {
        // we will have only 5 checkpoints, this is hardcoded
        const size_t num_rows = integral_channels.size[0], num_columns = integral_channels.size[1];
        printf("Going to allocate a inhibition_scores_map of size (rows, cols) == (%zu, %zu)\n",
               num_rows, num_columns);
        inhibition_scores_map.realloc(5, num_rows, num_columns);
    }

    // we clean-up the inhibition scores map, we set the checkpoint score to a very low (unsigned int) value
    {
        inhibition_scores_map.initMem(0);
    }
    return;
}

/// this version is similar to _v2, but it additionally implements
/// inhibition across scales
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v5_scales_inhibition(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //block_z = 8,
            block_z = 1,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4
            block_y = 16, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    // FIXME should use the stride information when setting the block sizes ?
    dim3 block_dimensions(block_x, block_y, block_z);

    // each pixel will process two pixels in the vertical dimension
    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y),
                         div_up(num_scales, block_z));

    // prepare variables for kernel call --
    initalize_inhibition_scores_map(integral_channels);
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
    // v5 is like v4, but uses two pixels per thread
    // v7 implements scales inhibition

    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v7_scales_inhibition
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections,
                                                          inhibition_scores_map);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v7_scales_inhibition
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections,
                                                          inhibition_scores_map);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v5_scales_inhibition



/// this version is similar to _v3_coalesced, but it additionally implements
/// inhibition across scales
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v6_coalesced_scales_inhibition(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{
    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //block_z = 8,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4
            // in this version, block_y = 16 (or 32) is critical for the obtained speed
            block_y = 16, block_x = num_threads / (block_y);

    // FIXME should use the stride information when setting the block sizes ?
    dim3 block_dimensions(block_x, block_y);

    // we map the scales to repetitions on the y axis
    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height*num_scales, block_dimensions.y));

    // prepare variables for kernel call --
    initalize_inhibition_scores_map(integral_channels);
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
    // v5 is like v4, but uses two pixels per thread
    // v7 implements scales inhibition

    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v8_coalesced_scales_inhibition
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          max_search_range_height,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections,
                                                          inhibition_scores_map);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v8_coalesced_scales_inhibition
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          max_search_range_height,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections,
                                                          inhibition_scores_map);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v6_coalesced_scales_inhibition


// 6 pixels per block => 192 threads
// 8 pixels per block => 256 threads
//#define NUM_PIXELS_PER_BLOCK 6
#define NUM_PIXELS_PER_BLOCK 8
//#define NUM_PIXELS_PER_BLOCK 12
//#define NUM_PIXELS_PER_BLOCK (6*8)

#define WARP_SIZE 32
//#define WARP_SIZE 16
//#define WARP_SIZE 4

/// _v9 should only be used with integral_channels_detector_over_all_scales_impl_v7
/// in this version, each warp works together to compute the detector response
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v9_one_pixel_per_warp(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    __shared__ float pixel_detection_scores[NUM_PIXELS_PER_BLOCK];

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            pixel_index = threadIdx.x,
            delta_stage = threadIdx.y,
            delta_x = ((blockIdx.x * blockDim.x) + threadIdx.x)*stride.x(),
            delta_y = blockIdx.y*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        const int
                cascade_length = detection_cascade_per_scale.size[0],
                scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        float &detection_score = pixel_detection_scores[pixel_index];
        if(delta_stage == 0) // only one thread initalizes
        {
            detection_score = 0;
        }
        __syncthreads(); // make sure initialization is done

        // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
        // (relevant speed diference)

        int stage_index = delta_stage;
        for(; stage_index < cascade_length; stage_index+=WARP_SIZE)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            if(use_the_model_cascade
                    and (detection_score <= stage.cascade_threshold))
            {
                // detection score is below cascade threshold,
                // we are not interested on this object
                break;
            }

            float delta_detection_score = 0; // set to zero
            update_detection_score(x, y, stage, integral_channels, delta_detection_score);
            atomicAdd(&detection_score, delta_detection_score);

        } // end of "for each stage"

        // score >= threshold to be consistent with Markus's code
        if((delta_stage == 0) // only one thread adds the detection
                and (detection_score >= score_threshold)
                and (stage_index >= cascade_length)) // only if we did not abort earlier
        {
            // we got a detection
            add_detection(gpu_detections, x, y, scale_index, detection_score);

        } // end of "if detection score is high enough"

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v9_one_pixel_per_warp


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v7_one_pixel_per_warp(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;


    const int
            num_scales = scales_data.getNumElements(),
            block_z = 1, // we handle one scale at a time
            //num_threads = 192, // speed ??
            //num_threads = 256, // speed ??
            warpSize = WARP_SIZE, // FIXME how to retrieve this number from API ?
            block_y = warpSize,
            //num_pixels_per_block = num_threads / (block_y * block_z),
            block_x = NUM_PIXELS_PER_BLOCK;

    // FIXME should use the stride information when setting the block sizes ?
    dim3 block_dimensions(block_x, block_y, block_z);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         max_search_range_height,
                         div_up(num_scales, block_z));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v9_one_pixel_per_warp
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v9_one_pixel_per_warp
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v7_one_pixel_per_warp



/// _v10 should only be used with integral_channels_detector_over_all_scales_impl_v8
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v10_many_scales_per_block(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            scale_index = (blockIdx.x * blockDim.x + threadIdx.x),
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = (blockIdx.y * blockDim.y + threadIdx.y)*stride.x(),
            delta_y = (blockIdx.z * blockDim.z + threadIdx.z)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v10_many_scales_per_block


/// In this implementation we try to get the block focusing on scales first,
/// and spatial dimensions second.
/// The idea is that a key blocking point on GPU speed is that the whole block need to finish,
/// before launching the next one. If we group spatially, we have to wait that all neighbours have finished.
/// We expect that (~50) scales are more co-related than (~50) neighbours
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v8_many_scales_per_block(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            max_num_threads = 192,
            block_x = num_scales,
            block_y = 1, // we expect vertical correlation to be stronger
            block_z = max_num_threads / block_x;

    // FIXME should use the stride information when setting the block sizes ?
    dim3 block_dimensions(block_x, block_y, block_z);
    //dim3 block_dimensions(2, 8, 8);


    dim3 grid_dimensions(div_up(num_scales, block_dimensions.x),
                         div_up(max_search_range_width, block_dimensions.y),
                         div_up(max_search_range_height, block_dimensions.z));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v10_many_scales_per_block
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v10_many_scales_per_block
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v8_many_scales_per_block


void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{
    // call the templated generic implementation
    // v1 is faster than v0 (?)
    // v2 handles the strides (v0 and v1 omit them), reaches ~19 Hz in monocular mode, ~57 Hz with ground plane and strides (4,8)
    // v3 handles strides and is fully coalesced, reaches ~18 Hz in monocular mode, ~54 Hz with ground plane and strides (4,8)
    // v2 in unuk reaches 55 Hz, v3 56 Hz (monocular mode)
    //integral_channels_detector_over_all_scales_impl_v0(
    //integral_channels_detector_over_all_scales_impl_v1(
    // v2_xy_stride is the version you want
    integral_channels_detector_over_all_scales_impl_v2_xy_stride(
                //integral_channels_detector_over_all_scales_impl_v3_coalesced(
                //integral_channels_detector_over_all_scales_impl_v4_two_pixels_per_thread (
                //integral_channels_detector_over_all_scales_impl_v5_scales_inhibition(
                //integral_channels_detector_over_all_scales_impl_v6_coalesced_scales_inhibition(
                //integral_channels_detector_over_all_scales_impl_v7_one_pixel_per_warp(
                //integral_channels_detector_over_all_scales_impl_v8_many_scales_per_block(
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                detection_cascade_per_scale,
                score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}



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
        size_t &num_detections)
{
    // v2_xy_stride is the version you want
    integral_channels_detector_over_all_scales_impl_v2_xy_stride(
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                detection_cascade_per_scale,
                score_threshold, use_the_model_cascade, gpu_detections, num_detections);

    return;
}




void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{
    // call the templated generic implementation
    // v1 is faster than v0 (?)
    // v2 handles the strides (v0 and v1 omit them)
    // v3 handles strides but is slower than v2
    //integral_channels_detector_over_all_scales_impl_v0(
    //integral_channels_detector_over_all_scales_impl_v1(
    integral_channels_detector_over_all_scales_impl_v2_xy_stride(
                //integral_channels_detector_over_all_scales_impl_v3_coalesced(
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                detection_cascade_per_scale,
                score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}

} // end of namespace objects_detection
} // end of namespace doppia

#include "integral_channels_detector_with_stixels.cuda_include"

