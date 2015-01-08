#ifndef DOPPIA_OBJECTS_DETECTION_ADD_DETECTION_CU_HPP
#define DOPPIA_OBJECTS_DETECTION_ADD_DETECTION_CU_HPP

#include "integral_channels_detector.cu.hpp"

namespace doppia {
namespace objects_detection {

/// type int because atomicAdd does not support size_t
extern __device__ int num_gpu_detections[1];

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


} // end of namespace objects_detection
} // end of namespace doppia

#endif // DOPPIA_OBJECTS_DETECTION_ADD_DETECTION_CU_HPP
