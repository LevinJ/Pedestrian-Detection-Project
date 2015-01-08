#ifndef DOPPIA_OBJECTS_DETECTION_GET_FEATURE_VALUE_TEX2D_CU_HPP
#define DOPPIA_OBJECTS_DETECTION_GET_FEATURE_VALUE_TEX2D_CU_HPP

#include "integral_channels_detector.cu.hpp"

namespace doppia {
namespace objects_detection {

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 2>::Texture gpu_integral_channels_2d_texture_t;
extern gpu_integral_channels_2d_texture_t integral_channels_2d_texture;

/// this method is used by the generated code
/// @see homonym methods
template <typename channel_index_t, typename box_coordinate_t>
inline
__device__
float get_feature_value_tex2d(
        const int x, const int y,
        const channel_index_t channel_index,
        const box_coordinate_t box_min_corner_x,
        const box_coordinate_t box_min_corner_y,
        const box_coordinate_t box_max_corner_x,
        const box_coordinate_t box_max_corner_y,
        const int integral_channels_height)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory
    const float y_offset = y + channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    const float
            a = tex2D(tex, x + box_min_corner_x, box_min_corner_y + y_offset), // top left
            b = tex2D(tex, x + box_max_corner_x, box_min_corner_y + y_offset), // top right
            c = tex2D(tex, x + box_max_corner_x, box_max_corner_y + y_offset), // bottom right
            d = tex2D(tex, x + box_min_corner_x, box_max_corner_y + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;
    return feature_value;
}

} // end of namespace objects_detection
} // end of namespace doppia

#endif // DOPPIA_OBJECTS_DETECTION_GET_FEATURE_VALUE_TEX2D_CU_HPP
