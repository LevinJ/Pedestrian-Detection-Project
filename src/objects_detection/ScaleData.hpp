#ifndef DOPPIA_SCALEDATA_HPP
#define DOPPIA_SCALEDATA_HPP

#include "DetectorSearchRange.hpp"

namespace doppia {

/// Helper structure to store data for each specific scale
/// @see BaseIntegralChannelsDetector
struct ScaleData
{
    typedef geometry::point_xy<size_t> image_size_t;
    typedef geometry::point_xy<boost::uint16_t> stride_t;
    typedef geometry::point_xy<boost::uint16_t> detection_window_size_t;

    image_size_t scaled_input_image_size;
    DetectorSearchRange scaled_search_range;
    detection_window_size_t scaled_detection_window_size;
    stride_t stride; ///< scaled x/y stride
};


} // namespace doppia

#endif // DOPPIA_SCALEDATA_HPP
