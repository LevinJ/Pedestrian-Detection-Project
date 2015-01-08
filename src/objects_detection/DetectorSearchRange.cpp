#include "DetectorSearchRange.hpp"

#include <stdexcept>
#include <cassert>

namespace doppia {

bool DetectorSearchRangeMetaData::operator ==(const DetectorSearchRangeMetaData &other) const
{
    const DetectorSearchRangeMetaData &self = *this;
    bool ret = true;
    ret &= self.detection_window_scale == other.detection_window_scale;
    ret &= self.detection_window_ratio == other.detection_window_ratio;
    ret &= self.range_scaling == other.range_scaling;
    ret &= self.range_ratio == other.range_ratio;

    ret &= self.detector_occlusion_level == other.detector_occlusion_level;
    ret &= self.detector_occlusion_type == other.detector_occlusion_type;
    ret &= self.semantic_category == other.semantic_category;

    return ret;
}

// no explicit constructor or destructor for DetectorSearchRange, this is essentially a simple struct

DetectorSearchRange DetectorSearchRange::get_rescaled(const float scaling, const float ratio) const
{
    if(scaling <= 0)
    {
        throw std::invalid_argument("DetectorSearchRange::get_rescaled expects a scaling factor > 0");
    }
    DetectorSearchRange scaled_range;

    scaled_range.range_scaling = range_scaling * scaling;
    scaled_range.range_ratio = range_ratio * ratio;
    scaled_range.detection_window_scale = detection_window_scale * scaling;
    scaled_range.detection_window_ratio = detection_window_ratio * ratio; // ratio is invariant to scaling
    // floor and ceil are needed to make sure we are not off by one pixel
    scaled_range.min_x = std::floor(min_x * scaling*ratio);
    scaled_range.max_x = std::ceil(max_x * scaling*ratio);
    scaled_range.min_y = std::floor(min_y * scaling);
    scaled_range.max_y = std::ceil(max_y * scaling);

    scaled_range.detector_occlusion_level = detector_occlusion_level; // no change
    scaled_range.detector_occlusion_type = detector_occlusion_type; // no change

    assert(scaled_range.range_scaling > 0);
    return scaled_range;
}


bool DetectorSearchRange::operator==(const DetectorSearchRange &other) const
{
    const DetectorSearchRange &self = *this;
    bool ret = true;
    ret &= self.detection_window_scale == other.detection_window_scale;
    ret &= self.detection_window_ratio == other.detection_window_ratio;
    ret &= self.range_scaling == other.range_scaling;
    ret &= self.range_ratio == other.range_ratio;
    ret &= self.min_x == other.min_x;
    ret &= self.max_x == other.max_x;
    ret &= self.min_y == other.min_y;
    ret &= self.max_y == other.max_y;

    ret &= self.detector_occlusion_level == other.detector_occlusion_level;
    ret &= self.detector_occlusion_type == other.detector_occlusion_type;

    return ret;
}


/// copy the shared members (and leave to other ones unchanged)
DetectorSearchRange &DetectorSearchRange::operator =(const DetectorSearchRangeMetaData &data)
{
    detection_window_scale = data.detection_window_scale;
    detection_window_ratio = data.detection_window_ratio;
    range_scaling = data.range_scaling;
    range_ratio = data.range_ratio;
    detector_occlusion_level = data.detector_occlusion_level;
    detector_occlusion_type = data.detector_occlusion_type;

    return *this;
}


} // end of namespace doppia
