#ifndef DETECTORSEARCHRANGE_HPP
#define DETECTORSEARCHRANGE_HPP

#include "SoftCascadeOverIntegralChannelsModel.hpp"

#include <boost/cstdint.hpp>
#include <vector>

namespace doppia {


/// Base function that we will use to build DetectorSearchRange
/// during the application flow we first compute the meta data of the required search ranges,
/// and later on we compute the actual range.
class DetectorSearchRangeMetaData {

public:
    typedef SoftCascadeOverIntegralChannelsModel::occlusion_type_t occlusion_type_t;

    /// detection window scale to consider
    float detection_window_scale;

    /// ratio used for the current detection
    /// ratio == width / height
    float detection_window_ratio;

    /// is this range itself scaled ?
    float range_scaling;

    /// is this range itself ratio-adjusted ?
    float range_ratio;

    /// occlusion level of the detector associated to this search range
    float detector_occlusion_level;

    /// semantic category associated with this detector
    std::string semantic_category;

    /// occlusion type of the detector associated to this search range
    occlusion_type_t detector_occlusion_type;

    /// Needed to use this class as a key in a std::map
    bool operator==(const DetectorSearchRangeMetaData &other) const;

};


/// Helper class used by the detectors to define the set of scales to search
/// and where in the image search at each scale
class DetectorSearchRange: public DetectorSearchRangeMetaData {
public:

    /// when range_scaling == 1.0 the x and y coordinates are in the input image coordinates
    /// the x,y point corresponds the upper left position of the detection window
    /// this is _not_ the window center
    /// when range_scaling == 1.0 then scaled_max_x = original_max*range_scaling
    /// ( @warning max_x + scaled_detection_window_size may be out of range )
    boost::uint16_t min_x, max_x, min_y, max_y;

    DetectorSearchRange get_rescaled(const float scaling, const float ratio = 1.0f) const;

    /// Needed to use this class as a key in a std::map
    bool operator==(const DetectorSearchRange &other) const;

    /// copy the shared members (and leave to other ones unchanged)
    DetectorSearchRange &operator=(const DetectorSearchRangeMetaData &data);
};

/// we expect the search range to be ordered from smallest scale to largest scale
typedef std::vector<DetectorSearchRangeMetaData> detector_search_ranges_data_t;

/// we expect the search range to be ordered from smallest scale to largest scale
typedef std::vector<DetectorSearchRange> detector_search_ranges_t;

} // end of namespace doppia

#endif // DETECTORSEARCHRANGE_HPP
