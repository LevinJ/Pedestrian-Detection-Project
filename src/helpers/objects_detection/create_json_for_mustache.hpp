#ifndef DOPPIA_CREATE_JSON_FOR_MUSTACHE_HPP
#define DOPPIA_CREATE_JSON_FOR_MUSTACHE_HPP

#include "objects_detection/BaseIntegralChannelsDetector.hpp"

#include <vector>

namespace doppia {

void create_json_for_mustache(std::vector<BaseIntegralChannelsDetector::cascade_stages_t> &detection_cascade_per_scale);

} // end namespace doppia

#endif // DOPPIA_CREATE_JSON_FOR_MUSTACHE_HPP
