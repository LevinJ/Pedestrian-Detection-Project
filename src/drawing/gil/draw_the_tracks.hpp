#ifndef DOPPIA_DRAW_THE_TRACKS_HPP
#define DOPPIA_DRAW_THE_TRACKS_HPP

#include "objects_tracking/DummyObjectsTracker.hpp"

#include <map>

namespace doppia {

void draw_the_tracks(
        const DummyObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const boost::gil::rgb8_view_t &view);


} // end of namespace doppia

#endif // DOPPIA_DRAW_THE_TRACKS_HPP
