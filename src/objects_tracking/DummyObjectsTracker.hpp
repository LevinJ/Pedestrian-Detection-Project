#ifndef DOPPIA_DUMMYOBJECTSTRACKER_HPP
#define DOPPIA_DUMMYOBJECTSTRACKER_HPP

#include "AbstractObjectsTracker.hpp"
#include "tracked_detections/TrackedDetection2d.hpp"

#include <list>

namespace doppia {

/// Simple 2d objects tracker
class DummyObjectsTracker: public AbstractObjectsTracker
{
public:

    typedef TrackedDetection2d track_t;
    typedef std::list<track_t> tracks_t;

    typedef std::list<detection_t> new_detections_t;
public:
    static boost::program_options::options_description get_args_options();

    DummyObjectsTracker(const boost::program_options::variables_map &options);
   ~DummyObjectsTracker();

    /// set the detections on current frame
    /// calling this function indicates the evolution of time
    /// (compute() must be called before requesting get_*() )
    void set_detections(const detections_t &detections);

    /// update the internal state of the tracker
    void compute();

    const detections_t &get_current_detections() const;

    const tracks_t &get_tracks() const;

protected:

    /// ID given to the next track we will create
    int next_track_id;
    const int max_extrapolation_length;

    new_detections_t new_detections;
    detections_t current_detections;

    tracks_t tracks;

    bool track_is_outside_image(const track_t &track) const;
};

/// helper methods (used by Dummy3dObjectsTracker)
/// @{
bool box_is_outside_image(const TrackedDetection2d::rectangle_t &bbox, const int image_width, const int image_height);

/// PASCAL VOC criterion
float intersection_over_union(const TrackedDetection2d::rectangle_t &a, const TrackedDetection2d::rectangle_t &b);

/// @}

} // namespace doppia

#endif // DOPPIA_DUMMYOBJECTSTRACKER_HPP
