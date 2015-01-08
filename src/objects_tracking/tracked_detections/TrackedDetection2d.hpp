#ifndef DOPPIA_TRACKEDDETECTION2D_HPP
#define DOPPIA_TRACKEDDETECTION2D_HPP

#include "objects_tracking/AbstractObjectsTracker.hpp"

namespace doppia {


/// Helper class for DummyObjectsTracker
/// Represents a track in the 2d image plane (only)
class TrackedDetection2d {

public:
    typedef Detection2d::ObjectClasses object_class_t;

    typedef Detection2d::rectangle_t rectangle_t;
    typedef AbstractObjectsTracker::detection_t detection_t;
    typedef AbstractObjectsTracker::detections_t detections_t;


public:
    TrackedDetection2d(const int id, const detection_t &detection, const int max_extrapolation_length_);
    virtual ~TrackedDetection2d();

    /// found a match, we move forward in time
    virtual void add_matched_detection(const detection_t &detection);

    /// we did not found a match, we move forward in time
    virtual void skip_one_detection();


    /// Accessor methods
    /// @{
    int get_max_extrapolation_length() const;
    int get_extrapolation_length() const;
    size_t get_length() const;

    const detection_t &get_current_detection() const;
    const rectangle_t &get_current_bounding_box() const;

    const detections_t &get_detections_in_time() const;
    const int get_id() const;
    /// @}

public:
    /// (we mark this method as public just to make things easier in helper functions)
    rectangle_t compute_extrapolated_bounding_box() const;

    /// (we mark this method as public just to make things easier in helper functions)
    void set_current_bounding_box_as_occluded();

public:
    const object_class_t object_class;


protected:

    const int track_id;
    rectangle_t current_bounding_box;
    detections_t detections_in_time;
    float max_detection_score;


    const int max_extrapolation_length;
    int num_extrapolated_detections, num_true_detections_in_time;
    int num_consecutive_detections, max_consecutive_detections;
};

} // end of namespace doppia

#endif // DOPPIA_TRACKEDDETECTION2D_HPP
