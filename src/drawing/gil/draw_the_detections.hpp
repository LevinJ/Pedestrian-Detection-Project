#ifndef DOPPIA_DRAW_THE_DETECTIONS_HPP
#define DOPPIA_DRAW_THE_DETECTIONS_HPP

#include "objects_detection/AbstractObjectsDetector.hpp"

namespace doppia {

/// Helper method for drawing (used by other graphical user interfaces)
void draw_the_detections(
        const AbstractObjectsDetector::detections_t &detections,
        const AbstractObjectsDetector::detections_t &ground_truth_detections,
        float &max_detection_score,
        const int additional_border,
        const boost::gil::rgb8_view_t &view);

} // end of namespace doppia

#endif // DOPPIA_DRAW_THE_DETECTIONS_HPP
