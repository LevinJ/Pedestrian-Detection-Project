#ifndef ABSTRACTNONMAXIMALSUPPRESSION_HPP
#define ABSTRACTNONMAXIMALSUPPRESSION_HPP

#include "objects_detection/AbstractObjectsDetector.hpp"

namespace doppia {

class AbstractNonMaximalSuppression
{
public:

    typedef AbstractObjectsDetector::detection_t detection_t;
    typedef AbstractObjectsDetector::detections_t detections_t;

    AbstractNonMaximalSuppression();
    virtual ~AbstractNonMaximalSuppression();

    /// Provide the (large) set of raw detections from the sliding window (or similar) detector
    virtual void set_detections(const detections_t &) = 0;

    virtual void compute() = 0;

    /// @returns the filtered set of detections
    virtual const detections_t &get_detections();

protected:
    detections_t maximal_detections;

};

} // end of namespace doppia

#endif // ABSTRACTNONMAXIMALSUPPRESSION_HPP
