#ifndef DOPPIA_ABSTRACTOBJECTSTRACKER_HPP
#define DOPPIA_ABSTRACTOBJECTSTRACKER_HPP

#include "objects_detection/AbstractObjectsDetector.hpp"


namespace doppia {

class AbstractObjectsTracker
{
public:

    typedef AbstractObjectsDetector::detection_t detection_t;
    typedef AbstractObjectsDetector::detections_t detections_t;


    AbstractObjectsTracker();
    virtual ~AbstractObjectsTracker();

    /// set the detections on current frame
    /// calling this function indicates the evolution of time
    /// (compute() must be called before requesting get_*() )
    virtual void set_detections(const detections_t &detections) = 0;

    virtual void set_image_size(const int width, const int height);

    /// update the internal state of the tracker
    virtual void compute() = 0;

    virtual const detections_t &get_current_detections() const = 0;

    /// get the expected detections for
    //virtual const detections_t &get_predicted_detections()  = 0;

    /// get the current tracks
    //virtual const detections_t &get_current_tracks()  = 0;

protected:

    int image_width, image_height;

};

} // namespace doppia

#endif // DOPPIA_ABSTRACTOBJECTSTRACKER_HPP
