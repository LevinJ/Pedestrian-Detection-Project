#ifndef DOPPIA_ABSTRACTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
#define DOPPIA_ABSTRACTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP

#include "Detection2d.hpp"
#include "SoftCascadeOverIntegralChannelsModel.hpp" // for the model_window_size_t and object_window_t

#include <vector>

namespace doppia
{

/// Helper class that will convert all the detections windows from
/// the model size window (area of the image used for detection),
/// to the object size window (area of the image that is expected to be a tigh box around the detected object)
class AbstractModelWindowToObjectWindowConverter
{
public:

    typedef SoftCascadeOverIntegralChannelsModel::model_window_size_t model_window_size_t;
    typedef SoftCascadeOverIntegralChannelsModel::object_window_t object_window_t;
    typedef Detection2d detection_t;
    typedef std::vector<Detection2d> detections_t;


    AbstractModelWindowToObjectWindowConverter();
    virtual ~AbstractModelWindowToObjectWindowConverter();

    /// will convert the input detection from model_window to object_window
    /// this function is speed critical, since it will be called over all windows,
    /// before non-maximal suppresion
    virtual void operator()(detections_t &detections) const;

};

} // end of namespace doppia

#endif // DOPPIA_ABSTRACTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
