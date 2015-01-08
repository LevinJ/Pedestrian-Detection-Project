#ifndef MODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
#define MODELWINDOWTOOBJECTWINDOWCONVERTER_HPP

#include "AbstractModelWindowToObjectWindowConverter.hpp"

namespace doppia
{

/// Helper class that will convert all the detections windows from
/// the model size window (area of the image used for detection),
/// to the object size window (area of the image that is expected to be a tigh box around the detected object)
class ModelWindowToObjectWindowConverter: public AbstractModelWindowToObjectWindowConverter
{
public:

    /// constructor to be used
    ModelWindowToObjectWindowConverter(const model_window_size_t&model_window_size,
                                       const object_window_t &object_window);
    ~ModelWindowToObjectWindowConverter();

    /// will convert the input detection from model_window to object_window
    /// this function is speed critical, since it will be called over all windows,
    /// before non-maximal suppresion
    void operator()(detections_t &detections) const;

protected:

    // object_width / model_width
    float object_width_to_model_width, object_height_to_model_height;

    // delta_x_to_model_width = (object_center_x - model_center_x) / model_width
    float delta_x_to_model_width, delta_y_to_model_height;


};


} // end of namespace doppia

#endif // MODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
