#ifndef DOPPIA_FASTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
#define DOPPIA_FASTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP

#include "AbstractModelWindowToObjectWindowConverter.hpp"



namespace doppia
{

/// Helper class that will convert all the detections windows from
/// the model size window (area of the image used for detection),
/// to the object size window (area of the image that is expected to be a tigh box around the detected object)
class FastModelWindowToObjectWindowConverter: public AbstractModelWindowToObjectWindowConverter
{
public:

    /// default constructor
    FastModelWindowToObjectWindowConverter();

    /// constructor to be used
    FastModelWindowToObjectWindowConverter(const model_window_size_t&model_window_size,
                                       const object_window_t &object_window);
    ~FastModelWindowToObjectWindowConverter();

    /// will convert the input detection from model_window to object_window
    /// this function is speed critical, since it will be called over all windows,
    /// before non-maximal suppresion
    void operator()(detections_t &detections) const;

protected:

    float scaling_factor_x, scaling_factor_y;
};

} // namespace doppia

#endif // DOPPIA_FASTMODELWINDOWTOOBJECTWINDOWCONVERTER_HPP
