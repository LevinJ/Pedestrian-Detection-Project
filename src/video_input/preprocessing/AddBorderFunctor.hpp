#ifndef DOPPIA_ADDBORDERFUNCTOR_HPP
#define DOPPIA_ADDBORDERFUNCTOR_HPP

#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#else
#include "video_input/AbstractVideoInput.hpp"
#endif

namespace doppia {


/// Small helper class that adds a border to an image
class AddBorderFunctor
{
public:
#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    typedef boost::gil::rgb8_image_t image_t;
    typedef boost::gil::rgb8c_view_t image_view_t;
#else
    typedef AbstractVideoInput::input_image_t image_t;
    typedef AbstractVideoInput::input_image_view_t image_view_t;
#endif

    AddBorderFunctor(const int additional_border);
    ~AddBorderFunctor();

    image_view_t operator()(image_view_t &input_view);

    const int additional_border;

protected:

    image_t image_with_border;
    image_t::view_t image_with_border_view;

};


} // end of namespace doppia

#endif // DOPPIA_ADDBORDERFUNCTOR_HPP
