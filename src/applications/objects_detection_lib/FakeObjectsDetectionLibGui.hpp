#ifndef DOPPIA_FAKEOBJECTSDETECTIONLIBGUI_HPP
#define DOPPIA_FAKEOBJECTSDETECTIONLIBGUI_HPP

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

namespace objects_detection {

class FakeObjectsDetectionLibGui
{
public:
    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    void set_monocular_input(input_image_const_view_t &);
    void set_left_input(input_image_const_view_t &);
    void set_right_input(input_image_const_view_t &);

    void update();
};

} // end of namespace objects_detection


#endif // DOPPIA_FAKEOBJECTSDETECTIONLIBGUI_HPP
