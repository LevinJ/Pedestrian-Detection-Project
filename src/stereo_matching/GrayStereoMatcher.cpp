#include "GrayStereoMatcher.hpp"

namespace doppia {

GrayStereoMatcher::GrayStereoMatcher()
{
}



if (rgb2gray and true) // (left_image.current_type_is<rgb8_image_t>() == true) )
{

    // cast to color view
    //rgb8c_view_t left_view(const_view(left_image._dynamic_cast<rgb8_image_t>())), right_view(const_view(right_image._dynamic_cast<rgb8_image_t>()));
    rgb8c_view_t left_view(const_view(left_image)), right_view(const_view(right_image));

    // resize destination images
    gray_left_image.recreate(left_view.dimensions());
    gray_right_image.recreate(right_view.dimensions());

    // copy
    copy_and_convert_pixels(left_view, view(gray_left_image));
    copy_and_convert_pixels(right_view, view(gray_right_image));

    // cast to any_image_view
    any_image<AbstractStereoMatcher::input_images_t>::const_view_t gray_left_view(const_view(gray_left_image)), gray_right_view(const_view(gray_right_image));

    // set the gray views
    stereo_matcher_p->set_rectified_images_pair(gray_left_view, gray_right_view);
}
else

} // end of namespace doppia
