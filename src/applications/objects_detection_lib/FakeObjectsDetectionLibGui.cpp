#include "FakeObjectsDetectionLibGui.hpp"

namespace objects_detection {

void FakeObjectsDetectionLibGui::set_monocular_input(input_image_const_view_t &)
{
    return; // we do nothing, at all
}

void FakeObjectsDetectionLibGui::set_left_input(input_image_const_view_t &)
{
    return; // we do nothing, at all
}

void FakeObjectsDetectionLibGui::set_right_input(input_image_const_view_t &)
{
    return; // we do nothing, at all
}

void FakeObjectsDetectionLibGui::update()
{
    return; // we do nothing, at all
}

} // end of namespace objects_detection
