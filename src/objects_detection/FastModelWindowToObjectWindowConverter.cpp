#include "FastModelWindowToObjectWindowConverter.hpp"

#include <boost/foreach.hpp>

#include <cstdio>
#include <stdexcept>

namespace doppia {

FastModelWindowToObjectWindowConverter::FastModelWindowToObjectWindowConverter()
{
    // by default we keep the detection windows as they are
    scaling_factor_x = 1.0f;
    scaling_factor_y = 1.0f;
    return;
}


FastModelWindowToObjectWindowConverter::FastModelWindowToObjectWindowConverter(
        const model_window_size_t &model_window_size,
        const object_window_t &object_window)
{

    const int
            object_width = object_window.max_corner().x() - object_window.min_corner().x(),
            object_height = object_window.max_corner().y() - object_window.min_corner().y(),
            object_center_x = (object_window.max_corner().x() + object_window.min_corner().x())/2,
            object_center_y = (object_window.max_corner().y() + object_window.min_corner().y())/2;

    if(false)
    {
        printf("object_center x,y == %i, %i\n", object_center_x, object_center_y);
        printf("model_window_size x,y == %i, %i\n", model_window_size.x(), model_window_size.y());
        printf("binHIer");
    }

    // we give a 1 pixel slack to handle impair numbers
    if((std::abs(object_center_x - static_cast<int>(model_window_size.x()/2)) > 1) or
       (std::abs(object_center_y - static_cast<int>(model_window_size.y()/2)) > 1))
    {
        printf("object_center x,y == %i, %i\n", object_center_x, object_center_y);
        printf("model_window_size x,y == %i, %i\n", model_window_size.x(), model_window_size.y());
        throw std::invalid_argument("FastModelWindowToObjectWindowConverter expects "
                                    "the object_window to be centered in the model window. "
                                    "Handling asymmetric configurations is not yet implemented");
    }

    scaling_factor_x = static_cast<float>(object_width) / model_window_size.x();
    scaling_factor_y = static_cast<float>(object_height) / model_window_size.y();
   // scaling_factor_x = 1.0f;
   //scaling_factor_y = 1.0f;


    return;
}


FastModelWindowToObjectWindowConverter::~FastModelWindowToObjectWindowConverter()
{
    // nothing to do here
    return;
}


/// will convert the input detection from model_window to object_window
void FastModelWindowToObjectWindowConverter::operator()(detections_t &detections) const
{
    BOOST_FOREACH(detection_t &detection, detections)
    {
        if(detection.object_class != detection_t::Pedestrian)
        {
            continue;
        }
        // pedestrian detection
        detection_t::rectangle_t &box = detection.bounding_box;
        const float
                half_delta_x = (box.max_corner().x() - box.min_corner().x())/2.0f,
                half_delta_y = (box.max_corner().y() - box.min_corner().y())/2.0f,
                center_x = box.min_corner().x() + half_delta_x,
                center_y = box.min_corner().y() + half_delta_y,
                // alpha, beta, gamma, delta, epsilon, zeta, eta...
                half_epsilon_x = half_delta_x*scaling_factor_x,
                half_epsilon_y = half_delta_y*scaling_factor_y;

        box.min_corner().x(center_x - half_epsilon_x);
        box.min_corner().y(center_y - half_epsilon_y);
        box.max_corner().x(center_x + half_epsilon_x);
        box.max_corner().y(center_y + half_epsilon_y);
    } // end of "for each detection"

    return;
}
} // namespace doppia
