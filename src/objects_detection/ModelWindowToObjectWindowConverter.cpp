#include "ModelWindowToObjectWindowConverter.hpp"

#include <boost/foreach.hpp>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include <iostream>

namespace doppia
{



ModelWindowToObjectWindowConverter::ModelWindowToObjectWindowConverter(
        const model_window_size_t &model_window_size,
        const object_window_t &object_window)
{

    const float
            object_width = object_window.max_corner().x() - object_window.min_corner().x(),
            object_height = object_window.max_corner().y() - object_window.min_corner().y(),
            model_width = model_window_size.x(),
            model_height = model_window_size.y(),
            object_center_x = (object_window.max_corner().x() + object_window.min_corner().x())/2.0f,
            object_center_y = (object_window.max_corner().y() + object_window.min_corner().y())/2.0f,
            model_center_x = model_width / 2.0f,
            model_center_y = model_height / 2.0f,
            delta_x = object_center_x - model_center_x,
            delta_y = object_center_y - model_center_y;
    if (true){

        std::cout << "object width: "<< object_width << " object height: " << object_height << std::endl;
        std::cout << "model width: "<< model_width << " model height: " << model_height << std::endl;



    }

    // object_width / model_width
    object_width_to_model_width = static_cast<float>(object_width) / model_width;
    object_height_to_model_height = static_cast<float>(object_height) / model_height;

    // delta_x_to_model_width = (object_center_x - model_center_x) / model_width
    delta_x_to_model_width =  delta_x / model_width;
    delta_y_to_model_height = delta_y / model_height;

    return;
}


ModelWindowToObjectWindowConverter::~ModelWindowToObjectWindowConverter()
{
    // nothing to do here
    return;
}


/// will convert the input detection from model_window to object_window
void ModelWindowToObjectWindowConverter::operator()(detections_t &detections) const
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
                model_width = box.max_corner().x() - box.min_corner().x(),
                model_height = box.max_corner().y() - box.min_corner().y(),
                object_width = object_width_to_model_width * model_width,
                object_height = object_height_to_model_height * model_height,
                half_object_width = object_width / 2.0f,
                half_object_height = object_height / 2.0f,
                model_center_x = (box.max_corner().x() + box.min_corner().x()) / 2.0f,
                model_center_y = (box.max_corner().y() + box.min_corner().y()) / 2.0f,
                delta_x = delta_x_to_model_width * model_width,
                delta_y = delta_y_to_model_height * model_height,
                object_center_x = model_center_x + delta_x,
                object_center_y = model_center_y + delta_y;

        box.min_corner().x(object_center_x - half_object_width);
        box.min_corner().y(object_center_y - half_object_height);
        box.max_corner().x(object_center_x + half_object_width);
        box.max_corner().y(object_center_y + half_object_height);
    } // end of "for each detection"

    return;
}


} // end of namespace doppia
