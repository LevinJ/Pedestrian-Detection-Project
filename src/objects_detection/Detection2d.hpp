#ifndef DETECTION2D_HPP
#define DETECTION2D_HPP

#include "helpers/geometry.hpp"

#include <boost/cstdint.hpp>

#if defined(TESTING)
#include <vector>
#endif

namespace doppia {

class Detection2d
{
public:
    Detection2d();
    ~Detection2d();

    /// coordinate_t must allow negative positions (border of image, etc...)
    typedef boost::int16_t coordinate_t;
    typedef doppia::geometry::point_xy<coordinate_t> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;


    /// class of the detected objects
    enum ObjectClasses { Car, Pedestrian, Bike, Motorbike, Bus, Tram, StaticObject, Unknown };

    ObjectClasses object_class;

    rectangle_t bounding_box;

    /// Detector score, the higher the score the higher the detection confidence
    /// Score should be normalized across classes (important for non-maximal suppression)
    float score;

#if defined(TESTING)
    std::vector<float> score_trajectory;
    size_t detector_index;
#endif

};

} // end of namespace doppia

#endif // DETECTION2D_HPP
