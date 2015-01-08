#ifndef STIXEL_HPP
#define STIXEL_HPP

#include <vector>

namespace doppia {


class Stixel
{
public:

    /// width of the stixel, in pixels
    int width;

    /// center position of the stixel, in pixels
    int x;

    /// top and bottom position, in pixels
    int bottom_y, top_y;

    /// was the height estimated using a default value or actual data ?
    bool default_height_value;

    /// depth corresponding to the nearest point covered by the stixel, in pixels
    int disparity;

    /// class of the object covered by the stixel
    enum Types { Unknown, Occluded, Car, Pedestrian, StaticObject } type;

    /// horizontal displacement starting from the center of stixel and ending at the center of the matched stixel
    /// "backward" indicates that _every_ stixel in the current frame has a matched stixel in the previous frame.
    /// valid_delta_x indicates whether the delta_x assignment is valid or invalid.
    /// valid_delta_x == false means :
    /// -- either the stixel in the current frame has type Occluded
    /// -- or there can not be found a match in the previous frame for the stixel in the current frame
    /// -- or the stixel in previous frame to which the stixel in the current frame is matched to, has type Occluded
    int backward_delta_x;
    bool valid_backward_delta_x;

    /// number of stixels in the previous frame covered by the stixel in the current frame
    /// the count of the width is done from left to right with respect to the stixel
    /// pointed by backward_delta_x
    int backward_width;

};

typedef std::vector<Stixel> stixels_t;


} // end of namespace doppia

#endif // STIXEL_HPP
