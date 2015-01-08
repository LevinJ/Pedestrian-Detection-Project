#ifndef GRAYSTEREOMATCHER_HPP
#define GRAYSTEREOMATCHER_HPP

#include "AbstractStereoMatcher.hpp"

namespace doppia {

/// Adapter class that transforms color
/// images to gray before calling a gray only stereo matcher
class GrayStereoMatcher : public AbstractStereoMatcher
{
public:
    GrayStereoMatcher();
    ~GrayStereoMatcher();

    void set_rectified_images_pair(input_image_view_t &left, input_image_view_t &right);


};

} // end of namespace doppia

#endif // GRAYSTEREOMATCHER_HPP
