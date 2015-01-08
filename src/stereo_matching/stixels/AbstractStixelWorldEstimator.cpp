#include "AbstractStixelWorldEstimator.hpp"

namespace doppia
{

    // provide default implementations for some methods
    void AbstractStixelWorldEstimator::set_rectified_images_pair(const input_image_const_view_t &left,
                                                                 const input_image_const_view_t &right)
    {
        input_left_view = left;
        input_right_view = right;
        return;
    }


    AbstractStixelWorldEstimator::~AbstractStixelWorldEstimator()
    {
        // nothing to do here
        return;
    }

} // end of namespace doppia
