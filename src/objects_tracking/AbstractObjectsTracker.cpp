#include "AbstractObjectsTracker.hpp"

namespace doppia {

AbstractObjectsTracker::AbstractObjectsTracker()
{
    // nothing to do here
    return;
}

AbstractObjectsTracker::~AbstractObjectsTracker()
{
    // nothing to do here
    return;
}

void AbstractObjectsTracker::set_image_size(const int width, const int height)
{
    image_width = width;
    image_height = height;

    return;
}

} // end of namespace doppia
