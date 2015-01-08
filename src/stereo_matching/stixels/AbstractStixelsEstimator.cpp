#include "AbstractStixelsEstimator.hpp"

namespace doppia {

AbstractStixelsEstimator::AbstractStixelsEstimator()
{
    // nothing to do here
    return;
}

AbstractStixelsEstimator::~AbstractStixelsEstimator()
{
    // nothing to do here
    return;
}

const stixels_t &AbstractStixelsEstimator::get_stixels() const
{
    return the_stixels;
}


} // end of namespace doppia
