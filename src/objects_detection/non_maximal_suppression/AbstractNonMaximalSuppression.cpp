#include "AbstractNonMaximalSuppression.hpp"

namespace doppia {

AbstractNonMaximalSuppression::AbstractNonMaximalSuppression()
{
    return;
}

AbstractNonMaximalSuppression::~AbstractNonMaximalSuppression()
{
    // nothing to do here
    return;
}


const AbstractNonMaximalSuppression::detections_t & AbstractNonMaximalSuppression::get_detections()
{
    return maximal_detections;
}


} // end of namespace doppia
