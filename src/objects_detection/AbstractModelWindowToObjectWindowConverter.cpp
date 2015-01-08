#include "AbstractModelWindowToObjectWindowConverter.hpp"

#include <stdexcept>

namespace doppia {

AbstractModelWindowToObjectWindowConverter::AbstractModelWindowToObjectWindowConverter()
{
    // nothing to do here
    return;
}

AbstractModelWindowToObjectWindowConverter::~AbstractModelWindowToObjectWindowConverter()
{
    // nothing to do here
    return;
}


void AbstractModelWindowToObjectWindowConverter::operator ()(detections_t &/*detections*/) const
{
    // default implementation does nothing
    // (we do have a default implementation so that we can pass AbstractModelWindowToObjectWindowConverter by copy,
    // instead of via pointers)
    throw std::runtime_error("AbstractModelWindowToObjectWindowConverter::operator () should never be called");
    return;
}

} // end of namespace doppia
