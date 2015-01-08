#ifndef VIDEOINPUTFACTORY_HPP
#define VIDEOINPUTFACTORY_HPP

#include "AbstractVideoInput.hpp"

namespace doppia
{

/// \class VideoInputFactory
/// \brief Creates an instance of a AbstractVideoInput
class VideoInputFactory
{
public:

    static boost::program_options::options_description get_args_options();
    static AbstractVideoInput* new_instance(const boost::program_options::variables_map &options);

};

} // end of namespace doppia

#endif // VIDEOINPUTFACTORY_HPP
