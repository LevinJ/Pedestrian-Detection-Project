#ifndef DOPPIA_INTEGRALCHANNELSCOMPUTERFACTORY_HPP
#define DOPPIA_INTEGRALCHANNELSCOMPUTERFACTORY_HPP

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace doppia {

// forward declaration
class AbstractIntegralChannelsComputer;

class IntegralChannelsComputerFactory
{
public:
    static boost::program_options::options_description get_options_description();
    static AbstractIntegralChannelsComputer* new_instance(const boost::program_options::variables_map &options,
                                                          const std::string &method = "hog6_luv");
};

} // end of namespace doppia

#endif // DOPPIA_INTEGRALCHANNELSCOMPUTERFACTORY_HPP
