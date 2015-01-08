#ifndef DISPARITYCOSTVOLUMEESTIMATORFACTORY_HPP
#define DISPARITYCOSTVOLUMEESTIMATORFACTORY_HPP


#include <boost/program_options.hpp>

namespace doppia {

// forward declaration
class AbstractDisparityCostVolumeEstimator;

class DisparityCostVolumeEstimatorFactory
{
public:

    static boost::program_options::options_description get_args_options();
    static AbstractDisparityCostVolumeEstimator*
    new_instance(const boost::program_options::variables_map &options);

};

} // end of namespace doppia

#endif // DISPARITYCOSTVOLUMEESTIMATORFACTORY_HPP
