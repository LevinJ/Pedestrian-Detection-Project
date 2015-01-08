#ifndef NONMAXIMALSUPPRESSIONFACTORY_HPP
#define NONMAXIMALSUPPRESSIONFACTORY_HPP

#include <boost/program_options.hpp>
#include <string>

namespace doppia {

// forward declaration
class AbstractNonMaximalSuppression;

class NonMaximalSuppressionFactory
{
public:
    static boost::program_options::options_description get_args_options();

    // Since we may instanciate multiple non maximal suppresion modules, simply giving the program options
    // is not enough, the actual non_maximal_suppression method must be indicated explicitly
    static AbstractNonMaximalSuppression* new_instance(const std::string method_name,
                                                       const boost::program_options::variables_map &options);
};


} // end of namespace doppia

#endif // NONMAXIMALSUPPRESSIONFACTORY_HPP
