#include "NonMaximalSuppressionFactory.hpp"

#include "GreedyNonMaximalSuppression.hpp"
#include "FixedWindowNonMaximalSuppression.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <stdexcept>
#include <string>


namespace
{

using namespace std;

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "NonMaximalSuppressionFactory");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "NonMaximalSuppressionFactory");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "NonMaximalSuppressionFactory");
}

} // end of anonymous namespace


namespace doppia
{

using namespace std;
using boost::shared_ptr;
using namespace boost::program_options;

options_description
NonMaximalSuppressionFactory::get_args_options()
{
    options_description desc("NonMaximalSuppressionFactory options");

    desc.add_options()

            ;

    //desc.add(AbstractNonMaximalSuppression::get_args_options());
    //desc.add(FixedWindowNonMaximalSuppression::get_args_options());
    desc.add(GreedyNonMaximalSuppression::get_args_options());

    return desc;
}


AbstractNonMaximalSuppression*
NonMaximalSuppressionFactory::new_instance(const string method,
                                           const variables_map &options)
{

    AbstractNonMaximalSuppression* non_maximal_suppression_p = NULL;
    if(method.compare("greedy") == 0)
    {
        non_maximal_suppression_p = new GreedyNonMaximalSuppression(options);
    }
    else if(method.compare("fixed_window") == 0)
    {
        non_maximal_suppression_p = new FixedWindowNonMaximalSuppression();
    }
    else if (method.compare("none") == 0)
    {
        non_maximal_suppression_p = NULL;
    }
    else
    {
        printf("NonMaximalSuppressionFactory received method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown non maximal suppression method value");
    }

    return non_maximal_suppression_p;
}


} // end of namespace doppia
