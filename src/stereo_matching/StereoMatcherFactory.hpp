#ifndef STEREOMATCHERFACTORY_HPP
#define STEREOMATCHERFACTORY_HPP

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {


// forward declaration
class AbstractVideoInput;
class AbstractStereoMatcher;

/// \class VideoInputFactory
/// \brief Creates an instance of a AbstractVideoInput
class StereoMatcherFactory
{
public:

    static boost::program_options::options_description get_args_options();
    static AbstractStereoMatcher* new_instance(const boost::program_options::variables_map &options,
                                               boost::shared_ptr<const AbstractVideoInput> video_input_p);

};


} // end of namespace doppia

#endif // STEREOMATCHERFACTORY_HPP
