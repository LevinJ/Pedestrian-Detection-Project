#ifndef DOPPIA_MODELWINDOWTOOBJECTWINDOWCONVERTERFACTORY_HPP
#define DOPPIA_MODELWINDOWTOOBJECTWINDOWCONVERTERFACTORY_HPP

#include "AbstractModelWindowToObjectWindowConverter.hpp"

#include <boost/program_options.hpp>

namespace doppia {


class ModelWindowToObjectWindowConverterFactory
{
public:

    typedef AbstractModelWindowToObjectWindowConverter::model_window_size_t model_window_size_t;
    typedef AbstractModelWindowToObjectWindowConverter::object_window_t object_window_t;

    //static boost::program_options::options_description get_args_options();

    static AbstractModelWindowToObjectWindowConverter * new_instance(const model_window_size_t&model_window_size,
                                                                     const object_window_t &object_window);
};

} // end of namespace doppia

#endif // DOPPIA_MODELWINDOWTOOBJECTWINDOWCONVERTERFACTORY_HPP
