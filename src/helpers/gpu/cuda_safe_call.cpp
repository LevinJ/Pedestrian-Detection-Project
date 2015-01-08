#include "cuda_safe_call.hpp"

#include <stdexcept>
#include <boost/format.hpp>
#include <string>
#include <iostream>

namespace doppia {

void cuda_error(const char *error_string, const char *file, const int line, const char *func)
{
    //const int code = CV_GpuApiCallError;

    if (std::uncaught_exception())
    {
        //const char* errorStr = cvErrorStr(code);
        const char* errorStr = "CUDA GPU API call error";
        const char* function = func ? func : "unknown function";

        std::cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
        std::cerr.flush();
    }
    else
    {
        std::string error_message =
                boost::str(boost::format("Cuda error (\"%s\") when calling function '%s' in file %s, line %i")
                           % error_string % func % file % line);

       throw std::runtime_error(error_message.c_str());
    }

    return;
}


} // end of namespace doppia
