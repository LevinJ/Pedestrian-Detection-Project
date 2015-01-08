#include "replace_environment_variables.hpp"

#include <wordexp.h>
#include <sstream>

namespace doppia {

/// will use wordexp to replace environment variables in a path (and also the ~ sign)
std::string replace_environment_variables(const std::string &path)
{
    // code based on http://stackoverflow.com/questions/1902681
    std::stringstream stream;
    wordexp_t p;

    const int flags = 0;
    wordexp(path.c_str(), &p, flags);

    for (size_t i=0; i< p.we_wordc; i+=1)
    {
        stream << p.we_wordv[i];
    }

    wordfree(&p);

    return stream.str();
}


boost::filesystem::path replace_environment_variables(const boost::filesystem::path &the_path)
{
    return replace_environment_variables(the_path.string());
}

} // end of namespace doppia
