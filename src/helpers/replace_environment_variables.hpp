#ifndef replace_environment_variables_H
#define replace_environment_variables_H

#include <boost/filesystem/path.hpp>
#include <string>

namespace doppia {

/// will use wordexp to replace environment variables in a path (and also the ~ sign)
std::string replace_environment_variables(const std::string &path);


boost::filesystem::path replace_environment_variables(const boost::filesystem::path &the_path);

} // end of namespace doppia

#endif // replace_environment_variables_H
