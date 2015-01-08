#ifndef MONOKEL_MODULELOG_HPP
#define MONOKEL_MODULELOG_HPP

#include "Log.hpp"

#include <string>
#include <ostream>

namespace doppia
{

/// Helper class used to log at the module level.
/// To be used at the file level as
/// namespace { ModuleLog("module name") log; }
/// and then log.debug() << "Debug message" << std::endl;
class ModuleLog
{
public:
    ModuleLog(const std::string module_name_);
    ~ModuleLog();

    std::ostream &log_error() const;
    std::ostream &error() const;

    std::ostream &log_warning() const;
    std::ostream &warning() const;

    std::ostream &log_info() const;
    std::ostream &info() const;

    std::ostream &log_debug() const;
    std::ostream &debug() const;

protected:

    const std::string module_name;

};

/// Methods included in the header to enable inlining --

inline
std::ostream &ModuleLog::log_error() const
{
    return  logging::log(logging::ErrorMessage, module_name);
}


inline
std::ostream &ModuleLog::error() const
{
    return  logging::log(logging::ErrorMessage, module_name);
}


inline
std::ostream &ModuleLog::log_warning() const
{
    return  logging::log(logging::WarningMessage, module_name);
}


inline
std::ostream &ModuleLog::warning() const
{
    return  logging::log(logging::WarningMessage, module_name);
}


inline
std::ostream &ModuleLog::log_info() const
{
    return  logging::log(logging::InfoMessage, module_name);
}


inline
std::ostream &ModuleLog::info() const
{
    return  logging::log(logging::InfoMessage, module_name);
}


inline
std::ostream &ModuleLog::log_debug() const
{
    return  logging::log(logging::DebugMessage, module_name);
}


inline
std::ostream &ModuleLog::debug() const
{
    return  logging::log(logging::DebugMessage, module_name);
}


} //  end of namespace doppia


#define MODULE_LOG_MACRO(MODULE_NAME) \
    namespace \
    { \
    doppia::ModuleLog log((MODULE_NAME)); \
    } // end of anonymous namespace

#endif // MONOKEL_MODULELOG_HPP
