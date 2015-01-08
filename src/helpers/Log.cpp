


#include "Log.hpp"


#include <boost/thread/once.hpp>

#include <vector>


// C Standard Library headers ( for stat(2) and getpwuid() )
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>


typedef struct stat struct_stat;


std::string
       logging::current_posix_time_string()
{
    char time_string[2048];
    time_t t = time(0);
    struct tm* time_struct = localtime(&t);
    strftime(time_string, 2048, "%F %T", time_struct);
    return std::string(time_string);
}

// ---------------------------------------------------
// Create a single instance of the SystemLog
// ---------------------------------------------------
namespace logging {
logging::null_ostream g_null_ostream;
}
namespace {
    static boost::once_flag call_once_flag = BOOST_ONCE_INIT;
    boost::shared_ptr<logging::Log> system_log_ptr;
    void init_system_log() {
        system_log_ptr = boost::shared_ptr<logging::Log>(new logging::Log());
    }
}

// ---------------------------------------------------
// Basic stream support
// ---------------------------------------------------
std::ostream& logging::log( int log_level, std::string log_namespace ) {
    return get_log()(log_level, log_namespace);
}


// ---------------------------------------------------
// LogInstance Methods
// ---------------------------------------------------
logging::LogInstance::LogInstance(std::string log_filename, bool prepend_infostamp) : m_prepend_infostamp(prepend_infostamp) {
    // Open file and place the insertion pointer at the end of the file (ios_base::ate)
    m_log_ostream_ptr = new std::ofstream(log_filename.c_str(), std::ios::app);
    if (not  static_cast<std::ofstream*>(m_log_ostream_ptr)->is_open())
    {
        std::cerr << "Could not open log file " << log_filename << " for writing." << std::endl;
        throw std::runtime_error("Could not open log file for writing.");
    }

    *m_log_ostream_ptr << "\n\n" << "Vision Workbench log started at " << current_posix_time_string() << ".\n\n";

    m_log_stream.set_stream(*m_log_ostream_ptr);
    return;
}

logging::LogInstance::LogInstance(std::ostream& log_ostream, bool prepend_infostamp) : m_log_stream(log_ostream),
m_log_ostream_ptr(NULL),
m_prepend_infostamp(prepend_infostamp) {
    // nothing to do here
    return;
}

std::ostream& logging::LogInstance::operator() (const int log_level, const std::string log_namespace) {
    if (m_rule_set(log_level, log_namespace)) {
        if (m_prepend_infostamp)
        {
            m_log_stream << current_posix_time_string() << " {" << boost::this_thread::get_id() << "} [ " << log_namespace << " ] : ";
        }
        switch (log_level) {
        case ErrorMessage:   m_log_stream << "Error: ";   break;
        case WarningMessage: m_log_stream << "Warning: "; break;
        default: break;
        }
        return m_log_stream;
    } else {
        return g_null_ostream;
    }
}


std::ostream& logging::Log::operator() (int log_level, std::string log_namespace) {

    boost::mutex::scoped_lock multi_ostreams_lock(m_multi_ostreams_mutex);

    // Check to see if we have an ostream defined yet for this thread.
    if(m_multi_ostreams.find( boost::this_thread::get_id() ) == m_multi_ostreams.end())
    {
        m_multi_ostreams[ boost::this_thread::get_id() ] = boost::shared_ptr<multi_ostream>(new multi_ostream);
    }

    // Reset and add the console log output...
    m_multi_ostreams[ boost::this_thread::get_id() ]->clear();
    m_multi_ostreams[ boost::this_thread::get_id() ]->add(m_console_log->operator()(log_level, log_namespace));

    // ... and the rest of the active log streams.
    std::vector<boost::shared_ptr<LogInstance> >::iterator iter = m_logs.begin();
    for (;iter != m_logs.end(); ++iter)
    {
        m_multi_ostreams[ boost::this_thread::get_id() ]->add((*iter)->operator()(log_level,log_namespace));
    }

    return *m_multi_ostreams[ boost::this_thread::get_id() ];
}

logging::Log& logging::get_log() {
    boost::call_once( init_system_log, call_once_flag);
    return *system_log_ptr;
}
