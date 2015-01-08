#ifndef DOPPIA_PROGRESS_DISPLAY_WITH_ETA_HPP
#define DOPPIA_PROGRESS_DISPLAY_WITH_ETA_HPP


#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <boost/timer.hpp>
#include <boost/utility.hpp>  // for noncopyable
#include <boost/cstdint.hpp>  // for uintmax_t
#include <iostream>           // for ostream, cout, etc
#include <string>             // for string

namespace doppia {

class progress_display : private boost::noncopyable
{

public:
    explicit progress_display( unsigned long expected_count,
                               std::ostream & os = std::cout,
                               const std::string & s1 = "\n", //leading strings
                               const std::string & s2 = "",
                               const std::string & s3 = "" );

    virtual ~progress_display();

    void restart( unsigned long expected_count );

    unsigned long  operator+=( unsigned long increment );
    unsigned long  operator++();
    unsigned long  count() const;
    unsigned long  expected_count() const;

protected:
    std::ostream &     m_os;  //< may not be present in all imps
    const std::string  m_s1;  //< string is more general, safer than
    const std::string  m_s2;  //<  const char *, and efficiency or size are
    const std::string  m_s3;  //<  not issues

    unsigned long _count, _expected_count, _next_tic_count;
    unsigned int  _tic;

    virtual void display_tic();
};


/// Progress display with expected time of arrival
/// based on (now deprecated) boost::progress_display
class progress_display_with_eta : public progress_display
{

public:
    explicit progress_display_with_eta( unsigned long expected_count,
                               std::ostream & os = std::cout,
                               const std::string & s1 = "\n", //leading strings
                               const std::string & s2 = "",
                               const std::string & s3 = "" );

    void restart( unsigned long expected_count );

protected:

    boost::posix_time::ptime start_time;
    std::string eta_text;

    void display_tic();
};




} // end of namespace doppia

#endif // DOPPIA_PROGRESS_DISPLAY_WITH_ETA_HPP
