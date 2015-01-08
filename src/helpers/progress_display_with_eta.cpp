#include "progress_display_with_eta.hpp"

#include <boost/date_time/posix_time/posix_time.hpp> // also input/output functions

namespace doppia {


progress_display::progress_display( unsigned long expected_count,
                                    std::ostream & os,
                                    const std::string & s1,
                                    const std::string & s2,
                                    const std::string & s3)
// os is hint; implementation may ignore, particularly in embedded systems
    : m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
{
    restart(expected_count);
    return;
}


progress_display::~progress_display()
{
    // nothing to do here
    return;
}


void progress_display::restart( unsigned long expected_count )
//  Effects: display appropriate scale
//  Postconditions: count()==0, expected_count()==expected_count
{
    _count = _next_tic_count = _tic = 0;
    _expected_count = expected_count;

    m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
         << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
         << std::endl  // endl implies flush, which ensures display
         << m_s3;
    if ( !_expected_count )
    {
        _expected_count = 1;  // prevent divide by zero
    }

    return;
} // restart


unsigned long  progress_display::operator+=( unsigned long increment )
//  Effects: Display appropriate progress tic if needed.
//  Postconditions: count()== original count() + increment
//  Returns: count().
{
    if ( (_count += increment) >= _next_tic_count )
    {
        display_tic();
    }
    return _count;
}


unsigned long  progress_display::operator++()
{
    return operator+=( 1 );
}


unsigned long  progress_display::count() const
{
    return _count;
}


unsigned long  progress_display::expected_count() const
{
    return _expected_count;
}


void progress_display::display_tic()
{
    // use of floating point ensures that both large and small counts
    // work correctly.  static_cast<>() is also used several places
    // to suppress spurious compiler warnings.
    const unsigned int tics_needed =
            static_cast<unsigned int>(
                (static_cast<double>(_count)/_expected_count)*50.0 );
    do {
        m_os << '*' << std::flush;
    } while ( ++_tic < tics_needed );

    _next_tic_count =
            static_cast<unsigned long>((_tic/50.0)*_expected_count);

    if ( _count == _expected_count )
    {
        if ( _tic < 51 )
        {
            m_os << '*';
        }

        m_os << std::endl;
    }
    return;
}



// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

progress_display_with_eta::progress_display_with_eta(
        unsigned long expected_count,
        std::ostream & os,
        const std::string & s1,
        const std::string & s2,
        const std::string & s3)
// os is hint; implementation may ignore, particularly in embedded systems
    : progress_display(expected_count, os, s1, s2, s3)
{
    // need to set here, since progress_display will call "its own restart method"
    start_time = boost::posix_time::second_clock::universal_time();
    return;
}


void progress_display_with_eta::restart( unsigned long expected_count )
//  Effects: display appropriate scale
//  Postconditions: count()==0, expected_count()==expected_count
{
    start_time = boost::posix_time::second_clock::universal_time();
    progress_display::restart(expected_count);
    return;
} // restart


void progress_display_with_eta::display_tic()
{
    // we erase the previous eta text
    for(size_t c=0; c < eta_text.size(); c+=1)
    {
        m_os << "\b \b";
    }

    // use of floating point ensures that both large and small counts
    // work correctly.  static_cast<>() is also used several places
    // to suppress spurious compiler warnings.
    const unsigned int tics_needed =
            static_cast<unsigned int>(
                (static_cast<double>(_count)/_expected_count)*50.0 );
    do {
        m_os << '*';
    } while ( ++_tic < tics_needed );

    using namespace boost::posix_time;
    const ptime current_time(second_clock::universal_time());
    const time_duration
            delta_time = current_time - start_time,
            expected_time = delta_time * static_cast<double>(_expected_count - _count) / _count;
    eta_text = " (" + boost::posix_time::to_simple_string(expected_time) + " more)";

    if ( _count == _expected_count )
    {
        if ( _tic < 51 )
        {
            m_os << '*';
        }

        const std::string final_time = " Took " + boost::posix_time::to_simple_string(delta_time);
        m_os << final_time;
        m_os << std::endl;
    }
    else
    {
        m_os << eta_text << std::flush;
    }

    _next_tic_count = static_cast<unsigned long>((_tic/50.0)*_expected_count);

    return;
}

} // namespace doppia
