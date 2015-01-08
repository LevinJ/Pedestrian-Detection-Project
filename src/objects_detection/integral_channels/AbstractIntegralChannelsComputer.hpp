#ifndef DOPPIA_ABSTRACTINTEGRALCHANNELSCOMPUTER_HPP
#define DOPPIA_ABSTRACTINTEGRALCHANNELSCOMPUTER_HPP

#include "AbstractChannelsComputer.hpp"

namespace doppia {

class AbstractIntegralChannelsComputer : public AbstractChannelsComputer
{

public:

    // (assuming input_channels is uint8_t)
    // uint32 will support images up to size 4x4x2000x2000 (x255)
    typedef boost::multi_array<boost::uint32_t, 3> integral_channels_t;
    typedef integral_channels_t::reference integral_channel_t;
    typedef integral_channels_t::const_reference const_integral_channel_t;

    typedef integral_channels_t::array_view<3>::type integral_channels_view_t;
    typedef integral_channels_t::const_array_view<3>::type integral_channels_const_view_t;


public:
    AbstractIntegralChannelsComputer();
    virtual ~AbstractIntegralChannelsComputer();

    //virtual int get_shrinking_factor() = 0;
    virtual const integral_channels_t &get_integral_channels() = 0;
    virtual void save_channels_to_file() = 0;
};

} //  end of namespace doppia

#endif // DOPPIA_ABSTRACTINTEGRALCHANNELSCOMPUTER_HPP
