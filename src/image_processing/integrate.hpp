#ifndef INTEGRATE_HPP
#define INTEGRATE_HPP

#include <algorithm>
#include <stdexcept>

/// Helper function that integrates an image
/// We assume the inputs are 2d multi_array views
/// code based on http://en.wikipedia.org/wiki/Summed_area_table
template<typename ChannelView, typename IntegralChannelView>
void integrate(const ChannelView &channel, IntegralChannelView &integral_channel);

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

namespace doppia {
// Templated function integrated in the header because I could not make the .cpp instanciation work
// FIXME should move this to a separate helper file ?

/// Helper function that integrates an image
/// We assume the inputs are 2d multi_array views
/// code based on http://en.wikipedia.org/wiki/Summed_area_table
template<typename ChannelView, typename IntegralChannelView>
void integrate(const ChannelView &channel, IntegralChannelView &integral_channel)
{

    if((channel.num_dimensions() != 2) or (integral_channel.num_dimensions() != 2))
    {
        throw std::invalid_argument("integral(...) expected to receive 2 dimensional multi_array views");
    }

    if(((channel.shape()[0] + 1) != integral_channel.shape()[0])
            or ((channel.shape()[1] + 1) != integral_channel.shape()[1]))
    {
        throw std::invalid_argument("integral(...) received views of incompatible size, "
                                    "integral_channel should of size channel + 1 (on both dimensions)");
    }

    // first row of the integral_channel, is set to zero
    {
        typename IntegralChannelView::reference integral_channel_row = integral_channel[0];

        std::fill(integral_channel_row.begin(), integral_channel_row.end(), 0);
    }

    // we count rows in the integral_channel, they are shifted by one pixel from the image rows
    for(size_t row=1; row < integral_channel.shape()[0]; row+=1)
    {
        typename IntegralChannelView::reference
                integral_channel_previous_row = integral_channel[row -1],
                integral_channel_row = integral_channel[row];
        typename ChannelView::const_reference channel_row = channel[row - 1];

        integral_channel_row[0] = 0;

        // integral_channel_row.size() == (channel_row.size() + 1) so everything is fine
        for(size_t col=0; col < channel_row.size(); col+=1)
        {
            integral_channel_row[col+1] =
                    channel_row[col] +
                    integral_channel_row[col] + integral_channel_previous_row[col+1]
                    - integral_channel_previous_row[col];
        } // end of "for each column in the input channel"

    } // end of "for each row in the input channel"

    return;
}



} // end of namespace doppia

#endif // INTEGRATE_HPP
