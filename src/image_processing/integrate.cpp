#include "integrate.hpp"

namespace doppia {

// this is a templated method, fully implemented in the header file


// Compile specific instances of the templated method
// (needed in other parts of the code, see for instance test_objects_detection.cpp )
// FIXME why this did not work as desired ?
/*template <>
void integrate< boost::multi_array<boost::uint8_t, 2>,
boost::multi_array<boost::uint32_t, 2> >
(const boost::multi_array<boost::uint8_t, 2> &,
 boost::multi_array<boost::uint32_t, 2> &);

template <>
void integrate<IntegralChannelsForPedestrians::channels_t::reference,
IntegralChannelsForPedestrians::integral_channels_t::reference>
(const IntegralChannelsForPedestrians::channels_t::reference &,
 IntegralChannelsForPedestrians::integral_channels_t::reference &);

template <>
void integrate< boost::multi_array<boost::uint8_t, 3>::reference,
IntegralChannelsForPedestrians::integral_channels_t::reference>
(const boost::multi_array<boost::uint8_t, 3>::reference &,
 IntegralChannelsForPedestrians::integral_channels_t::reference &);
*/



} // end of namespace doppia
