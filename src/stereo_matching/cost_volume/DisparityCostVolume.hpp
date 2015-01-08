#ifndef DISPARITYCOSTVOLUME_HPP
#define DISPARITYCOSTVOLUME_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/cstdint.hpp>

#include "CostVolume.hpp"

namespace doppia
{

/// Container class used to store a
/// disparity cost volume, resulting from a stereo matching process

/// data is organized as y (rows), x (columns), disparity
//class DisparityCostVolume: public CostVolume<float>
//class DisparityCostVolume: public CostVolume<boost::int32_t>
//class DisparityCostVolume: public CostVolume<boost::int16_t>
class DisparityCostVolume: public CostVolume<boost::uint8_t>
{

public:

    DisparityCostVolume();
    DisparityCostVolume(const DisparityCostVolume &other_volume);
    virtual ~DisparityCostVolume();

};

} // end of namespace doppia



#endif // DISPARITYCOSTVOLUME_HPP
