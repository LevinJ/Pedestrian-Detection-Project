#include "DisparityCostVolume.hpp"

#include "stereo_matching/cost_functions.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/fill_multi_array.hpp"

namespace doppia
{

using namespace boost;
using namespace boost::gil;
using namespace std;


DisparityCostVolume::DisparityCostVolume()
    : CostVolume<cost_t>(0,0,0)
{

    return;
}

DisparityCostVolume::DisparityCostVolume(const DisparityCostVolume &other_volume)
    : CostVolume<cost_t>(0,0,0)
{
    resize(other_volume);
    return;
}

DisparityCostVolume::~DisparityCostVolume()
{
    // nothing to do here
    return;
}


} // end of namespace doppia

