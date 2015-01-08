#include "GroundPlaneMovingAverage.hpp"

#include "helpers/get_option_value.hpp"

#include "boost/foreach.hpp"
#include <stdexcept>

namespace doppia {

using namespace std;
using namespace boost;

program_options::options_description GroundPlaneMovingAverage::get_args_options()
{
    program_options::options_description desc("GroundPlaneMovingAverage options");

    desc.add_options()

            ("ground_plane_estimator.moving_average_length",
             boost::program_options::value<int>()->default_value(10),
             "how many frames are used in the moving average")

            ;

    return desc;

}


GroundPlaneMovingAverage::GroundPlaneMovingAverage(
     const boost::program_options::variables_map &options)
{
    const int buffer_length =
            get_option_value<int>(options, "ground_plane_estimator.moving_average_length");

    if(buffer_length <= 0)
    {
        throw std::invalid_argument("ground_plane_estimator.moving_average_length must be > 0");
    }

    buffer.set_capacity(buffer_length);
    weights_buffer.set_capacity(buffer_length);
    return;
}


GroundPlaneMovingAverage::~GroundPlaneMovingAverage()
{
    // nothing to do here
    return;
}


void GroundPlaneMovingAverage::add_estimate(const GroundPlane &estimate, const float weight)
{
    buffer.push_back(estimate);
    weights_buffer.push_back(weight);

    float height_average = 0, pitch_average = 0;
    float weights_sum = 0;
    //BOOST_FOREACH(const GroundPlane &ground_plane, buffer)
    weights_buffer_t::const_iterator weights_it = weights_buffer.begin();
    buffer_t::const_iterator buffer_it = buffer.begin();
    for(; buffer_it != buffer.end() and weights_it != weights_buffer.end();
        ++buffer_it, ++weights_it)
    {
        const GroundPlane &ground_plane = *buffer_it;
        const float weight = *weights_it;
        height_average += weight*ground_plane.get_height();
        pitch_average += weight*ground_plane.get_pitch();
        weights_sum += weight;
    }

    //const int buffer_size = buffer.size();
    //buffer_size is be > 0
    assert(weights_sum > 0);
    height_average /= weights_sum;
    pitch_average /= weights_sum;

    current_estimate.set_from_metric_units(pitch_average, 0, height_average);
    return;
}

const GroundPlane &GroundPlaneMovingAverage::get_current_estimate()
{
    return current_estimate;
}


} // end of namespace doppia
