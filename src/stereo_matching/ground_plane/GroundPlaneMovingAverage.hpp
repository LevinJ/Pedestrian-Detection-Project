#ifndef GROUNDPLANEMOVINGAVERAGE_HPP
#define GROUNDPLANEMOVINGAVERAGE_HPP

#include "GroundPlane.hpp"

#include <boost/program_options.hpp>
#include <boost/circular_buffer.hpp>

namespace doppia {

/// Very simplistic temporal filter for the ground plane estimate
class GroundPlaneMovingAverage
{
public:
    static boost::program_options::options_description get_args_options();

    GroundPlaneMovingAverage(const boost::program_options::variables_map &options);
    ~GroundPlaneMovingAverage();

    void add_estimate(const GroundPlane &, const float weight);
    const GroundPlane &get_current_estimate();

protected:

    typedef Eigen::aligned_allocator<GroundPlane> ground_plane_allocator_t;
    typedef boost::circular_buffer<GroundPlane, ground_plane_allocator_t> buffer_t;
    buffer_t buffer;
    typedef boost::circular_buffer<float> weights_buffer_t;
    weights_buffer_t weights_buffer;
    GroundPlane current_estimate;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

} // end of namespace doppia

#endif // GROUNDPLANEMOVINGAVERAGE_HPP
