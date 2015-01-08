#ifndef ABSTRACTDISPARITYCOSTVOLUMEESTIMATOR_HPP
#define ABSTRACTDISPARITYCOSTVOLUMEESTIMATOR_HPP

#include <boost/program_options.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view.hpp>

#include <string>

namespace doppia {

// forward declaration
class DisparityCostVolume;

class AbstractDisparityCostVolumeEstimator
{

protected:
    /// helper constructor for unit testing
    /// child class can create non-initializing constructors
    AbstractDisparityCostVolumeEstimator();

public:
    static boost::program_options::options_description get_args_options();

    AbstractDisparityCostVolumeEstimator(const boost::program_options::variables_map &options);
    virtual ~AbstractDisparityCostVolumeEstimator();

    /// get the maximum cost computed
    float get_maximum_cost_per_pixel() const;

    typedef boost::gil::gray8c_view_t::point_t point_t;
    /// do memory allocation
    void resize_cost_volume(const point_t &input_dimensions, DisparityCostVolume &cost_volume) const;

    virtual void compute(boost::gil::gray8c_view_t &left,
                         boost::gil::gray8c_view_t &right,
                         DisparityCostVolume &cost_volume) = 0;

    virtual void compute(boost::gil::rgb8c_view_t  &left,
                         boost::gil::rgb8c_view_t &right,
                         DisparityCostVolume &cost_volume) = 0;

protected:
    bool first_computation;
    size_t max_disparity;
    float maximum_cost_per_pixel;
    std::string pixels_matching_method;

};

} // end of namespace doppia

#endif // ABSTRACTDISPARITYCOSTVOLUMEESTIMATOR_HPP
