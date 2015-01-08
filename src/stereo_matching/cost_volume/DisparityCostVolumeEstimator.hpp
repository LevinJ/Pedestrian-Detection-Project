#ifndef DISPARITYCOSTVOLUMEESTIMATOR_HPP
#define DISPARITYCOSTVOLUMEESTIMATOR_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/cstdint.hpp>

#include "AbstractDisparityCostVolumeEstimator.hpp"

namespace doppia
{

class GradientTransform; // forward declaration

class DisparityCostVolumeEstimator: public AbstractDisparityCostVolumeEstimator
{

protected:
    /// helper constructor for unit testing
    /// child class can create non-initializing constructors
    DisparityCostVolumeEstimator();

public:

    static boost::program_options::options_description get_args_options();

    DisparityCostVolumeEstimator(const boost::program_options::variables_map &options);
    virtual ~DisparityCostVolumeEstimator();

    virtual void compute(boost::gil::gray8c_view_t &left,
                         boost::gil::gray8c_view_t &right,
                         DisparityCostVolume &cost_volume);

    virtual void compute(boost::gil::rgb8c_view_t  &left,
                         boost::gil::rgb8c_view_t &right,
                         DisparityCostVolume &cost_volume);

protected:

    float threshold_percent;

    ///  Generic stereo matching using block matching
    template <typename ImgT> void compute_impl(ImgT &left, ImgT &right, DisparityCostVolume &cost_volume);


    /// ImgT is expected to be bitsetN_view_t, gray8c_view_t, rgb8c_view_t, dev3n8c_view_t or dev5n8c_view_t
    template <typename ImgT, typename PixelsCostType>
    void compute_costs_impl(const ImgT &left, const ImgT &right,
                            PixelsCostType &pixels_distance, DisparityCostVolume &cost_volume);


};

} // end of namespace doppia



#endif // DISPARITYCOSTVOLUMEESTIMATOR_HPP
