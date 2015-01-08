#ifndef FASTDISPARITYCOSTVOLUMEESTIMATOR_HPP
#define FASTDISPARITYCOSTVOLUMEESTIMATOR_HPP

#include "AbstractDisparityCostVolumeEstimator.hpp"

namespace doppia {

class FastDisparityCostVolumeEstimator: public AbstractDisparityCostVolumeEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    FastDisparityCostVolumeEstimator(const boost::program_options::variables_map &options);
    ~FastDisparityCostVolumeEstimator();


    void compute(boost::gil::gray8c_view_t &left,
                 boost::gil::gray8c_view_t &right,
                 DisparityCostVolume &cost_volume);

    void compute(boost::gil::rgb8c_view_t  &left,
                 boost::gil::rgb8c_view_t &right,
                 DisparityCostVolume &cost_volume);


protected:

    ///  Generic stereo matching using block matching
    template <typename ImgT> void compute_impl(ImgT &left, ImgT &right, DisparityCostVolume &cost_volume);

    /// ImgT is expected to be bitsetN_view_t, gray8c_view_t or rgb8c_view_t
    template <typename ImgT, typename PixelsCostType>
    void compute_costs_impl(const ImgT &left, const ImgT &right,
                            PixelsCostType &pixels_distance, DisparityCostVolume &cost_volume);

};

} // end namespace doppia

#endif // FASTDISPARITYCOSTVOLUME_HPP
