#ifndef RESIDUALIMAGEFILTER_HPP
#define RESIDUALIMAGEFILTER_HPP

#include <boost/gil/typedefs.hpp>

#include <opencv2/core/core.hpp>

namespace doppia {

class ResidualImageFilter
{
public:
    ResidualImageFilter();
    ~ResidualImageFilter();

    void operator()(const boost::gil::rgb8c_view_t &src_view,
                    const boost::gil::rgb8_view_t &dst_view);

protected:

    void compute_residual_image(const boost::gil::rgb8c_view_t &src_view,
                                const boost::gil::rgb8_view_t &dst_view);

    cv::Mat smoothed_src, residual;

};

} // end of namespace doppia

#endif // RESIDUALIMAGEFILTER_HPP
