#include "ResidualImageFilter.hpp"

// OpenCv used for fast image filtering
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

namespace doppia {

using namespace boost;
using namespace boost::gil;

ResidualImageFilter::ResidualImageFilter()
{
    // nothing to do here
    return;
}

ResidualImageFilter::~ResidualImageFilter()
{
    // nothing to do here
    return;
}


void ResidualImageFilter::operator()(const rgb8c_view_t &src_view, const rgb8_view_t &dst_view)
{
    compute_residual_image(src_view, dst_view);
    return;
}


void ResidualImageFilter::compute_residual_image(const rgb8c_view_t &src_view, const rgb8_view_t &dst_view)
{
    // Implementation based on Toby Vaudrey code
    // http://www.cs.auckland.ac.nz/~tobi/openCV-Examples.html

    using namespace cv;
    gil::opencv::ipl_image_wrapper ipl_src = gil::opencv::create_ipl_image(src_view);
    gil::opencv::ipl_image_wrapper ipl_dst = gil::opencv::create_ipl_image(dst_view);

    Mat src(ipl_src.get()), dst(ipl_dst.get());

    const int channel_type = CV_16SC3; // CV_32FC3
    if(smoothed_src.size() != src.size())
    { // lazy allocation
        smoothed_src = Mat(src.size(), channel_type);
        residual = Mat(src.size(), channel_type);
    }

    // FIXME hardcoded value
    const Size blur_size = Size(7,7);
    //const Size blur_size = Size(9,9);

    // smooth image using mean filter
    blur(src, smoothed_src, blur_size);

    // compute Residual Image
    // (r = f - s)
    residual = src - smoothed_src;
    residual.convertTo(dst, dst.type());
    return;
}

} // end of namespace doppia
