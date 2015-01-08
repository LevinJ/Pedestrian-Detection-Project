#include "AddBorderFunctor.hpp"

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace doppia {

using namespace boost::gil;
using namespace boost::gil::opencv;


AddBorderFunctor::AddBorderFunctor(const int additional_border_)
    : additional_border(additional_border_)
{
    // nothing to do here
    return;
}


AddBorderFunctor::~AddBorderFunctor()
{
    // nothing to do here
    return;
}


AddBorderFunctor::image_view_t AddBorderFunctor::operator()(image_view_t &input_image_view)
{
    if(additional_border <= 0)
    {
        // nothing to do
        return input_image_view;
    }

    ipl_image_wrapper image_with_border_ipl = create_ipl_image(input_image_view);
    const bool copy_data = true;
    cv::Mat input_image_mat(image_with_border_ipl.get(), copy_data);

    image_with_border.recreate(input_image_view.width()+2*additional_border,
                               input_image_view.height()+2*additional_border);

    image_with_border_view = view(image_with_border);

    image_with_border_ipl = create_ipl_image(image_with_border_view);
    cv::Mat image_with_borders(image_with_border_ipl.get());

    cv::copyMakeBorder(input_image_mat, image_with_borders,
                       additional_border, additional_border, additional_border, additional_border,
                       cv::BORDER_REPLICATE);

    return const_view(image_with_border);
}


} // end of namespace doppia
