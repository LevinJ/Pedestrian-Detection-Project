#ifndef INTEGRALCHANNELSLINEARSVMSLIDINGWINDOW_HPP
#define INTEGRALCHANNELSLINEARSVMSLIDINGWINDOW_HPP

#include <boost/gil/image_view.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

#include "BaseObjectsDetectorWithNonMaximalSuppression.hpp"

namespace doppia {

class IntegralChannelsForPedestrians;
class LinearSvmModel;
class AbstractNonMaximalSuppression;

/// This class uses computed integral channels,
/// a linear SVM model and a non-maximum suppression module
/// to do a sliding window detection over an image
/// this module can optionally use a ground plane constraint
class IntegralChannelsLinearSvmSlidingWindow: public BaseObjectsDetectorWithNonMaximalSuppression
{
public:

    static boost::program_options::options_description get_args_options();

    IntegralChannelsLinearSvmSlidingWindow(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<IntegralChannelsForPedestrians> integral_channels_p,
        boost::shared_ptr<LinearSvmModel> linear_svm_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold);
    ~IntegralChannelsLinearSvmSlidingWindow();

    void set_image(const boost::gil::rgb8c_view_t &input_image,
                   const std::string &image_file_path = std::string());
    void compute();

protected:

    const float score_threshold;
    boost::shared_ptr<IntegralChannelsForPedestrians> integral_channels_p;
    boost::shared_ptr<LinearSvmModel> linear_svm_model_p;

    boost::gil::rgb8_image_t input_image;
    boost::gil::rgb8c_view_t input_view;

    float max_score_last_frame;

    DetectorSearchRange compute_scaled_search_range(const size_t scale_index) const;
};

} // end of namespace doppia

#endif // INTEGRALCHANNELSLINEARSVMSLIDINGWINDOW_HPP
