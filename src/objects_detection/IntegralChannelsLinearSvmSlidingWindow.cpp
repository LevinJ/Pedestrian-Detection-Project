#include "IntegralChannelsLinearSvmSlidingWindow.hpp"

#include "non_maximal_suppression/AbstractNonMaximalSuppression.hpp"
#include "integral_channels/IntegralChannelsForPedestrians.hpp"
#include "LinearSvmModel.hpp"

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/gil/image_view_factory.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <limits>

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "IntegralChannelsLinearSvmSlidingWindow");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "IntegralChannelsLinearSvmSlidingWindow");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "IntegralChannelsLinearSvmSlidingWindow");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;
using namespace boost::program_options;

options_description
IntegralChannelsLinearSvmSlidingWindow::get_args_options()
{
    options_description desc("IntegralChannelsLinearSvmSlidingWindow options");

    desc.add_options()


            ;

    return desc;
}


IntegralChannelsLinearSvmSlidingWindow::IntegralChannelsLinearSvmSlidingWindow(
    const variables_map &options,
    boost::shared_ptr<IntegralChannelsForPedestrians> integral_channels_p_,
    boost::shared_ptr<LinearSvmModel> linear_svm_model_p_,
    boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
    const float score_threshold_)
    : BaseObjectsDetectorWithNonMaximalSuppression(options, non_maximal_suppression_p),
      score_threshold(score_threshold_),
      integral_channels_p(integral_channels_p_),
      linear_svm_model_p(linear_svm_model_p_)
{

    if(integral_channels_p->get_feature_vector_length() != linear_svm_model_p->get_w().size())
    {
        log_error() << "IntegralChannels features length does not match the LinearSvmModel size" << std::endl;
        log_error() << "linear_svm_model_p->w.size() == " << linear_svm_model_p->get_w().size() << std::endl;

        throw std::invalid_argument("IntegralChannels features length does not match the LinearSvmModel size");
    }

    const float w_energy = (linear_svm_model_p->get_w().array().abs()).sum();
    log_debug() << "w_energy == " << w_energy << std::endl;
    if(w_energy < 1E-3)
    {
        throw std::invalid_argument("IntegralChannelsLinearSvmSlidingWindow received an empty linear SVM model");
    }

    max_score_last_frame = score_threshold * 2;
    return;
}


IntegralChannelsLinearSvmSlidingWindow::~IntegralChannelsLinearSvmSlidingWindow()
{
    // nothing to do here
    return;
}



void IntegralChannelsLinearSvmSlidingWindow::set_image(const boost::gil::rgb8c_view_t &input_view_, const string &image_file_path)
{
    input_image.recreate(input_view_.dimensions());
    input_view = gil::const_view(input_image);
    gil::copy_pixels(input_view_, gil::view(input_image));
    return;
}


void IntegralChannelsLinearSvmSlidingWindow::compute()
{

    detections.clear();

    const float scale_additive_step = (max_detection_window_scale - min_detection_window_scale) / num_scales;

    // here we use the detection_window size on the channels
    // FIXME hardcoded INRIAPerson dimensions, should transfer the data via LinearSvmModel (or constructor arguments)
    const int
            resizing_factor = integral_channels_p->get_shrinking_factor(),
            detection_window_width = 64 / resizing_factor,
            detection_window_height = 128 / resizing_factor,
            actual_xstride = max<int>(1, x_stride/resizing_factor),
            actual_ystride = max<int>(1, y_stride/resizing_factor);

    // retrieve the window values, and apply the LinearSvm,
    // threshold the detection

    // some debugging variables
    const bool save_score_image = false;
    const bool save_integral_channels = false;
    static int exit_counter = 0;

    // we keep track of the scores range
    float max_score = -numeric_limits<float>::max();
    float min_score = numeric_limits<float>::max();
    //const float half_max_score_last_frame = max_score_last_frame / 2;

    // for each scale
    for(float scale = min_detection_window_scale;
        //false;
        scale < max_detection_window_scale;
        scale = std::min(max_detection_window_scale, scale + scale_additive_step))
    {
        // resize the input --
        // scale is the detection window scale,
        // if the window is kept fix, then the image needs to be scaled at 1/scale
        const int scaled_x = input_view.width() * 1/scale;
        const int scaled_y = input_view.height() * 1/scale;

        const gil::opencv::ipl_image_wrapper input_ipl = gil::opencv::create_ipl_image(input_view);
        const cv::Mat input_mat(input_ipl.get());
        cv::Mat scaled_input;

        cv::resize(input_mat, scaled_input, cv::Size(scaled_x, scaled_y) );

        const gil::rgb8c_view_t scaled_input_view =
                gil::interleaved_view(scaled_input.cols, scaled_input.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(scaled_input.data),
                                      static_cast<size_t>(scaled_input.step));

        // recompute the features --
        integral_channels_p->set_image(scaled_input_view);
        integral_channels_p->compute();


        if(save_integral_channels and (exit_counter == 2) and (scale < 0.8))
        {
            integral_channels_p->save_channels_to_file();
            exit(-1); // FIXME this stops everything
        }

        const IntegralChannelsForPedestrians::channels_t &channels = integral_channels_p->get_channels();
        const int num_channels = channels.shape()[0];
        const int channels_height = channels.shape()[1];
        const int channels_width = channels.shape()[2];

        cv::Mat scores_mat;

        if(save_score_image)
        {
            scores_mat = cv::Mat(channels_height, channels_width, CV_32FC1);
            scores_mat.setTo(0);
        }

        // for each x,y score_threshold
#pragma omp parallel for
        for(int y=0; y < (channels_height - detection_window_height); y+=actual_ystride)
        {
            for(int x=0; x < (channels_width - detection_window_width); x+=actual_xstride)
            {
                // retrieve the features vector --
                Eigen::VectorXf feature_vector;
                feature_vector.setZero(integral_channels_p->get_feature_vector_length());

                int i = 0;
                // feature vector is composed of multiple channels
                for(int c=0; c<num_channels; c+=1)
                {
                    const IntegralChannelsForPedestrians::channels_t::const_reference
                            channel = channels[c];
                    const int max_v = y + detection_window_height;
                    for(int v=y; v<max_v; v+=1)
                    {
                        assert(v < channels_height);
                        const int max_u = x + detection_window_width;
                        for(int u=x; u<max_u; u+=1)
                        {
                            assert(i < feature_vector.size());
                            assert(u < channels_width);
                            feature_vector(i) = channel[v][u];
                            i += 1;
                        } // end of "for each column"
                    } // end of "for each row"

                } // end of "for each channel"

                assert(i == feature_vector.size());

                // compute the score --
                const float score = linear_svm_model_p->compute_score(feature_vector);
                { // may have multi-thread issues here, but we do not care
                    max_score = max(score, max_score);
                    min_score = min(score, min_score);
                }

                if(save_score_image)
                {
                    scores_mat.at<float>(y,x) = score;
                }

                if(score > score_threshold)
                    //if((score > score_threshold) or (score > half_max_score_last_frame))
                {
                    detection_t t_detection;
                    t_detection.score = score;
                    t_detection.object_class = detection_t::Pedestrian;

                    // resize the bounding box to fit the image
                    const float s = resizing_factor*scale;
                    t_detection.bounding_box.min_corner() = detection_t::point_t(x*s, y*s);
                    t_detection.bounding_box.max_corner() = detection_t::point_t((x+detection_window_width)*s,
                                                                                 (y+detection_window_height)*s);
#pragma omp critical
                    { // the detections object is shared amongst threads
                        detections.push_back(t_detection);
                    }
                }

            } // end of "for each column"
        } // end of "for each row"

        if(save_score_image)
        {
            cv::Mat normalized_scores;
            cv::normalize(scores_mat, normalized_scores, 255, 0, cv::NORM_MINMAX);

            const string filename = boost::str(boost::format("scores_at_%.2f.png") % scale);
            cv::imwrite(filename, normalized_scores);
            log_info() << "Created debug file " << filename << std::endl;
        }

    } // end of "for each scale"


    log_debug() << "number of raw (before non maximal suppression) detections on this frame == "
                << detections.size() << std::endl;
    log_debug() << "max_score on this frame == " << max_score << std::endl;
    log_debug() << "min_score on this frame == " << min_score << std::endl;
    max_score_last_frame = max_score;


    if(save_integral_channels or save_score_image)
    {
        exit_counter += 1;
    }

    if(save_score_image)
    {
        if(exit_counter == 2)
        {
            exit(-1); // FIXME this stops everything
        }
    }

    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }


    if(non_maximal_suppression_p)
    {
        non_maximal_suppression_p->set_detections(detections);
        non_maximal_suppression_p->compute();
    }
    else
    {
        // detections == detections
    }

    return;
}


DetectorSearchRange IntegralChannelsLinearSvmSlidingWindow::compute_scaled_search_range(const size_t) const
{
    throw std::runtime_error("IntegralChannelsLinearSvmSlidingWindow::compute_scaled_search_range is not yet implemented");
    return DetectorSearchRange();
}

} // end of namespace doppia
