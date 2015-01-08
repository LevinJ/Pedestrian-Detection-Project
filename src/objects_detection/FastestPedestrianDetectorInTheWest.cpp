#include "FastestPedestrianDetectorInTheWest.hpp"

#include "non_maximal_suppression/AbstractNonMaximalSuppression.hpp"
#include "integral_channels/IntegralChannelsForPedestrians.hpp"
#include "SoftCascadeOverIntegralChannelsModel.hpp"

#include "helpers/fill_multi_array.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/gil/image_view_factory.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>

#include <limits>
#include <cmath>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "FastestPedestrianDetectorInTheWest");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "FastestPedestrianDetectorInTheWest");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "FastestPedestrianDetectorInTheWest");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;


options_description
FastestPedestrianDetectorInTheWest ::get_args_options()
{
    options_description desc("FastestPedestrianDetectorInTheWest options");

    desc.add_options()

            ;

    return desc;
}


FastestPedestrianDetectorInTheWest::FastestPedestrianDetectorInTheWest(
        const variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border)
    : BaseIntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p, score_threshold, additional_border),
      IntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p, score_threshold, additional_border)
{

    // will compute one integral channel per octave
    {
        // {min, max}_scale is the scale of the detection window
        const int num_octaves = std::ceil( std::abs((log(max_detection_window_scale) - log(min_detection_window_scale))/log(2)) );

        if(num_octaves <= 0)
        {
            throw std::runtime_error("FastestPedestrianDetectorInTheWest::FastestPedestrianDetectorInTheWest failed to compute num_octaves > 0");
        }

        integral_channels_computers.resize(num_octaves);
        integral_channels_scales.resize(num_octaves);
        const int lowest_exponent = static_cast<int>(log(min_detection_window_scale)/log(2));
        for(int i=0; i < num_octaves; i+=1)
        {
            // integral_channels_scale = 1/detection_window_scale => change of sign in the exponent
            integral_channels_scales[i] = pow(2, -(lowest_exponent + i));
            assert(integral_channels_scales[i] > 0);
        }

        log_info() << str(format("Will compute the integral channels at %i scales") % num_octaves) << std::endl;
    }

    return;
}


FastestPedestrianDetectorInTheWest::~FastestPedestrianDetectorInTheWest()
{
    // nothing to do here
    return;
}


/// FIXME this method is obsolete, will reuse code of current IntegralChannelsDetector version
void FastestPedestrianDetectorInTheWest::compute_integral_channels()
{
    // some debugging variables
    const bool save_integral_channels = false;
    static bool first_call = true;

    assert(integral_channels_scales.size() == integral_channels_computers.size());

    for(size_t i=0; i < integral_channels_computers.size(); i+=1)
    {
        IntegralChannelsForPedestrians &integral_channels_computer = integral_channels_computers[i];
        const float &scale = integral_channels_scales[i];

        // resize the input --
        // integral_channels_scale is the factor to which the input image should be reajusted
        const int scaled_x = input_view.width() * scale;
        const int scaled_y = input_view.height() * scale;

        const gil::opencv::ipl_image_wrapper input_ipl = gil::opencv::create_ipl_image(input_view);
        const cv::Mat input_mat(input_ipl.get());
        cv::Mat scaled_input;

        cv::resize(input_mat, scaled_input, cv::Size(scaled_x, scaled_y) );

        const gil::rgb8c_view_t scaled_input_view =
                gil::interleaved_view(scaled_input.cols, scaled_input.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(scaled_input.data),
                                      static_cast<size_t>(scaled_input.step));

        // compute the features at this octave
        integral_channels_computer.set_image(scaled_input_view);
        integral_channels_computer.compute();

        if(save_integral_channels and (i == integral_channels_computers.size() / 2))
        {
            integral_channels_computer.save_channels_to_file();
            // stop everything
            throw std::runtime_error("Stopped the program so we can debug it. "
                                     "See the result of integral_channels_computer.save_channels_to_file()");
        }

        if(first_call)
        {
            log_debug() << str(format("Integral channel of scale %.3f has size (%i, %i)")
                               % scale
                               % integral_channels_computer.get_channels().shape()[2]
                               % integral_channels_computer.get_channels().shape()[1]) << std::endl;
        }
    } // end of "for each integral channels computer"

    first_call = false;

    return;
}


int detection_window_scale_to_octave_index(const float detection_window_scale, const float min_detection_window_scale,
                                           const int num_octaves)
{
    const int octave = boost::math::iround((log(detection_window_scale) - log(min_detection_window_scale))/log(2));
    assert(octave <= num_octaves); // can be equal, but not superior (we would be missing octaves)
    return std::min<int>(octave, num_octaves-1);
}


/// FIXME what is different with IntegralChannelsDetector ?
/// can we maximise the code reuse of current IntegralChannelsDetector version ?
void FastestPedestrianDetectorInTheWest::compute_detections_at_specific_scale(
        const size_t scale_index,
        const bool save_score_image,
        const bool first_call)
{
    const DetectorSearchRangeMetaData &original_search_range_data = search_ranges_data[scale_index];

    const int octave_index = detection_window_scale_to_octave_index(
                original_search_range_data.detection_window_scale, min_detection_window_scale, integral_channels_scales.size());
    const IntegralChannelsForPedestrians &integral_channels_computer = integral_channels_computers[octave_index];
    const integral_channels_t &integral_channels = integral_channels_computer.get_integral_channels();

    const float integral_channels_scale = integral_channels_scales[octave_index];

    // new scale means new features to evaluate

    // we want relative_scale = original_search_range.detection_window_scale / pow(2, current_octave)
    // integral_channels_scale == 1 / pow(2, current_octave)
    const float relative_scale = original_search_range_data.detection_window_scale * integral_channels_scale;
    const cascade_stages_t cascade_stages = cascade_model_p->get_rescaled_stages(relative_scale);

    const bool print_stages = false;
    const bool print_cascade_statistics = false;
    if(print_stages and first_call)
    {
        log_info() << "Detection cascade stages at detection window scale "
                   << original_search_range_data.detection_window_scale << std::endl;

        print_detection_cascade_variant_stages(log_info(), cascade_stages);
        //throw std::runtime_error("Stopped the program so we can debug it. See the output of print_detection_cascade_stages");
    }

    if(save_score_image)
    {
        log_info() << str(format("For scale %.3f using octave %i which corresponds to integral_channel_scale %.3f")
                          % original_search_range_data.detection_window_scale
                          % octave_index
                          % integral_channels_scale) << std::endl;
    }


#if defined(TESTING)
    const float channels_resizing_factor = 1.0f/integral_channels_computer.get_shrinking_factor();
    const float original_to_channel_scale =
            original_search_range_data.detection_window_scale * integral_channels_scale * channels_resizing_factor;
    const stride_t actual_stride(
                max<stride_t::coordinate_t>(1, x_stride*original_to_channel_scale),
                max<stride_t::coordinate_t>(1, y_stride*original_to_channel_scale));
#endif

    // resize the search range based on the new image size
    if(original_search_range_data.detection_window_ratio != 1.0)
    {
        throw std::invalid_argument("FastestPedestrianDetectorInTheWest does not support detection_window_ratio != 1.0");
    }

    //const float original_to_channel_ratio = 1.0;
    //const DetectorSearchRange scaled_search_range =
    //        original_search_range.get_rescaled(integral_channels_scale*channels_resizing_factor,
    //                                           original_to_channel_ratio);
    const DetectorSearchRange scaled_search_range = compute_scaled_search_range(scale_index);

    if((scaled_search_range.max_x == 0) or (scaled_search_range.max_y == 0))
    {
        // empty search range, the pedestrians would be bigger than the actual image
        return;
    }


    // max < integral_channels.shape (and not <=) since integral images are original image size + 1 row/column
    if((scaled_search_range.max_y >= integral_channels.shape()[1]) or
            (scaled_search_range.max_x >= integral_channels.shape()[2]))
    {
        log_error() << "scaled_search_range.max_y ==" << scaled_search_range.max_y << std::endl;
        log_error() << "integral_channels.shape()[1] ==" << integral_channels.shape()[1] << std::endl;
        log_error() << "scaled_search_range.max_x ==" << scaled_search_range.max_x << std::endl;
        log_error() << "integral_channels.shape()[2] ==" << integral_channels.shape()[2] << std::endl;

        throw std::runtime_error("FastestPedestrianDetectorInTheWest::compute_detections "
                                 "scaled_search_range.max_x or max_y does not fit the expected size. "
                                 "Something went terribly wrong");
    }

    detections_t *non_rescaled_detections_p = NULL;
    doppia::compute_detections_at_specific_scale(
                stages_left_in_the_row,
                stages_left,
                detections_scores,
                integral_channels,
                scale_one_detection_window_size,
                original_search_range_data.detection_window_scale,
                detections, non_rescaled_detections_p,
                cascade_stages,
                score_threshold,
                extra_data_per_scale[scale_index],
                print_stages,
                print_cascade_statistics,
                save_score_image);

#if defined(TESTING)
    // store some key values for testing via DetectorsComparisonTestApplication
    this->scaled_search_range = scaled_search_range;
    this->actual_xstride = actual_xstride;
    this->actual_ystride = actual_ystride;
    this->actual_cascade_stages = cascade_stages;
    this->actual_integral_channels_p = &integral_channels;
#endif


    return;
}

void FastestPedestrianDetectorInTheWest::compute_detections()
{
    // some debugging variables
    const bool save_score_image = false;
    static bool first_call = true;

    for(size_t scale_index=0; scale_index < search_ranges_data.size(); scale_index +=1)
    {
        compute_detections_at_specific_scale(scale_index,
                                             save_score_image, first_call);
    } // end of "for each search_range in search_ranges"

    if(save_score_image)
    {
        // stop everything
        throw std::runtime_error("Stopped the program so we can debug it. "
                                 "See the scores_at_*.png score images");
    }

    log_info() << "number of raw (before non maximal suppression) detections on this frame == "
               << detections.size() << std::endl;

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    compute_non_maximal_suppresion();

    first_call = false;
    return;
}

void FastestPedestrianDetectorInTheWest::compute()
{
    detections.clear();

    compute_integral_channels();
    compute_detections();

    return;
}

} // end of namespace doppia
