#include "BaseFastestPedestrianDetectorInTheWest.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "helpers/Log.hpp"

#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/format.hpp>

#include <cstdio>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "BaseFastestPedestrianDetectorInTheWest");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "BaseFastestPedestrianDetectorInTheWest");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "BaseFastestPedestrianDetectorInTheWest");
}

} // end of anonymous namespace


namespace doppia {


BaseFastestPedestrianDetectorInTheWest::BaseFastestPedestrianDetectorInTheWest(
        const boost::program_options::variables_map &options)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0)
{
    // nothing to do here
    return;
}


BaseFastestPedestrianDetectorInTheWest::~BaseFastestPedestrianDetectorInTheWest()
{
    // nothing to do here
    return;
}



void BaseFastestPedestrianDetectorInTheWest::compute_scaled_detection_cascades()
{
    //printf("BaseFastestPedestrianDetectorInTheWest::compute_scaled_detection_cascades\n");
    assert(cascade_model_p);

    detection_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    detection_window_size_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges_data.size();
    detection_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);

    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {
        const DetectorSearchRangeMetaData &search_range_data = search_ranges_data[scale_index];

        original_detection_window_scales.push_back(search_range_data.detection_window_scale);

        // search the nearest octave scale ---
        const int octave = boost::math::iround(std::log(search_range_data.detection_window_scale)/std::log(2.0f));
        const float octave_detection_window_scale = std::pow<float>(2, static_cast<float>(octave));

        // update the search range scale --
        const float
                original_detection_window_scale = search_range_data.detection_window_scale,
                relative_scale = original_detection_window_scale / octave_detection_window_scale;

        const cascade_stages_t cascade_stages = cascade_model_p->get_rescaled_stages(relative_scale);

        detection_cascade_per_scale.push_back(cascade_stages);
        detector_cascade_relative_scale_per_scale.push_back(relative_scale);
        detection_window_size_per_scale.push_back(scale_one_detection_window_size);
    } // end of "for each search range"

    return;
} // end of BaseFastestPedestrianDetectorInTheWest::compute_scaled_detection_cascades


void BaseFastestPedestrianDetectorInTheWest::compute_extra_data_per_scale(
        const size_t input_width, const size_t input_height)
{
    static bool first_call = true;
    using boost::math::iround;

    extra_data_per_scale.clear();
    extra_data_per_scale.reserve(search_ranges_data.size());

    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const float channels_resizing_factor = 1.0f/IntegralChannelsForPedestrians::get_shrinking_factor();

    for(size_t scale_index=0; scale_index < search_ranges_data.size(); scale_index+=1)
    {
        const DetectorSearchRangeMetaData &search_range_data = search_ranges_data[scale_index];

        // search the nearest octave scale ---
        const int octave = boost::math::iround(std::log(search_range_data.detection_window_scale)/std::log(2.0f));
        const float octave_detection_window_scale = std::pow<float>(2, static_cast<float>(octave));

        // set the extra data --
        ScaleData extra_data;

        const float
                //original_to_channel_scale = 1.0f/search_range.detection_window_scale,
                original_to_channel_ratio = 1.0f/search_range_data.detection_window_ratio,
                //original_to_channel_scale_x = original_to_channel_scale * original_to_channel_ratio;
                octave_to_scaled_input = 1.0f/octave_detection_window_scale;

        if(original_to_channel_ratio != 1.0)
        {
            throw std::invalid_argument("BaseFastestPedestrianDetectorInTheWest does not handle ratios != 1");
        }

        // update the scaled input sizes
        {

            // ratio is defined  as width/height; we apply the "inverse ratio"
            const size_t
                    scaled_x = input_width*octave_to_scaled_input,
                    scaled_y = input_height*octave_to_scaled_input;

            // FIXME move the size checks from GpuIntegralChannelsDetector::resize_input_and_compute_integral_channels into a function here
            extra_data.scaled_input_image_size = image_size_t(scaled_x, scaled_y);
        }


        // update the scaled search ranges and strides
        {
            const float
                    detection_window_scale = original_detection_window_scales[scale_index],
                    octave_to_channel_scale = octave_to_scaled_input*channels_resizing_factor,
                    stride_scaling = detection_window_scale*octave_to_channel_scale;

            extra_data.stride = stride_t(
                                    std::max<stride_t::coordinate_t>(1, iround(x_stride*stride_scaling)),
                                    std::max<stride_t::coordinate_t>(1, iround(y_stride*stride_scaling)));
            if(first_call)
            {
                log_debug()
                        << boost::str(
                               boost::format(
                                   "Detection window scale %.3f has strides (x,y) == (%.3f, %.3f) [image pixels] =>\t(%i, %i) [channel pixels]\n")
                               % detection_window_scale
                               % (x_stride*stride_scaling)
                               % (y_stride*stride_scaling)
                               % extra_data.stride.x()
                               % extra_data.stride.y()
                               );
            }

            // resize the search range based on the new image size
            extra_data.scaled_search_range =
                    //search_range_data.get_rescaled(octave_to_channel_scale, original_to_channel_ratio);
                    compute_scaled_search_range(scale_index);
        }

        // update the scaled detection window sizes
        {
            const detection_window_size_t &original_detection_window_size = detection_window_size_per_scale[scale_index];
            const float
                    original_window_scale = search_range_data.detection_window_scale,
                    original_window_ratio = search_range_data.detection_window_ratio,
                    original_window_scale_x = original_window_scale*original_window_ratio;

            const detection_window_size_t::coordinate_t
                    detection_width = iround(original_detection_window_size.x()*original_window_scale_x),
                    detection_height = iround(original_detection_window_size.y()*original_window_scale);

            extra_data.scaled_detection_window_size = detection_window_size_t(detection_width, detection_height);
        }


        extra_data_per_scale.push_back(extra_data);
    } // end of "for each search range"

    // sanity check
    check_extra_data_per_scale();

    first_call = false;
    return;
} // end of BaseFastestPedestrianDetectorInTheWest::compute_extra_data_per_scale


} // end of namespace doppia
