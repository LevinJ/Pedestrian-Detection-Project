#include "GpuFastestPedestrianDetectorInTheWest.hpp"

#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/copy.hpp>

#include <boost/variant/get.hpp>

namespace doppia {

GpuFastestPedestrianDetectorInTheWest::GpuFastestPedestrianDetectorInTheWest(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   cascade_model_p,
                                   non_maximal_suppression_p, score_threshold, additional_border),
      GpuIntegralChannelsDetector(
          options,
          cascade_model_p, non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseFastestPedestrianDetectorInTheWest(options)
{

    return;
}


GpuFastestPedestrianDetectorInTheWest::~GpuFastestPedestrianDetectorInTheWest()
{
    // nothing to do here
    return;
}

const bool use_fractional_features = false;
//const bool use_fractional_features = true;

void GpuFastestPedestrianDetectorInTheWest::compute_detections_at_specific_scale_v1(
        const size_t search_range_index,
        const bool first_call)
{

    if(use_fractional_features == false)
    {
        GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v1(
                    search_range_index, first_call);
        return;
    }

    doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const ScaleData &scale_data = extra_data_per_scale[search_range_index];

    // const stride_t &actual_stride = scale_data.stride;
    // on current GPU code the stride is ignored, and all pixels of each scale are considered (~x/y_stride == 1E-10)
    // FIXME either consider the strides (not a great idea, using stixels is better), or print a warning at run time

    // compute the scores --
    {
        // compute the detections, and keep the results on the gpu memory
        doppia::objects_detection::integral_channels_detector(
                    integral_channels,
                    search_range_index,
                    scale_data,
                    //gpu_fractional_detection_cascade_per_scale,
                    //gpu_detection_variant_cascade_per_scale,
                    // FIXME quick hack, will raise runtime exception if types to not match
                    boost::get<gpu_detection_cascade_per_scale_t>(gpu_detection_variant_cascade_per_scale),
                    score_threshold, use_the_detector_model_cascade,
                    gpu_detections, num_gpu_detections);
    }

    // ( the detections will be colected after iterating over all the scales )

#if defined(BOOTSTRAPPING_LIB)
    current_image_scale = 1.0f/search_ranges_data[search_range_index].detection_window_scale;
#endif

    return;
}



} // end of namespace doppia
