#include "AbstractObjectsDetector.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>

#include <stdexcept>
#include <cmath>


namespace doppia {

MODULE_LOG_MACRO("AbstractObjectsDetector")

using namespace std;
using namespace boost;
using namespace boost::program_options;

options_description
AbstractObjectsDetector::get_args_options()
{
    options_description desc("AbstractObjectsDetector options");

    desc.add_options()

            ("objects_detector.min_scale", value<float>()->default_value(0.3),
             "minimum detection window scale explored for detections")
            // default max_scale 0.3 == 40 pixels (A. Ess annotations height) / 128 pixels (pedestrian height)

            ("objects_detector.max_scale", value<float>()->default_value(5.0),
             "maximum detection window scale explored for detections")
            // default max_scale 5 == 640 pixels (VGA height) / 128 pixels (pedestrian height)

            ("objects_detector.num_scales", value<int>()->default_value(10),
             "number of scales to explore. (this is combined with num_ratios)")

            ("objects_detector.min_ratio", value<float>()->default_value(1),
             "minimum ratio (width/height) of the detection window used to explore detections")

            ("objects_detector.max_ratio", value<float>()->default_value(1),
             "max ratio (width/height) of the detection window used to explore detections")

            ("objects_detector.num_ratios", value<int>()->default_value(1),
             "number of ratios to explore. (this is combined with num_scales)")

            ("objects_detector.x_stride", value<float>()->default_value(4),
             "size of steps used when sliding the detection window. x axis. "
             "value can be a fraction smaller than 1 to do sub-pixel search (when possible). "
             "The stride is adapted with the scales, at scale 2, stride*2 will be used.")

            ("objects_detector.y_stride", value<float>()->default_value(4),
             "size of steps used when sliding the detection window. y axis. "
             "value can be a fraction smaller than 1 to do sub-pixel search (when possible). "
             "The stride is adapted with the scales, at scale 2, stride*2 will be used.")

            ("objects_detector.resize_detections", value<bool>()->default_value(true),
             "resize the detection windows to match the actual object size and not the model size")

            ;

    return desc;
}


AbstractObjectsDetector::AbstractObjectsDetector(const variables_map &options)
    :
      min_detection_window_scale(get_option_value<float>(options, "objects_detector.min_scale")),
      max_detection_window_scale(get_option_value<float>(options, "objects_detector.max_scale")),
      num_scales(get_option_value<int>(options, "objects_detector.num_scales")),
      min_detection_window_ratio(get_option_value<float>(options, "objects_detector.min_ratio")),
      max_detection_window_ratio(get_option_value<float>(options, "objects_detector.max_ratio")),
      num_ratios(get_option_value<int>(options, "objects_detector.num_ratios")),
      x_stride(get_option_value<float>(options, "objects_detector.x_stride")),
      y_stride(get_option_value<float>(options, "objects_detector.y_stride")),
      resize_detection_windows(get_option_value<bool>(options, "objects_detector.resize_detections"))
{

    if(num_scales <= 0)
    {
        throw std::runtime_error("objects_detector.num_scales should be >= 1");
    }

    if(num_ratios <= 0)
    {
        throw std::runtime_error("objects_detector.num_ratios should be >= 1");
    }

    if((num_scales*num_ratios) > 500)
    {
        log.warning() << "objects_detector.num_scales*objects_detector.num_ratios is larger than 500, "
                         "please be very, very patient" << std::endl;
        //throw std::runtime_error("objects_detector.num_scales larger than 100? I refuse to waste CPU/GPU.");
    }

    if(min_detection_window_scale > max_detection_window_scale)
    {
        throw std::runtime_error("objects_detector.min_scale should be smaller or equal to objects_detector.max_scale");
    }

    if(min_detection_window_ratio > max_detection_window_ratio)
    {
        throw std::runtime_error("objects_detector.min_ratio should be smaller or equal to objects_detector.max_ratio");
    }

    return;
}


AbstractObjectsDetector::~AbstractObjectsDetector()
{
    // nothing to do here
    return;
}


const AbstractObjectsDetector::detections_t & AbstractObjectsDetector::get_detections()
{
    return detections;
}


const AbstractObjectsDetector::detections_t &AbstractObjectsDetector::get_raw_detections() const
{
    return detections;
}


void AbstractObjectsDetector::set_raw_detections(const AbstractObjectsDetector::detections_t &detections_)
{
    detections = detections_;
    return;
}


void AbstractObjectsDetector::set_stixels(const stixels_t &/*stixels*/)
{
    // default implementation simply disregards the estimated stixels
    return;
}


void AbstractObjectsDetector::set_ground_plane_corridor(const ground_plane_corridor_t &/*corridor*/)
{
    // default implementation simply disregards the estimated ground plane corridor
    return;
}


void AbstractObjectsDetector::compute_search_ranges_meta_data(detector_search_ranges_data_t &search_ranges_data) const
{

    search_ranges_data.clear(); // remove previous ranges if they exist
    float scale_logarithmic_step = 0;
    if(num_scales > 1)
    {
        scale_logarithmic_step = (std::log(max_detection_window_scale) - std::log(min_detection_window_scale)) / (num_scales -1);
    }

    // ratio step is linear
    const float ratio_step = (max_detection_window_ratio - min_detection_window_ratio) / num_ratios;

    float scale = min_detection_window_scale;
    assert(scale > 0);
    for(int scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        float ratio = min_detection_window_ratio;
        for(int ratios_index = 0; ratios_index < num_ratios; ratios_index += 1)
        {
            DetectorSearchRangeMetaData range_data;

            // min/max x/y are set in the original images coordinates
            range_data.detection_window_scale = scale;
            range_data.detection_window_ratio = ratio;
            range_data.range_scaling = 1;
            range_data.range_ratio = 1;

            // by default, no occlusion
            range_data.detector_occlusion_type = SoftCascadeOverIntegralChannelsModel::NoOcclusion;
            range_data.detector_occlusion_level = 0;

            search_ranges_data.push_back(range_data);

            ratio += ratio_step;
        } // end of "for each ratio"

        scale = std::min(max_detection_window_scale, exp(std::log(scale) + scale_logarithmic_step));
    } // end of "for each scale"

    return;
}


} // end of namespace doppia
