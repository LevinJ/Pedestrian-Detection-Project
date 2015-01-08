#include "VeryFastIntegralChannelsDetector.hpp"

#include "helpers/Log.hpp"

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "VeryFastIntegralChannelsDetector");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "VeryFastIntegralChannelsDetector");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "VeryFastIntegralChannelsDetector");
}

} // end of anonymous namespace


namespace doppia {

VeryFastIntegralChannelsDetector::VeryFastIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   non_maximal_suppression_p, score_threshold, additional_border),
      IntegralChannelsDetector(
          options,
          boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
          non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseVeryFastIntegralChannelsDetector(options, detector_model_p)
{
    // nothing to do here
    return;
}


VeryFastIntegralChannelsDetector::~VeryFastIntegralChannelsDetector()
{
    // nothing to do here
    return;
}



void VeryFastIntegralChannelsDetector::process_raw_detections()
{
    const size_t num_raw_detections = detections.size();

    //recenter_detections(detections); // FIXME should be enabled

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    // filter the detections based on the previous frame stixels
    filter_detections(detections,
                      estimated_stixels, estimated_ground_plane_corridor,
                      stixels_vertical_margin, additional_border);

    log_info() << "number of detections (before non maximal suppression)  on this frame == "
               << num_raw_detections << " (raw) / " <<  detections.size() << " (after filtering)" << std::endl;

    compute_non_maximal_suppresion();

    return;
}


} // end of namespace doppia
