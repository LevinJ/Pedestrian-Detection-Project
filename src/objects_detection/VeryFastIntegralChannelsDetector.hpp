#ifndef DOPPIA_VERYFASTINTEGRALCHANNELSDETECTOR_HPP
#define DOPPIA_VERYFASTINTEGRALCHANNELSDETECTOR_HPP

#include "IntegralChannelsDetector.hpp"
#include "BaseVeryFastIntegralChannelsDetector.hpp"


namespace doppia {

/// This is the core detector of
/// R. Benenson, M. Mathias, R. Timofte and L. Van Gool; CVPR 2012 submission
/// This detector should run faster than the FastestPedestrianDetectorInTheWest by avoid any resizing of the
/// input image
class VeryFastIntegralChannelsDetector:
                public IntegralChannelsDetector, public BaseVeryFastIntegralChannelsDetector
{
public:
    VeryFastIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
            const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~VeryFastIntegralChannelsDetector();

protected:

    void process_raw_detections();

    /// helper class for testing
    friend class VeryFastDetectorScaleStatisticsApplication;
};

} // end of namespace doppia

#endif // DOPPIA_VERYFASTINTEGRALCHANNELSDETECTOR_HPP
