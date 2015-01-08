#include "FastestPedestrianDetectorInTheWestV2.hpp"

namespace doppia {

FastestPedestrianDetectorInTheWestV2::FastestPedestrianDetectorInTheWestV2(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   cascade_model_p, non_maximal_suppression_p,
                                   score_threshold, additional_border),
      IntegralChannelsDetector(
          options,
          cascade_model_p, non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseFastestPedestrianDetectorInTheWest(options)
{

    // nothing to do here
    return;
}


FastestPedestrianDetectorInTheWestV2::~FastestPedestrianDetectorInTheWestV2()
{
    // nothing to do here
    return;
}


void FastestPedestrianDetectorInTheWestV2::compute_scaled_detection_cascades()
{
    // FIXME do we really need this function ?
    BaseFastestPedestrianDetectorInTheWest::compute_scaled_detection_cascades();
    return;
}


} // end of namespace doppia
