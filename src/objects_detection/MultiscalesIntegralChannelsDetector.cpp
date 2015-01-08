#include "MultiscalesIntegralChannelsDetector.hpp"


namespace doppia {

MultiscalesIntegralChannelsDetector::MultiscalesIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   non_maximal_suppression_p, score_threshold, additional_border),
      IntegralChannelsDetector(
          options,
          boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
          non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseMultiscalesIntegralChannelsDetector(options, detector_model_p)
{
    // nothing to do here
    return;
}


MultiscalesIntegralChannelsDetector::~MultiscalesIntegralChannelsDetector()
{
    // nothing to do here
    return;
}


} // end of namespace doppia
