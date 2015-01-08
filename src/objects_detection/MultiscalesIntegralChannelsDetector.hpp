#ifndef MULTISCALESINTEGRALCHANNELSDETECTOR_HPP
#define MULTISCALESINTEGRALCHANNELSDETECTOR_HPP

#include "IntegralChannelsDetector.hpp"
#include "BaseMultiscalesIntegralChannelsDetector.hpp"

#include "MultiScalesIntegralChannelsModel.hpp"

namespace doppia {

class MultiscalesIntegralChannelsDetector:
        public IntegralChannelsDetector, public BaseMultiscalesIntegralChannelsDetector
{
public:
    MultiscalesIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border);
    ~MultiscalesIntegralChannelsDetector();


};

} // end of namespace doppia

#endif // MULTISCALESINTEGRALCHANNELSDETECTOR_HPP
