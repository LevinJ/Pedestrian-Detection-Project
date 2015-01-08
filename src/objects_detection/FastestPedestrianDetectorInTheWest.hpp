#ifndef FASTESTPEDESTRIANDETECTORINTHEWEST_HPP
#define FASTESTPEDESTRIANDETECTORINTHEWEST_HPP

#include "IntegralChannelsDetector.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

namespace doppia {

/// This class implements
/// "The Fastest Pedestrian Detector in the West", Dollar et al., BMVC 2010
/// it also allows to add ground plane constrains
class FastestPedestrianDetectorInTheWest: public IntegralChannelsDetector
{
public:
    static boost::program_options::options_description get_args_options();

    FastestPedestrianDetectorInTheWest(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border);
    ~FastestPedestrianDetectorInTheWest();

    void compute();

protected:

    std::vector<IntegralChannelsForPedestrians> integral_channels_computers;
    /// stores the scaling factor of the input image before computing the integral channels
    std::vector<float> integral_channels_scales;

    void compute_integral_channels();
    void compute_detections();

    void compute_detections_at_specific_scale(const size_t scale_index,
                                              const bool save_score_image = false,
                                              const bool first_call = false);

    /// helper class for testing
    friend class DetectorsComparisonTestApplication;
};

} // end of namespace doppia

#endif // FASTESTPEDESTRIANDETECTORINTHEWEST_HPP
