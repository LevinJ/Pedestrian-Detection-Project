#ifndef DOPPIA_BASEFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
#define DOPPIA_BASEFASTESTPEDESTRIANDETECTORINTHEWEST_HPP

#include "BaseIntegralChannelsDetector.hpp"

namespace doppia {

/// Common code shared by FastestPedestrianDetectorInTheWest and GpuFastestPedestrianDetectorInTheWest
/// This variant is similar in spirit (and class hierarchy level) to BaseMultiscalesIntegralChannelsDetector
/// @see BaseMultiscalesIntegralChannelsDetector
/// (declaring BaseIntegralChannelsDetector as virtual inheritance means that
/// children of BaseMultiscalesIntegralChannelsDetector must be also children of BaseIntegralChannelsDetector)
/// http://www.parashift.com/c++-faq-lite/multiple-inheritance.html#faq-25.9
class BaseFastestPedestrianDetectorInTheWest: public virtual BaseIntegralChannelsDetector
{

public:
    typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stage_t fast_fractional_stage_t;
    typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stages_t fast_fractional_stages_t;

    typedef fast_fractional_stage_t fractional_cascade_stage_t;
    typedef fast_fractional_stages_t fractional_cascade_stages_t;

protected:
    /// the constructor is protected because this base class is should not be instanciated directly
    BaseFastestPedestrianDetectorInTheWest(const boost::program_options::variables_map &options);
    ~BaseFastestPedestrianDetectorInTheWest();

protected:

    /// updates the values inside detection_cascade_per_scale and detection_window_size_per_scale
    void compute_scaled_detection_cascades();

    void compute_extra_data_per_scale(const size_t input_width, const size_t input_height);

};

} // namespace doppia

#endif // DOPPIA_BASEFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
