#ifndef DOPPIA_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
#define DOPPIA_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP

#include "BaseIntegralChannelsDetector.hpp"
#include "IntegralChannelsDetectorModelsBundle.hpp"

#include <boost/program_options/variables_map.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {

class MultiScalesIntegralChannelsModel; // forward declaration


/// Common code shared by MultiscalesIntegralChannelsDetector and GpuMultiscalesIntegralChannelsDetector
/// (declaring BaseIntegralChannelsDetector as virtual inheritance means that
/// children of BaseMultiscalesIntegralChannelsDetector must be also children of BaseIntegralChannelsDetector)
/// http://www.parashift.com/c++-faq-lite/multiple-inheritance.html#faq-25.9
class BaseMultiscalesIntegralChannelsDetector: public virtual BaseIntegralChannelsDetector
{

protected:

    /// This model-less constructor is only meant to be used by BaseIntegralChannelsModelsBundleDetector
    /// This constructor does nothing
    BaseMultiscalesIntegralChannelsDetector(const boost::program_options::variables_map &options);

    /// the constructor is protected because this base class is should not be instanciated directly
    BaseMultiscalesIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p);
    ~BaseMultiscalesIntegralChannelsDetector();

protected:

    const bool use_ground_plane;

    boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p;

    /// Helper function that checks that the model shrinking factor matches the detector options
    /// Will throw an exception in case of mismatch
    void check_shrinking_factor() const;

    /// Helper function that searches for the model with scale one and
    /// sets the detection window size and model_window_to_object_window_converter accordingly
    void search_scale_one();


    /// updates the values inside detection_cascade_per_scale
    /// this variant will also update search_ranges,
    /// (since we will be shifting the actual scales)
    void compute_scaled_detection_cascades();

};



/// helper function called inside BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades
/// reordering search_ranges by scale and making sure
/// detection_{cascade, window_size}_per_scale is also in correct order
void reorder_by_search_range_scale(
        detector_search_ranges_data_t  &search_ranges,
        std::vector<BaseIntegralChannelsDetector::cascade_stages_t>  &detection_cascade_per_scale,
        std::vector<AbstractObjectsDetector::detection_window_size_t> &detection_window_size_per_scale,
        std::vector<float> &original_detection_window_scales);


/// helper function called inside BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades
std::vector<const IntegralChannelsDetectorModelsBundle::detector_t *>find_nearest_scale_detector_model(
        const float detection_window_scale,
        const IntegralChannelsDetectorModelsBundle::detectors_t &detectors);


} // end of namespace doppia

#endif // DOPPIA_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
