#ifndef INTEGRALCHANNELSDETECTOR_HPP
#define INTEGRALCHANNELSDETECTOR_HPP

#include "BaseIntegralChannelsDetector.hpp"

#include "SoftCascadeOverIntegralChannelsModel.hpp"
#include "integral_channels/IntegralChannelsForPedestrians.hpp"


#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/multi_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/cstdint.hpp>

#include <vector>

namespace doppia {
// forward declaration
class IntegralChannelsDetector;
}

namespace bootstrapping {
// forward declaration
class FalsePositivesDataCollector;
}


namespace doppia {

/// This class implements
/// "The Fastest Pedestrian Detector in the West", Dollar et al., BMVC 2010
/// it also allows to add ground plane constrains
class IntegralChannelsDetector: public virtual BaseIntegralChannelsDetector
{
public:

    typedef boost::multi_array<float, 2> detections_scores_t;

    /// using uint8_t instead of bool to avoid the problems related to vector<bool>
    /// http://en.wikipedia.org/wiki/Vector_%28C%2B%2B%29#vector.3Cbool.3E_specialization
    typedef std::vector<boost::uint8_t> stages_left_in_the_row_t;
    typedef boost::multi_array<boost::uint8_t, 2> stages_left_t;

    IntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~IntegralChannelsDetector();

    void set_image(const boost::gil::rgb8c_view_t &input_image,
                   const std::string &image_file_path = std::string());
    void compute();

protected:

    boost::scoped_ptr<IntegralChannelsForPedestrians> integral_channels_computer_p;

    boost::gil::rgb8_image_t input_image;
    boost::gil::rgb8c_view_t input_view;

    float max_score_last_frame;

    /// are there cascade stages left to be executed on this pixel ?
    stages_left_t stages_left;

    /// are there cascade stages left to be executed on this row of pixels ?
    stages_left_in_the_row_t stages_left_in_the_row;

    /// pixel wise detections scores
    detections_scores_t detections_scores;

    void compute_detections_at_specific_scale(
            const size_t search_range_index,
            const bool save_score_image = false,
            const bool first_call = false);


    const IntegralChannelsForPedestrians::integral_channels_t &
    resize_input_and_compute_integral_channels(const size_t search_range_index,
                                               const bool first_call = false);

    size_t get_input_width() const;
    size_t get_input_height() const;

    /// pass from raw detections to the final detections
    virtual void process_raw_detections();

#if defined(BOOTSTRAPPING_LIB)
    friend class bootstrapping::FalsePositivesDataCollector;
    float current_image_scale;

    /// we keep the detections on the rescaled image coordinates,
    /// instead of mapping them back to the original image scale
    detections_t non_rescaled_detections;
#endif

#if defined(TESTING)
    /// helper class for testing
    friend class DetectorsComparisonTestApplication;

    /// helper variables stored for testing via DetectorsComparisonTestApplication
    DetectorSearchRange scaled_search_range;
    int actual_ystride, actual_xstride;
    cascade_stages_t actual_cascade_stages;
    const IntegralChannelsForPedestrians::integral_channels_t *actual_integral_channels_p;
#endif
};



/// helper method shared with FastestPedestrianDetectorInTheWest
void collect_the_detections(
        const DetectorSearchRange &search_range,
        const IntegralChannelsDetector::detections_scores_t &detections_scores,
        const BaseIntegralChannelsDetector::stride_t &stride,
        const float detection_score_threshold,
        const IntegralChannelsDetector::detection_window_size_t &original_detection_window_size,
        IntegralChannelsDetector::detections_t &detections,
        const bool use_input_image_coordinates = true);

/// helper method shared with FastestPedestrianDetectorInTheWest
void compute_detections_at_specific_scale(
        IntegralChannelsDetector::stages_left_in_the_row_t &stages_left_in_the_row,
        IntegralChannelsDetector::stages_left_t &stages_left,
        IntegralChannelsDetector::detections_scores_t &detections_scores,
        const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
        const IntegralChannelsDetector::detection_window_size_t &detection_window_size,
        const float original_detection_window_scale,
        IntegralChannelsDetector::detections_t &detections,
        IntegralChannelsDetector::detections_t *non_rescaled_detections_p,
        const IntegralChannelsDetector::cascade_stages_t &cascade_stages,
        const float score_threshold,
        const ScaleData &scale_data,
        const bool print_stages,
        const bool print_cascade_statistics,
        const bool save_score_image,
        const bool use_the_detector_model_cascade = true); // FIXME move this parameter



void filter_detections(AbstractObjectsDetector::detections_t &detections,
                       const stixels_t &estimated_stixels,
                       const AbstractObjectsDetector::ground_plane_corridor_t &ground_plane_corridor,
                       const int vertical_margin,
                       const int additional_border);

} // namespace doppia

#endif // INTEGRALCHANNELSDETECTOR_HPP
