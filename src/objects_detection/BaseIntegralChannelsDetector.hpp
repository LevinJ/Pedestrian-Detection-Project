#ifndef DOPPIA_BASEINTEGRALCHANNELSDETECTOR_HPP
#define DOPPIA_BASEINTEGRALCHANNELSDETECTOR_HPP

#include "BaseObjectsDetectorWithNonMaximalSuppression.hpp"
#include "ScaleData.hpp"

namespace doppia {


/// This base class fleshes what is common amongst IntegralChannelsDetector and GpuIntegralChannelsDetector
/// The code and members defined here are mainly related to handling of the multiple scales
/// @see IntegralChannelsDetector
/// @see GpuIntegralChannelsDetector
class BaseIntegralChannelsDetector: public BaseObjectsDetectorWithNonMaximalSuppression
{
public:

    typedef SoftCascadeOverIntegralChannelsModel::variant_stages_t variant_stages_t;
    typedef variant_stages_t cascade_stages_t; // FIXME kept to make the code change easier, to remove

    typedef ScaleData::image_size_t image_size_t;
    typedef ScaleData::stride_t stride_t;

    static boost::program_options::options_description get_args_options();

protected:

    /// the constructor is protected because this base class is should not be instanciated directly
    BaseIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold, const int additional_border);
    ~BaseIntegralChannelsDetector();

    void set_stixels(const stixels_t &stixels);
    void set_ground_plane_corridor(const ground_plane_corridor_t &corridor);

protected:

    const float score_threshold;
    bool use_the_detector_model_cascade;

    boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p;

    detection_window_size_t scale_one_detection_window_size;

    /// for single scale detector, all the elements of this vector will be copies of
    /// scale_one_detection_window_size, but for multiscales detector, they will be different
    /// @see MultiScalesIntegralChannelsDetector
    std::vector<detection_window_size_t> detection_window_size_per_scale;

    /// for each entry inside AbstractObjectsDetector::search_ranges we
    /// store the corresponding detector cascade
    std::vector<variant_stages_t> detection_cascade_per_scale;

    /// relative scale of the detection cascade (linked to detection_window_size per scale)
    /// (used for boundary checks only)
    std::vector<float> detector_cascade_relative_scale_per_scale;

    /// helper container for compute_extra_data_per_scale
    std::vector<float> original_detection_window_scales;

    /// additional data needed to compute detections are one specific scale
    std::vector<ScaleData> extra_data_per_scale;
    int max_search_range_width, max_search_range_height;

    /// updates the values inside detection_cascade_per_scale and detection_window_size_per_scale
    virtual void compute_scaled_detection_cascades();

    virtual void compute_extra_data_per_scale(const size_t input_width, const size_t input_height);

    /// helper function that validates the internal consistency of the extra_data_per_scale
    virtual void check_extra_data_per_scale();

    // store information related to ground plane and stixels
    ///@{
    const int additional_border;

    /// for each row in the image, assuming that is the object's bottom position,
    /// this vector stores the expected top object row
    /// top row value is -1 for rows above the horizon
    std::vector<int> estimated_ground_plane_corridor;

    stixels_t estimated_stixels;
    const int stixels_vertical_margin, stixels_scales_margin;
    ///@}

    /// this method must be implemented by the children classes
    virtual size_t get_input_width() const = 0;

    /// this method must be implemented by the children classes
    virtual size_t get_input_height() const = 0;

    /// obtain the 'final' search range including scale, occlusion, detector size and shrinking factor information
    DetectorSearchRange compute_scaled_search_range(const size_t scale_index) const;
};


/// helper method used by IntegralChannelsDetector and GpuIntegralChannelsDetector
void add_detection(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const ScaleData &scale_data,
        AbstractObjectsDetector::detections_t &detections);

/// similar to add_detection, but keeps things in the integral channel reference frame instead of the input image
void add_detection_for_bootstrapping(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const AbstractObjectsDetector::detection_window_size_t &original_detection_window_size,
        AbstractObjectsDetector::detections_t &detections);



} // end of namespace doppia

#endif // DOPPIA_BASEINTEGRALCHANNELSDETECTOR_HPP
