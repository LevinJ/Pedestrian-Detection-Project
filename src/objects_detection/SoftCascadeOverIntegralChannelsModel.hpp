#ifndef SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP
#define SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP

#include "cascade_stages/SoftCascadeOverIntegralChannelsStage.hpp"
#include "cascade_stages/SoftCascadeOverIntegralChannelsFastStage.hpp"
#include "cascade_stages/SoftCascadeOverIntegralChannelsFastFractionalStage.hpp"
#include "cascade_stages/SoftCascadeOverIntegralChannelsStumpStage.hpp"
#include "cascade_stages/SoftCascadeOverIntegralChannelsStumpSetStage.hpp"

#include "helpers/geometry.hpp"

#include <boost/variant/variant.hpp>

#include <vector>
#include <iosfwd>

// forward declaration
namespace doppia_protobuf {
class DetectorModel;
class SoftCascadeOverIntegralChannelsModel;
}

namespace doppia {


/// See "Robust Object Detection Via Soft Cascade", Bourdev and Brandt, CVPR 2005
/// See also FastestPedestrianDetectorInTheWest
class SoftCascadeOverIntegralChannelsModel
{
public:

    typedef SoftCascadeOverIntegralChannelsStage plain_stage_t;
    typedef std::vector<plain_stage_t> plain_stages_t;

    // stage_t can be converted directly into fast_stage_t
    typedef SoftCascadeOverIntegralChannelsFastStage fast_stage_t;
    typedef std::vector<fast_stage_t> fast_stages_t;

    // stage_t can be converted directly into fast_fractional_stage_t
    typedef SoftCascadeOverIntegralChannelsFastFractionalStage fast_fractional_stage_t;
    typedef std::vector<fast_fractional_stage_t> fast_fractional_stages_t;

    typedef SoftCascadeOverIntegralChannelsStumpStage stump_stage_t;
    typedef std::vector<stump_stage_t> stump_stages_t;

    typedef SoftCascadeOverIntegralChannelsTwoStumpsStage two_stumps_stage_t;
    typedef std::vector<two_stumps_stage_t> two_stumps_stages_t;

    typedef SoftCascadeOverIntegralChannelsThreeStumpsStage three_stumps_stage_t;
    typedef std::vector<three_stumps_stage_t> three_stumps_stages_t;

    typedef SoftCascadeOverIntegralChannelsFourStumpsStage four_stumps_stage_t;
    typedef std::vector<four_stumps_stage_t> four_stumps_stages_t;

    // list of all possible stages types
    typedef
    boost::variant<
    plain_stages_t, // initial one
    fast_stages_t, // the classic one (CVPR2012 results)
    fast_fractional_stages_t, stump_stages_t, // bad ideas
    two_stumps_stages_t, three_stumps_stages_t, four_stumps_stages_t // stump sets
    > variant_stages_t;


    typedef geometry::point_xy<boost::uint16_t> model_window_size_t;
    typedef geometry::point_xy<boost::int16_t> object_window_point_t;
    typedef geometry::box<object_window_point_t> object_window_t;


    /// if occlusion_level > 0, we expect occlusion_type != NoOcclusion
    enum OcclusionTypes { NoOcclusion=0, LeftOcclusion=1, RightOcclusion=2, BottomOcclusion=3, TopOcclusion=4 };
    typedef OcclusionTypes occlusion_type_t;

public:
    /// this constructor will copy the protobuf data into a more efficient data structure
    SoftCascadeOverIntegralChannelsModel(const doppia_protobuf::DetectorModel &model);
    ~SoftCascadeOverIntegralChannelsModel();

    /// returns a soft cascade where each feature has been rescaled
    variant_stages_t get_rescaled_stages(const float relative_scale) const;

    variant_stages_t &get_stages();
    const variant_stages_t &get_stages() const;

    int get_shrinking_factor() const;

    /// Helper method that returns the cascade threshold of the last stage of the model
    float get_last_cascade_threshold() const;

    /// the detection window scale, with respect to the "canonical scale 1"
    float get_scale() const;
    std::string get_semantic_category() const;
    const model_window_size_t &get_model_window_size() const;
    const object_window_t &get_object_window() const;

    float get_occlusion_level() const;
    occlusion_type_t get_occlusion_type() const;

    bool has_soft_cascade() const;

protected:

    variant_stages_t stages;

    int shrinking_factor;
    float scale;
    std::string semantic_category;
    model_window_size_t model_window_size;
    object_window_t object_window;

    float occlusion_level;
    occlusion_type_t occlusion_type;

    void set_stages_from_model(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model);

    /// for left and top occlusion we need to shift the weak learners to stick to the top left corner
    void shift_stages_by_occlusion_level();

    /// Sanity check for the model, is its size/occlusion level/type consistent with its stages ?
    void sanity_check() const;
};

/// debugging helper
template<typename StageType>
void print_detection_cascade_stages(std::ostream &log, const std::vector<StageType> &stages);

void print_detection_cascade_variant_stages(std::ostream &log,
                                            const SoftCascadeOverIntegralChannelsModel::variant_stages_t &stages);


/// Helper method that gives the crucial information for the FPDW implementation
/// these numbers are obtained via
/// doppia/src/test/objects_detection/test_objects_detection + plot_channel_statistics.py
/// method exposed for usage inside DetectorsComparisonTestApplication
float get_channel_scaling_factor(const boost::uint8_t channel_index,
                                 const float relative_scale);

/// small helper function
template<typename Box>
float rectangle_area(const Box &box)
{
    return (box.max_corner().x() - box.min_corner().x())*(box.max_corner().y() - box.min_corner().y());
}


/// returns names such as "left", "right", "top", "bottom"
std::string get_occlusion_type_name(const SoftCascadeOverIntegralChannelsModel::occlusion_type_t occlusion_type);


} // end of namespace doppia

#endif // SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP
