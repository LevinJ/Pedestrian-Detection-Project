#include "BaseIntegralChannelsModelsBundleDetector.hpp"

#include <boost/foreach.hpp>

#include <set>
#include <map>
#include <cstdio>

namespace doppia {


typedef IntegralChannelsDetectorModelsBundle::detectors_t detectors_t;
typedef IntegralChannelsDetectorModelsBundle::detector_t detector_t;


BaseIntegralChannelsModelsBundleDetector::BaseIntegralChannelsModelsBundleDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p_)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      BaseMultiscalesIntegralChannelsDetector(options) // we call a variant that does nothing
{
    // the MultiScalesIntegralChannelsModel constructor already validated the consistency of the data

    detector_model_p = detector_model_p_;

    if(detector_model_p == false)
    {
        throw std::invalid_argument("BaseMultiscalesIntegralChannelsDetector requires "
                                    "a non-null MultiScalesIntegralChannelsModel");
    }

    check_shrinking_factor();
    search_scale_one();
    return;
}


BaseIntegralChannelsModelsBundleDetector::~BaseIntegralChannelsModelsBundleDetector()
{
    // nothing to do here
    return;
}

bool has_multiple_scales(const IntegralChannelsDetectorModelsBundle &model)
{
    bool model_has_multiple_scales = false;

    const detectors_t &detectors = model.get_detectors();
    assert(detectors.empty() == false);

    const float first_scale = detectors.front().get_scale();

    BOOST_FOREACH(const detector_t &detector, detectors)
    {

        if(detector.get_scale() != first_scale)
        {
            model_has_multiple_scales = true;
            break;
        } // end of "if has a different scale"

    } // end of "for each detector"


    return model_has_multiple_scales;
}

typedef std::map<detector_t::occlusion_type_t, detectors_t> detectors_per_occlusion_type_t;

class sort_by_occlusion_level_t
{
public:

    bool operator()(const detector_t &a, const detector_t &b) const
    {
        return a.get_occlusion_level() > b.get_occlusion_level(); // larger occlusions first.
    }
};

/// largest occlusion first, smallest occlusion last
void sort_by_occlusion_level(detectors_per_occlusion_type_t &detectors_per_occlusion_type)
{
    detectors_per_occlusion_type_t::iterator it = detectors_per_occlusion_type.begin();
    for(; it != detectors_per_occlusion_type.end(); ++it)
    {
        if(it->first == detector_t::NoOcclusion)
        {
            // if no occlusion, occlusion_level == 0 for all detectors
            continue;
        }

        detectors_t &detectors = it->second;

        std::sort(detectors.begin(), detectors.end(), sort_by_occlusion_level_t());
    } // end of "for each occlusion type"


    return;
}



/// we assume that all the provided detectors are of the same occlusion type
/// and that they are ordered from larger occlusion level to smaller occlusion level
template<detector_t::occlusion_type_t OcclusionTypeValue>
detector_search_ranges_data_t get_occluded_search_ranges_data_impl(
        const detectors_t &ordered_occlusion_detectors,
        const float detection_window_scale)
{
    detector_search_ranges_data_t search_ranges_data;

    for(size_t detector_index=0; detector_index < ordered_occlusion_detectors.size(); detector_index +=1 )
    {
        const detector_t &detector = ordered_occlusion_detectors[detector_index];
        assert(detector.get_occlusion_type() == OcclusionTypeValue);
        assert(detector.get_occlusion_level() > 0);

        const float current_occlusion_level = detector.get_occlusion_level();
        const float next_occlusion_level =
                (detector_index == (ordered_occlusion_detectors.size() - 1))
                ? 0 // past last occlusion detector, there is zero occlusion
                : ordered_occlusion_detectors[detector_index + 1].get_occlusion_level();

        if(current_occlusion_level <= next_occlusion_level)
        {
            printf("current_occlusion_level == %.3f, next_occlusion_level == %.3f\n",
                   current_occlusion_level, next_occlusion_level);
            throw std::runtime_error("get_occluded_search_ranges_impl expects "
                                     "to receive occlusion detectors in decreasing occlusion level order. "
                                     "This is not the case.");
        }

        DetectorSearchRangeMetaData range_data;

        range_data.detection_window_scale = detection_window_scale;
        range_data.detection_window_ratio = 1; // we do not handle other ratios
        range_data.range_scaling = 1; // no scaling or ration adjustement done
        range_data.range_ratio = 1;

        range_data.detector_occlusion_type = detector.get_occlusion_type();
        range_data.detector_occlusion_level = detector.get_occlusion_level();

        search_ranges_data.push_back(range_data);

    } // end of "for each detector"

    const bool print_occluded_search_ranges = true;
    if(print_occluded_search_ranges and (detection_window_scale == 1.0))
    {
        printf("For occlusion type '%s' at scale %.3f will use search ranges:\n",
               get_occlusion_type_name(OcclusionTypeValue).c_str(),
               detection_window_scale);

        BOOST_FOREACH(const DetectorSearchRangeMetaData range_data, search_ranges_data)
        {
            printf("\tocclusion level %.3f\n", range_data.detector_occlusion_level);
            //            printf("\tocclusion level %.3f, min x,y == (%i, %i), max x,y == (%i, %i)\n",
            //                   range.detector_occlusion_level,
            //                   range.min_x, range.min_y, range.max_x, range.max_y);

        } // end of "for each range"

    } // end of "if print occluded search ranges"

    return search_ranges_data;
}


detector_search_ranges_data_t get_occluded_search_ranges_data(
        const detector_t::occlusion_type_t occlusion_type,
        const detectors_t &detectors,
        const float detection_window_scale)
{

    switch(occlusion_type)
    {
    case detector_t::LeftOcclusion:
        return get_occluded_search_ranges_data_impl<detector_t::LeftOcclusion>(
                    detectors,
                    detection_window_scale);

    case detector_t::RightOcclusion:
        return get_occluded_search_ranges_data_impl<detector_t::RightOcclusion>(
                    detectors,
                    detection_window_scale);

    case detector_t::BottomOcclusion:
        return get_occluded_search_ranges_data_impl<detector_t::BottomOcclusion>(
                    detectors,
                    detection_window_scale);

    case detector_t::TopOcclusion:
        throw std::runtime_error("get_occluded_search_ranges, occlusion type TopOcclusion is not yet implemented");

    case detector_t::NoOcclusion:
        // the NoOcclusion models are already considered in the search_range,
        // so we will add zero entries.
        return detector_search_ranges_data_t();

    default:
        printf("Received occlusion_type == %i\n", occlusion_type);
        throw std::runtime_error("get_occluded_search_ranges received an occlusion type that is not yet implemented");
    }

    return detector_search_ranges_data_t(); // empty return
}


/// helper function called inside BaseIntegralChannelsModelsBundleDetector::compute_scaled_detection_cascades
/// will search for the closest scale, but for an exact match on the occlusion_level
std::vector<const detector_t *>find_nearest_scale_detector_model(const float detection_window_scale,
                                                    const float occlusion_level,
                                                    const detectors_t &detectors)
{
    const detector_t *nearest_detector_scale_p = NULL;

    std::vector<const detector_t *>nearest_detector_scale_p_vector;
    float min_abs_log_scale = std::numeric_limits<float>::max();

    const float search_range_log_scale = log(detection_window_scale);
    BOOST_FOREACH(const detector_t &detector, detectors)
    {
        if(detector.get_occlusion_level() != occlusion_level)
        { // we skip
            continue;
        }

        const float
                log_detector_scale = log(detector.get_scale()),
                abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

        if(abs_log_scale < min_abs_log_scale)
        {
            min_abs_log_scale = abs_log_scale;
            nearest_detector_scale_p = &detector;
        }
    } // end of "for each detector"

    if(nearest_detector_scale_p == NULL)
    {
        throw std::runtime_error("find_nearest_scale_detector_model failed to find a model "
                                 "for the requested occlusion level");
    }
    nearest_detector_scale_p_vector.push_back(nearest_detector_scale_p);
    bool first = true;
    BOOST_FOREACH(const detector_t &detector, detectors)
    {
        if(detector.get_occlusion_level() != occlusion_level)
        { // we skip
            continue;
    }

        const float
                log_detector_scale = log(detector.get_scale()),
                abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

        if (first){
            first = false;
            continue;

        }
        if((abs_log_scale == min_abs_log_scale) and (!first))
        {
            min_abs_log_scale = abs_log_scale;
            nearest_detector_scale_p = &detector;
            nearest_detector_scale_p_vector.push_back(nearest_detector_scale_p);
        }
    } // end of "for each detector"


    return nearest_detector_scale_p_vector;
}


/// updates the values inside detection_cascade_per_scale
/// this variant will also update search_ranges,
/// (since we will be shifting the actual scales)
/// and add aditional search ranges for the different occlusion types
void BaseIntegralChannelsModelsBundleDetector::compute_scaled_detection_cascades()
{
    static bool first_call = true;
    if(first_call)
    {
        printf("BaseIntegralChannelsModelsBundleDetector::compute_scaled_detection_cascades\n");
    }

    detection_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    detection_window_size_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges_data.size();
    detection_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);

    // detector bundles can correspond to two different cases:
    // - A single scale, multiple occlusion levels
    // - Multiple scales, possibly multiple occlusion levels

    const bool model_has_multiple_scales = has_multiple_scales(*detector_model_p);

    if(first_call)
    {
        printf("BaseIntegralChannelsModelsBundleDetector::compute_scaled_detection_cascades, "
               "model bundle %s\n", (model_has_multiple_scales)? "has multiple scales" : "has a single scale");
    }

    const detectors_t &all_detectors = detector_model_p->get_detectors();
    detectors_per_occlusion_type_t detectors_per_occlusion_type;
    std::set<detector_t::occlusion_type_t> occlusion_types;

    BOOST_FOREACH(const detector_t &detector, all_detectors)
    {
        detector_t::occlusion_type_t occlusion_type = detector_t::NoOcclusion;
        if(detector.get_occlusion_level() > 0)
        {
            occlusion_type = detector.get_occlusion_type();
            assert(occlusion_type != detector_t::NoOcclusion);
        }

        detectors_per_occlusion_type[occlusion_type].push_back(detector); // we copy the detector
        occlusion_types.insert(occlusion_type);
    } // end of "for each detector"

    // largest occlusion first, smallest occlusion last
    sort_by_occlusion_level(detectors_per_occlusion_type);


    const detector_search_ranges_data_t no_occlusion_search_ranges_data = search_ranges_data; // simple copy

    // for each entry of search ranges, we store which occlusion type does it correspond.
    // (should this be stored directly inside DetectorSearchRange ?)
    const boost::gil::rgb8c_view_t::point_t input_dimensions(get_input_width(), get_input_height());

    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {

        const DetectorSearchRangeMetaData &search_range_data = no_occlusion_search_ranges_data[scale_index];
        const float detection_window_scale = search_range_data.detection_window_scale;

        BOOST_FOREACH(const detector_t::occlusion_type_t occlusion_type, occlusion_types)
        {

            const detectors_t &detectors = detectors_per_occlusion_type[occlusion_type];

            const detector_search_ranges_data_t
                    occluded_search_ranges = get_occluded_search_ranges_data(
                                                 occlusion_type,
                                                 detectors,
                                                 detection_window_scale);

            search_ranges_data.insert(search_ranges_data.end(),
                                      occluded_search_ranges.begin(), occluded_search_ranges.end());

        } // end of "for each occlusion type"

    } // end of "for each scale"

    // handling of a special case used for debugging,
    // only occlusion detectors, no non occluded detectors
    if(detectors_per_occlusion_type[detector_t::NoOcclusion].empty())
    {
        // we remove all original search ranges (since they all correspond to NoOcclusion)
        search_ranges_data.erase(search_ranges_data.begin(), search_ranges_data.begin() + num_scales);
    } // end of "if no NoOcclusion detectors"


    detector_search_ranges_data_t additional_component_range_data;
    for(size_t search_range_index=0; search_range_index < search_ranges_data.size(); search_range_index+=1)
    {
        DetectorSearchRangeMetaData search_range_data = search_ranges_data[search_range_index];

        if(search_range_data.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("BaseIntegralChannelsModelsBundleDetector does not handle ratios != 1");
        }


        // search the nearest scale model (with the proper occlusion level) ---
        const detector_t::occlusion_type_t occlusion_type = search_range_data.detector_occlusion_type;
        // occlusion_type can be detector_t::NoOcclusion
        const detectors_t &detectors_with_same_occlusion_type = detectors_per_occlusion_type[occlusion_type];
        std::vector<const detector_t *>nearest_detector_scale_p_vector =
                (occlusion_type == detector_t::NoOcclusion)
                // if no occlusion we do not constrain the occlusion level
                ? find_nearest_scale_detector_model(search_range_data.detection_window_scale,
                                                    detectors_with_same_occlusion_type)
                  // otherwise we search for a specific occlusion level
                : find_nearest_scale_detector_model(search_range_data.detection_window_scale,
                                                    search_range_data.detector_occlusion_level,
                                                    detectors_with_same_occlusion_type);
        float detect_win_scale = search_range_data.detection_window_scale;
        for(size_t i =0; i< nearest_detector_scale_p_vector.size(); ++i){
            const detector_t * nearest_detector_scale_p = nearest_detector_scale_p_vector[i];

        if(nearest_detector_scale_p == NULL)
        {
            printf("occlusion_type == %i\n", occlusion_type);
            throw std::runtime_error("nearest_detector_scale_p == NULL in "
                                     "BaseIntegralChannelsModelsBundleDetector::compute_scaled_detection_cascades");
        }


        // update the search range scale --
        {
                original_detection_window_scales.push_back(detect_win_scale);
                search_range_data.detection_window_scale = detect_win_scale / nearest_detector_scale_p->get_scale();
                search_range_data.semantic_category = nearest_detector_scale_p->get_semantic_category();

            const float relative_scale = 1.0f; // we rescale the images, not the the features
            const cascade_stages_t cascade_stages = nearest_detector_scale_p->get_rescaled_stages(relative_scale);
            detection_cascade_per_scale.push_back(cascade_stages);

            detector_cascade_relative_scale_per_scale.push_back(relative_scale);
            detection_window_size_per_scale.push_back(nearest_detector_scale_p->get_model_window_size());
        }
            if(first_call)
            {
                printf("Selected model scale %.3f (occlusion type %s,\tocclusion level %.3f) "
                       "for detection window scale %.3f, semantic category = %s\n",
                       nearest_detector_scale_p->get_scale(),
                       get_occlusion_type_name(nearest_detector_scale_p->get_occlusion_type()).c_str(),
                       nearest_detector_scale_p->get_occlusion_level(),
                       search_range_data.detection_window_scale,
                       nearest_detector_scale_p->get_semantic_category().c_str());
            }
            if (i > 0)
            {
                additional_component_range_data.push_back(search_range_data);
            }
            else
            {
                search_ranges_data[search_range_index] = search_range_data;
            }


        } // end of "for each nearest detector"

    } // end of "for each search range"

    //add search ranges for additional components
    search_ranges_data.insert(search_ranges_data.end(),
                              additional_component_range_data.begin(), additional_component_range_data.end());

    if(use_ground_plane == false)
    { // we only re-order when not using ground planes

        // reordering search_ranges by scale and making sure detection_cascade_per_scale is also in correct order
        reorder_by_search_range_scale(search_ranges_data, detection_cascade_per_scale,
                                      detection_window_size_per_scale, original_detection_window_scales);
    }

    first_call = false;

    return;
}



} // end of namespace doppia
