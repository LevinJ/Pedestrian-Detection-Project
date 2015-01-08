#include "BaseMultiscalesIntegralChannelsDetector.hpp"

#include "MultiScalesIntegralChannelsModel.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "ModelWindowToObjectWindowConverterFactory.hpp"

#include "SearchRangeScaleComparator.hpp"

#include "helpers/get_option_value.hpp"

#include <boost/foreach.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <cstdio>


namespace doppia {

typedef MultiScalesIntegralChannelsModel::detectors_t detectors_t;
typedef MultiScalesIntegralChannelsModel::detector_t detector_t;

typedef AbstractObjectsDetector::detection_window_size_t detection_window_size_t;
typedef AbstractObjectsDetector::detections_t detections_t;
typedef AbstractObjectsDetector::detection_t detection_t;

typedef BaseIntegralChannelsDetector::cascade_stages_t cascade_stages_t;

bool will_use_ground_plane(const boost::program_options::variables_map &options)
{
    return (options.count("use_ground_plane") and get_option_value<bool>(options, "use_ground_plane"))
            or (options.count("use_stixels") and get_option_value<bool>(options, "use_stixels"));
}


/// This model less constructor is only meant to be used by BaseIntegralChannelsModelsBundleDetector
/// This constructor does nothing
BaseMultiscalesIntegralChannelsDetector::BaseMultiscalesIntegralChannelsDetector(
        const boost::program_options::variables_map &options)
    : // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      use_ground_plane(will_use_ground_plane(options))
{
    // we do nothing in this constructor
    return;
}


BaseMultiscalesIntegralChannelsDetector::BaseMultiscalesIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p_)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      use_ground_plane(will_use_ground_plane(options)),
      detector_model_p(detector_model_p_)
{
    // the MultiScalesIntegralChannelsModel constructor already validated the consistency of the data

    if(detector_model_p == false)
    {
        throw std::invalid_argument("BaseMultiscalesIntegralChannelsDetector requires "
                                    "a non-null MultiScalesIntegralChannelsModel");
    }

    check_shrinking_factor();
    search_scale_one();
    return;
}


BaseMultiscalesIntegralChannelsDetector::~BaseMultiscalesIntegralChannelsDetector()
{
    // nothing to do here
    return;
}


/// Helper function that checks that the model shrinking factor matches the detector options
/// Will throw an exception in case of mismatch
void BaseMultiscalesIntegralChannelsDetector::check_shrinking_factor() const
{
    BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
    {
        // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
        if(detector.get_shrinking_factor() != IntegralChannelsForPedestrians::get_shrinking_factor())
        {
            printf("detector for scale %.3f has shrinking_factor == %i\n",
                   detector.get_scale(), detector.get_shrinking_factor());

            printf("(Gpu)IntegralChannelsForPedestrians::get_shrinking_factor() == %i\n",
                   IntegralChannelsForPedestrians::get_shrinking_factor());

            throw std::invalid_argument("One of the input models has a different shrinking factor than "
                                        "the currently used integral channels computer");
        }

    } // end of "for each detector"

    return;
}


/// Helper function that searches for the model with scale one and
/// sets the detection window size and model_window_to_object_window_converter accordingly
void BaseMultiscalesIntegralChannelsDetector::search_scale_one()
{
    bool found_scale_one = false;

    BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
    {

        if(detector.get_scale() == 1.0f)
        {
            found_scale_one = true;

            // get the detection window size
            scale_one_detection_window_size = detector.get_model_window_size();

            // set the model to object window converter
            model_window_to_object_window_converter_p.reset(
                        ModelWindowToObjectWindowConverterFactory::new_instance(detector.get_model_window_size(),
                                                                                detector.get_object_window()));
        } // end of "if this is scale one"

    } // end of "for each detector"


    if(found_scale_one == false)
    {
        throw std::invalid_argument("Failed to construct MultiscalesIntegralChannelsDetector because "
                                    "the reiceved data does contain a detector for scale 1");
    }

    return;
}


/// helper function called inside BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades
/// reordering search_ranges by scale and making sure
/// detection_{cascade, window_size}_per_scale is also in correct order
void reorder_by_search_range_scale(
        detector_search_ranges_data_t  &search_ranges_data,
        std::vector<cascade_stages_t>  &detection_cascade_per_scale,
        std::vector<detection_window_size_t> &detection_window_size_per_scale,
        std::vector<float> &original_detection_window_scales)
{
    // (sorting two arrays in C++ is a pain, see
    // http://www.stanford.edu/~dgleich/notebook/2006/03/sorting_two_arrays_simultaneou.html )
    using boost::counting_iterator;

    std::vector<size_t> search_ranges_indices(counting_iterator<size_t>(0),
                                              counting_iterator<size_t>(search_ranges_data.size()));

    SearchRangeScaleComparator search_range_scale_comparator(search_ranges_data);
    std::sort(search_ranges_indices.begin(), search_ranges_indices.end(), search_range_scale_comparator);


    std::vector<cascade_stages_t> reordered_detection_cascade_per_scale;
    detector_search_ranges_data_t reordered_search_ranges;
    std::vector<detection_window_size_t> reordered_detection_window_size_per_scale;
    std::vector<float> reordered_original_detection_window_scales;

    reordered_detection_cascade_per_scale.resize(detection_cascade_per_scale.size());
    reordered_search_ranges.resize(search_ranges_data.size());
    reordered_detection_window_size_per_scale.resize(search_ranges_data.size());
    reordered_original_detection_window_scales.resize(search_ranges_data.size());

    assert(reordered_detection_cascade_per_scale.size() == reordered_search_ranges.size());

    for(size_t index=0; index < search_ranges_indices.size(); index +=1)
    {
        const size_t old_index = search_ranges_indices[index];
        reordered_search_ranges[index] = search_ranges_data[old_index];
        reordered_detection_cascade_per_scale[index] = detection_cascade_per_scale[old_index];
        reordered_detection_window_size_per_scale[index] = detection_window_size_per_scale[old_index];
        reordered_original_detection_window_scales[index] = original_detection_window_scales[old_index];
    }

    detection_cascade_per_scale = reordered_detection_cascade_per_scale;
    search_ranges_data = reordered_search_ranges;
    detection_window_size_per_scale = reordered_detection_window_size_per_scale;
    original_detection_window_scales = reordered_original_detection_window_scales;
    return;
}


/// helper function called inside BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades
std::vector<const detector_t *>find_nearest_scale_detector_model(const float detection_window_scale,
                                                    const detectors_t &detectors)
{
    const detector_t *nearest_detector_scale_p = NULL;
    std::vector<const detector_t *> nearest_detector_scale_pointers;

    float min_abs_log_scale = std::numeric_limits<float>::max();

    const float search_range_log_scale = log(detection_window_scale);
    //find best fit first
    BOOST_FOREACH(const detector_t &detector, detectors)
    {
        const float
                log_detector_scale = log(detector.get_scale()),
                abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

        if(abs_log_scale < min_abs_log_scale)
        {
            min_abs_log_scale = abs_log_scale;
            nearest_detector_scale_p = &detector;
        }
    } // end of "for each detector"

    assert(nearest_detector_scale_p != NULL);
    nearest_detector_scale_pointers.push_back(nearest_detector_scale_p);
    bool first = true;

    BOOST_FOREACH(const detector_t &detector, detectors)
    {
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
            nearest_detector_scale_pointers.push_back(nearest_detector_scale_p);
        }
    } // end of "for each detector"

    return nearest_detector_scale_pointers;
}


/// updates the values inside detection_cascade_per_scale
/// this variant will also update search_ranges,
/// (since we will be shifting the actual scales)
void BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades()
{
    static bool first_call = true;
    if(first_call)
    {
        printf("BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades\n");
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

    detector_search_ranges_data_t additional_component_range_data;

    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {
        DetectorSearchRangeMetaData search_range_data = search_ranges_data[scale_index];

        if(search_range_data.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("MultiscalesIntegralChannelsDetector does not handle ratios != 1");
        }

    

        // search the nearest scale model ---
        std::vector<const detector_t *>nearest_detector_scale_p_vector =
                find_nearest_scale_detector_model(search_range_data.detection_window_scale,
                                                  detector_model_p->get_detectors());
        float detect_win_scale = search_range_data.detection_window_scale;
        for(size_t i =0; i< nearest_detector_scale_p_vector.size(); ++i){
            const detector_t * nearest_detector_scale_p = nearest_detector_scale_p_vector[i];

        if(first_call)
        {
            printf("Selected model scale %.3f for detection window scale %.3f\n",
                   nearest_detector_scale_p->get_scale(), search_range_data.detection_window_scale);
        }


        // update the search range scale --
        {
                original_detection_window_scales.push_back(detect_win_scale);
                search_range_data.detection_window_scale = detect_win_scale/ nearest_detector_scale_p->get_scale();
                search_range_data.semantic_category = nearest_detector_scale_p->get_semantic_category();

            const float relative_scale = 1.0f; // we rescale the images, not the the features
            const cascade_stages_t cascade_stages = nearest_detector_scale_p->get_rescaled_stages(relative_scale);
            detection_cascade_per_scale.push_back(cascade_stages);

            detector_cascade_relative_scale_per_scale.push_back(relative_scale);
            detection_window_size_per_scale.push_back(nearest_detector_scale_p->get_model_window_size());
        }
            if (i>0){

                additional_component_range_data.push_back(search_range_data);
            }else{
                search_ranges_data[scale_index] = search_range_data;
            }


        } // end of "for each nearest detector"
    } // end of "for each search range"
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
