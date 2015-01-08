#include "BaseVeryFastIntegralChannelsDetector.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "cascade_stages/check_stages_and_range_visitor.hpp"
#include "cascade_stages/compute_cascade_window_size_visitor.hpp"

#include "helpers/objects_detection/create_json_for_mustache.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <cstdio>


namespace doppia {

MODULE_LOG_MACRO("BaseVeryFastIntegralChannelsDetector")

typedef AbstractObjectsDetector::detection_window_size_t detection_window_size_t;

typedef MultiScalesIntegralChannelsModel::detectors_t detectors_t;
typedef MultiScalesIntegralChannelsModel::detector_t detector_t;

typedef BaseIntegralChannelsDetector::cascade_stages_t cascade_stages_t;
typedef BaseVeryFastIntegralChannelsDetector::fractional_cascade_stages_t fractional_cascade_stages_t;

BaseVeryFastIntegralChannelsDetector::BaseVeryFastIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      // this constructor will be called, since the inheritance is non virtual
      BaseMultiscalesIntegralChannelsDetector(options, detector_model_p)
{
    // shuffling the scales makes the scales inhibition much more effective
    //should_shuffle_the_scales = true; // true by default, since it does not hurt the performance of other methods
    should_shuffle_the_scales = false; // FIXME just for testing


    if(options.count("use_stixels"))
    {
        const bool use_stixels = get_option_value<bool>(options, "use_stixels");
        if(use_stixels)
        {
            // scales inhibition in the stixels use a specific (alternating) access pattern to the scales,
            // which assumes ordered scales
            should_shuffle_the_scales = false;
        }
    }

    return;
}


BaseVeryFastIntegralChannelsDetector::~BaseVeryFastIntegralChannelsDetector()
{
    // nothing to do here
    return;
}

template<typename FeatureType>
void recenter_feature(FeatureType &feature, const float offset_x, const float offset_y)
{
    typename FeatureType::rectangle_t &box = feature.box;
    box.min_corner().x(box.min_corner().x() + offset_x);
    box.max_corner().x(box.max_corner().x() + offset_x);

    box.min_corner().y(box.min_corner().y() + offset_y);
    box.max_corner().y(box.max_corner().y() + offset_y);
    return;
}

template<typename CascadeStageType>
void recenter_cascade(std::vector<CascadeStageType> &stages, const float offset_x, const float offset_y)
{

    BOOST_FOREACH(CascadeStageType &stage, stages)
    {
        recenter_feature(stage.weak_classifier.level1_node.feature, offset_x, offset_y);
        recenter_feature(stage.weak_classifier.level2_true_node.feature, offset_x, offset_y);
        recenter_feature(stage.weak_classifier.level2_true_node.feature, offset_x, offset_y);
    } // end of "for each stage"

    return;
}

/// Helper method used in both CPU and GPU versions
/// for each detection windows, it shift the detection window such as the upper-left corner becomes the center
void recenter_detections(AbstractObjectsDetector::detections_t &detections)
{
    BOOST_FOREACH(Detection2d &detection, detections)
    {
        Detection2d::rectangle_t &box = detection.bounding_box;
        const float
                offset_x = (box.max_corner().x() - box.min_corner().x())/2.0f,
                offset_y = (box.max_corner().y() - box.min_corner().y())/2.0f;

        box.max_corner().x(box.max_corner().x() - offset_x);
        box.min_corner().x(box.min_corner().x() - offset_x);

        box.max_corner().y(box.max_corner().y() - offset_y);
        box.min_corner().y(box.min_corner().y() - offset_y);
    }

    return;
}


std::vector<size_t> get_shuffled_indices(const size_t size)
{
    std::vector<size_t> indices;

    if (size == 0)
    {
        return indices;
    }

    size_t step_size = size - 1;
    indices.push_back(0);

    while(indices.size() < size)
    {
        const std::vector<size_t> current_indices = indices; // copy
        BOOST_FOREACH(const size_t index, current_indices)
        {
            const size_t new_index = index + step_size;

            if (new_index < size)
            {
                const bool not_listed = (std::find(indices.begin(), indices.end(), new_index) == indices.end());
                if(not_listed)
                {
                    indices.push_back(new_index);
                }
            }

        } // end of "for each index in the current list"

        step_size = std::max<size_t>(1, step_size/2);

    } // end of "while not listed all the indices"

    return indices;
}


void BaseVeryFastIntegralChannelsDetector::compute_scaled_detection_cascades()
{
    static bool first_call = true;
    if(first_call)
    {
        printf("BaseVeryFastIntegralChannelsDetector::compute_scaled_detection_cascades\n");
    }

    detection_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    detection_window_size_per_scale.clear();
    detector_index_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges_data.size();
    detection_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    detector_index_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);


    // shuffling the scales makes the scales inhibition much more effective
    if(should_shuffle_the_scales)
    {
        printf("BaseVeryFastIntegralChannelsDetector has the scales shuffling enabled\n");
        const std::vector<size_t> scale_indices = get_shuffled_indices(num_scales);

        detector_search_ranges_data_t reordered_search_ranges;
        for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
        {
            if(false)
            {
                printf("Index %zi is now %zi\n", scale_index, scale_indices[scale_index]);
            }
            reordered_search_ranges.push_back(search_ranges_data[scale_indices[scale_index]]);
        }

        search_ranges_data = reordered_search_ranges;
    }

    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {
        DetectorSearchRangeMetaData &search_range_data = search_ranges_data[scale_index];

        if(search_range_data.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("BaseVeryFastIntegralChannelsDetector does not handle ratios != 1");
        }

        original_detection_window_scales.push_back(search_range_data.detection_window_scale);

        // search the nearest scale model ---
        const detector_t *nearest_detector_scale_p = NULL;
        size_t nearest_detector_scale_index = 0;
        float min_abs_log_scale = std::numeric_limits<float>::max();

        const float search_range_log_scale = std::log(search_range_data.detection_window_scale);
        size_t detector_index = 0;
        BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
        {
            const float
                    log_detector_scale = std::log(detector.get_scale()),
                    abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

            if(abs_log_scale < min_abs_log_scale)
            {
                min_abs_log_scale = abs_log_scale;
                nearest_detector_scale_p = &detector;
                nearest_detector_scale_index = detector_index;
            }

            detector_index += 1;
        } // end of "for each detector"

        assert(nearest_detector_scale_p != NULL);

        if(first_call)
        {
            printf("Selected model scale %.3f for detection window scale %.3f\n",
                   nearest_detector_scale_p->get_scale(), search_range_data.detection_window_scale);
        }

        // update the search range scale --
        search_range_data.detection_window_scale /= nearest_detector_scale_p->get_scale();


        const SoftCascadeOverIntegralChannelsModel::model_window_size_t &model_window_size =
                nearest_detector_scale_p->get_model_window_size();

        detector_t nearest_detector_scale =  *nearest_detector_scale_p; // simple copy

        //const bool recenter_the_search_range = true;
        const bool recenter_the_search_range = false;

        if(recenter_the_search_range)
        {
            const float
                    offset_x = model_window_size.x()/2.0f,
                    offset_y = model_window_size.y()/2.0f;

            detector_t::variant_stages_t &variant_stages = nearest_detector_scale.get_stages();
            detector_t::plain_stages_t *plain_stages_p = boost::get<detector_t::plain_stages_t *>(variant_stages);
            if(plain_stages_p != NULL)
            {
                recenter_cascade(*plain_stages_p, -offset_x, -offset_y);
            }
            else
            {
                throw std::runtime_error("recenter_the_search_range is only implemented for plain_stages_t cascade stages");
            }
        }
        else
        {
            nearest_detector_scale =  *nearest_detector_scale_p; // simple copy
        }


        // get the rescaled detection cascade --
        const float relative_scale = search_range_data.detection_window_scale;
        const cascade_stages_t cascade_stages = nearest_detector_scale.get_rescaled_stages(relative_scale);

        if(recenter_the_search_range)
        {
            //const float
            //        offset_x = (model_window_size.x() * relative_scale)/2.0f,
            //        offset_y = (model_window_size.y() * relative_scale)/2.0f;

            //search_range.min_x += offset_x; search_range.max_x += offset_x;
            //search_range.min_y += offset_y; search_range.max_y += offset_y;


            // FIXME just for testing, very conservative search range
            /*search_range_data.min_x += model_window_size.x(); search_range_data.max_x -= model_window_size.x();
            search_range_data.min_y += model_window_size.y();
            search_range_data.max_y = std::max<int>(0,  search_range_data.max_y - static_cast<int>(model_window_size.y()));*/

            throw std::runtime_error("recenter_the_search_range is deprecated, "
                                     "code needs update to support (again?) this feature");
        }

        detection_cascade_per_scale.push_back(cascade_stages);
        detector_cascade_relative_scale_per_scale.push_back(relative_scale);
        detection_window_size_per_scale.push_back(model_window_size);
        detector_index_per_scale.push_back(nearest_detector_scale_index);
    } // end of "for each search range"


    // In BaseMultiscalesIntegraChannelsDetector we want to re-order the search ranges to group them by similar
    // resized image size, in the very_fast case, all scales use the same image size; so no reordering is needed

    const bool call_create_json_for_mustache = false;
    if(call_create_json_for_mustache)
    {
        // this call will raise an exception and stop the execution
        create_json_for_mustache(detection_cascade_per_scale);
    }

    first_call = false;
    return;
}


void BaseVeryFastIntegralChannelsDetector::compute_extra_data_per_scale(
        const size_t input_width, const size_t input_height)
{
    static bool first_call = true;

    using boost::math::iround;

    extra_data_per_scale.clear();
    extra_data_per_scale.reserve(search_ranges_data.size());

    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const float channels_resizing_factor = 1.0f/IntegralChannelsForPedestrians::get_shrinking_factor();

    for(size_t scale_index=0; scale_index < search_ranges_data.size(); scale_index+=1)
    {
        const DetectorSearchRangeMetaData &search_range_data = search_ranges_data[scale_index];

        if(search_range_data.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("BaseVeryFastIntegralChannelsDetector does not handle ratios != 1");
        }

        // set the extra data --
        ScaleData extra_data;


        // update the scaled input sizes
        {
            // no image resizing, at any scale, yes this is magic !
            extra_data.scaled_input_image_size = image_size_t(input_width, input_height);
        }


        // update the scaled search ranges and strides
        {
            const float
                    detection_window_scale = original_detection_window_scales[scale_index],
                    input_to_channel_scale = channels_resizing_factor,
                    stride_scaling = detection_window_scale*input_to_channel_scale;

            extra_data.stride = stride_t(
                                    std::max<stride_t::coordinate_t>(1, iround(x_stride*stride_scaling)),
                                    std::max<stride_t::coordinate_t>(1, iround(y_stride*stride_scaling)));
            if(first_call)
            {
                log.debug()
                        << boost::str(
                               boost::format(
                                   "Detection window scale %.3f has strides (x,y) == (%.3f, %.3f) [image pixels] =>\t(%i, %i) [channel pixels]\n")
                               % detection_window_scale
                               % (x_stride*stride_scaling)
                               % (y_stride*stride_scaling)
                               % extra_data.stride.x()
                               % extra_data.stride.y()
                               );
            }

            // from input dimensions to integral channel dimensions
            extra_data.scaled_search_range =
                    //search_range_data.get_rescaled(input_to_channel_scale);
                    compute_scaled_search_range(scale_index);
        }

        // update the scaled detection window sizes
        {
            const detection_window_size_t &original_detection_window_size = detection_window_size_per_scale[scale_index];
            const float
                    original_window_scale = search_range_data.detection_window_scale,
                    original_window_ratio = search_range_data.detection_window_ratio,
                    original_window_scale_x = original_window_scale*original_window_ratio;

            const detection_window_size_t::coordinate_t
                    detection_width = iround(original_detection_window_size.x()*original_window_scale_x),
                    detection_height = iround(original_detection_window_size.y()*original_window_scale);

            extra_data.scaled_detection_window_size = detection_window_size_t(detection_width, detection_height);
        }


        extra_data_per_scale.push_back(extra_data);
    } // end of "for each search range"

    // FIXME if we use centered detection, this sanity check will fail (but do we use centered detections, at all ?)
    // sanity check
    check_extra_data_per_scale();

    first_call = false;
    return;
}


DetectorSearchRange BaseVeryFastIntegralChannelsDetector::compute_scaled_search_range(const size_t scale_index) const
{
    static bool first_call = true;

    const DetectorSearchRangeMetaData &search_data = search_ranges_data[scale_index];

    const int
            input_width = get_input_width(),
            input_height = get_input_height(),
            shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    const float input_to_channel_scale = 1.0f/shrinking_factor;

    if(search_data.detection_window_ratio != 1.0)
    {
        throw std::invalid_argument("BaseVeryFastIntegralChannelsDetector does not handle ratios != 1");
    }


    DetectorSearchRange scaled_range;

    scaled_range = search_data; // copy all meta data

    scaled_range.range_scaling *= input_to_channel_scale;
    //scaled_range.range_ratio *= 1;
    scaled_range.detection_window_scale *= input_to_channel_scale;
    //scaled_range.detection_window_ratio *= 1; // ratio is invariant to scaling


    const variant_stages_t &cascade = detection_cascade_per_scale[scale_index];
    compute_cascade_window_size_visitor visitor;
    const detection_window_size_t shrunk_detection_window_size = boost::apply_visitor(visitor, cascade);

    if(first_call)
    {
        log.debug() << boost::format("Scale %u shrunk detection window size (x,y) == (%i, %i)")
                       % scale_index % shrunk_detection_window_size.x() % shrunk_detection_window_size.y()
                    << std::endl;
    }

    // flooring/ceiling/rounding is important to avoid being "off by one pixel" in the search range
    const int
            shrunk_input_width = input_width*input_to_channel_scale,
            shrunk_input_height = input_height*input_to_channel_scale;

    //printf("Scale %zu shrunk input (width, height) == (%i, %i)\n",
    //       scale_index, shrunk_input_width, shrunk_input_height);

    const DetectorSearchRangeMetaData::occlusion_type_t
            occlusion_type = scaled_range.detector_occlusion_type;

    if (occlusion_type == SoftCascadeOverIntegralChannelsModel::NoOcclusion)
    {
        scaled_range.min_x = 0;
        scaled_range.min_y = 0;
        scaled_range.max_x = std::max<int>(0, shrunk_input_width -shrunk_detection_window_size.x());
        scaled_range.max_y = std::max<int>(0, shrunk_input_height -shrunk_detection_window_size.y());
    }
    else
    {
        throw std::runtime_error("BaseVeryFastIntegralChannelsDetector::compute_scaled_search_range "
                                 "does yet support occluded models");

    } // end of "no occlusion or some occlusion"

    assert(scaled_range.max_x <= shrunk_input_width);
    assert(scaled_range.max_y <= shrunk_input_height);

    const float detection_window_scale = original_detection_window_scales[scale_index];

    const bool print_ranges = false;
    if(print_ranges)
    {
        printf("Scale %zu (scale %.3f, occlusion %.3f '%s') will use range min (x,y) == (%i, %i), max (x,y) == (%i, %i)\n",
               scale_index,
               detection_window_scale, // instead of scaled_range.detection_window_scale,
               scaled_range.detector_occlusion_level,
               get_occlusion_type_name(scaled_range.detector_occlusion_type).c_str(),
               scaled_range.min_x, scaled_range.min_y,
               scaled_range.max_x, scaled_range.max_y);
    }


    if((scaled_range.max_x == 0) or (scaled_range.max_y == 0))
    {
        log.info()
                << boost::str(
                       boost::format(
                           "Scale %i (scale %.3f, occlusion %.3f '%s') has an empty search range\n")
                       % scale_index
                       % detection_window_scale // instead of scaled_range.detection_window_scale,
                       % scaled_range.detector_occlusion_level
                       % get_occlusion_type_name(scaled_range.detector_occlusion_type));
    }

    first_call = false;
    return scaled_range;
}


void BaseVeryFastIntegralChannelsDetector::check_extra_data_per_scale()
{

    if(extra_data_per_scale.size() != search_ranges_data.size())
    {
        throw std::runtime_error("BaseVeryFastIntegralChannelsDetector::check_extra_data_per_scale "
                                 "(extra_data_per_scale.size() != search_ranges.size())");
    }


    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();
    //const float channels_resizing_factor = 1.0f/shrinking_factor;

    for(size_t scale_index=0; scale_index < search_ranges_data.size(); scale_index+=1)
    {
        const ScaleData &extra_data = extra_data_per_scale[scale_index];
        //const image_size_t &scaled_input_size = extra_data.scaled_input_image_size;
        const DetectorSearchRange &scaled_search_range = extra_data.scaled_search_range;
        //const detection_window_size_t &scaled_detection_window_size = extra_data.scaled_detection_window_size;

        // strict check
        {
            const variant_stages_t &cascade = detection_cascade_per_scale[scale_index];
            const int
                    //shrunk_width = get_input_width() / shrinking_factor,
                    //shrunk_height = get_input_height() / shrinking_factor;
                    shrunk_width = extra_data.scaled_input_image_size.x() / shrinking_factor,
                    shrunk_height = extra_data.scaled_input_image_size.y() / shrinking_factor;

            check_stages_and_range_visitor visitor(scale_index, scaled_search_range, shrunk_width, shrunk_height);
            const bool everything_is_fine = boost::apply_visitor(visitor, cascade);

            if(not everything_is_fine)
            {
                printf("Model with occlusion '%s' (occlusion level %.3f) at scale %zi failed the safety checks\n",
                       get_occlusion_type_name(scaled_search_range.detector_occlusion_type).c_str(),
                       scaled_search_range.detector_occlusion_level,
                       scale_index);
                throw std::runtime_error("BaseVeryFastIntegralChannelsDetector::check_extra_data_per_scale "
                                         "one of the scales failed the (strict) safety checks");
            }
        } // end of "do strict check"

    } // end of "for each search range"

    return;
}


} // end of namespace doppia
