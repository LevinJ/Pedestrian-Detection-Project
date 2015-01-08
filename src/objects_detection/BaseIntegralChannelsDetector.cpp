#include "BaseIntegralChannelsDetector.hpp"

#include "BaseVeryFastIntegralChannelsDetector.hpp" // for dynamic_cast<>

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "ModelWindowToObjectWindowConverterFactory.hpp"

#include "cascade_stages/check_stages_and_range_visitor.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/format.hpp>

#include <algorithm> // for std::max
#include <cstdio>

namespace doppia
{

MODULE_LOG_MACRO("BaseIntegralChannelsDetector")

typedef AbstractObjectsDetector::detection_window_size_t detection_window_size_t;
typedef AbstractObjectsDetector::detections_t detections_t;
typedef AbstractObjectsDetector::detection_t detection_t;


boost::program_options::options_description
BaseIntegralChannelsDetector ::get_args_options()
{
    using namespace boost::program_options;

    options_description desc("BaseIntegralChannelsDetector options");

    desc.add_options()

            ("objects_detector.stixels_vertical_margin",
             value<int>()->default_value(30),
             "vertical margin, in pixels, used to filter out detections based on "
             "the stixels estimate from the previous frame (also used when only estimating ground plane)")

            ("objects_detector.stixels_scales_margin",
             value<int>()->default_value(5),
             "how many scales search around the stixel scale ? The number of scales evaluated is 2*margin. "
             "For values <= 0, all scales will be evaluated. ")

            ;

    return desc;
}


BaseIntegralChannelsDetector::BaseIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p_,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold_,
        const int additional_border_)
    : BaseObjectsDetectorWithNonMaximalSuppression(options, non_maximal_suppression_p),
      score_threshold(score_threshold_),
      use_the_detector_model_cascade(false),
      cascade_model_p(cascade_model_p_),
      additional_border(std::max(additional_border_, 0)),
      stixels_vertical_margin(get_option_value<int>(options, "objects_detector.stixels_vertical_margin")),
      stixels_scales_margin(get_option_value<int>(options, "objects_detector.stixels_scales_margin"))
{

    if(cascade_model_p)
    {
        // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
        if(cascade_model_p->get_shrinking_factor() != IntegralChannelsForPedestrians::get_shrinking_factor())
        {
            printf("cascade_model_p->get_shrinking_factor() == %i\n",
                   cascade_model_p->get_shrinking_factor());

            printf("(Gpu)IntegralChannelsForPedestrians::get_shrinking_factor() == %i\n",
                   IntegralChannelsForPedestrians::get_shrinking_factor());

            throw std::invalid_argument("The input model has a different shrinking factor than "
                                        "the currently used integral channels computer");
        }



        // get the detection window size
        scale_one_detection_window_size = cascade_model_p->get_model_window_size();


        // set the model to object window converter
        model_window_to_object_window_converter_p.reset(
                    ModelWindowToObjectWindowConverterFactory::new_instance(cascade_model_p->get_model_window_size(),
                                                                            cascade_model_p->get_object_window()));

        {
            const bool ignore_cascade = get_option_value<bool>(options, "objects_detector.ignore_soft_cascade");
            use_the_detector_model_cascade = (ignore_cascade == false) and cascade_model_p->has_soft_cascade();

            if(use_the_detector_model_cascade)
            {
                log.info() << "Will use the model soft cascade at run time" << std::endl;
            }
            else
            {
                log.info() << "Will not use a soft cascade at run time" << std::endl;
            }

        }
    }
    else
    {
        // may receive a null cascade_model_p when invoking from MultiscalesIntegralChannelsDetector,
        // thus we do not raise an exception at this stage
    }

    return;
}


BaseIntegralChannelsDetector::~BaseIntegralChannelsDetector()
{
    // nothing to do here
    return;
}


void BaseIntegralChannelsDetector::set_stixels(const stixels_t &stixels)
{
    // we store a copy
    estimated_stixels = stixels;

    if(estimated_stixels.size() != static_cast<size_t>(get_input_width() - 2*additional_border))
    {
        throw std::invalid_argument("BaseIntegralChannelsDetector::set_stixels expects to "
                                    "receive stixels of width 1 pixel, cover the whole image width");
    }

    return;
}


/// Update the search ranges based on the ground plane corridor
void update_search_range(const AbstractObjectsDetector::ground_plane_corridor_t &estimated_ground_plane_corridor,
                         const int input_height,
                         const int vertical_margin,
                         const detection_window_size_t &scale_one_detection_window_size,
                         const std::vector<float> &original_detection_window_scales,
                         std::vector<ScaleData> &extra_data_per_scale)
{
    using boost::math::iround;

    static bool first_call = true;

    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();
    const float channels_resizing_factor = 1.0f/shrinking_factor;

    float
            pixels_count_original = 0,
            pixels_count_updated = 0;

    for(size_t scale_index = 0; scale_index < extra_data_per_scale.size(); scale_index += 1)
    {
        ScaleData &scale_data = extra_data_per_scale[scale_index];
        DetectorSearchRange &search_range = scale_data.scaled_search_range;

        if((search_range.max_x == 0) or (search_range.max_y == 0))
        {
            // nothing to do here
            continue;
        }

        const int detection_height = scale_one_detection_window_size.y()*original_detection_window_scales[scale_index];

        //printf("update_search_range, scale index %zi, detection_height == %i\n",
        //       scale_index, detection_height);

        size_t min_margin_corridor_bottom_v = 0;
        float min_corridor_abs_margin = std::numeric_limits<float>::max();

        // v is the vertical pixel coordinate
        for(size_t bottom_v=0; bottom_v < estimated_ground_plane_corridor.size(); bottom_v+=1)
        {
            const int object_top_v = estimated_ground_plane_corridor[bottom_v];

            if(object_top_v < 0)
            {
                // above the horizon
                continue;
            }

            const int corridor_height = bottom_v - object_top_v;
            assert(corridor_height > 0);

            //printf("Scale index == %zi, detection_height == %i, bottom_v == %i, corridor_height == %i\n",
            //       scale_index, detection_height, bottom_v, corridor_height);

            const float corridor_abs_margin = abs(detection_height - corridor_height);
            if(corridor_abs_margin < min_corridor_abs_margin)
            {
                min_corridor_abs_margin = corridor_abs_margin;
                min_margin_corridor_bottom_v = bottom_v;
            }

        } // end of "for each vertical coordinate"


        //const DetectorSearchRangeMetaData &search_data = search_ranges_data[scale_index];
        //const float input_to_input_scaled = 1.0f/search_data.detection_window_scale;

        const float corridor_y_scale =
                //channels_resizing_factor*input_to_input_scaled;
                // this line works fine with gpu_very_fast, not ok for gpu_channels (why ?!)
                channels_resizing_factor * static_cast<float>(scale_data.scaled_input_image_size.y()) / input_height;

        //printf("update_search_range, scale index %zi, corridor_y_scale == %.3f\n",
        //scale_index, corridor_y_scale);

        //printf("input_to_input_scaled == %.3f, search_range.detection_window_scale == %.3f\n",
        //       input_to_input_scaled,
        //       search_range.detection_window_scale);

        // the search range is defined on the top-left pixel of the detection window
        // search_corridor_top - search_corridor_bottom ~= 2*vertical_margin
        const int
                search_corridor_top = estimated_ground_plane_corridor[min_margin_corridor_bottom_v] - vertical_margin,
                search_corridor_bottom = min_margin_corridor_bottom_v - detection_height + vertical_margin,
                search_corridor_shrunk_top = iround(search_corridor_top*corridor_y_scale),
                search_corridor_shrunk_bottom = iround(search_corridor_bottom*corridor_y_scale),
                // these variables are for debugging
                original_min_y = search_range.min_y,
                original_max_y = search_range.max_y,
                original_height = original_max_y - original_min_y,
                original_width = search_range.max_x - search_range.min_x;


        search_range.min_y = std::max<int>(original_min_y, search_corridor_shrunk_top);
        search_range.max_y = std::min<int>(original_max_y, search_corridor_shrunk_bottom);

        if(search_range.min_y > search_range.max_y)
        {
            const bool print_debug_details = false;
            if(print_debug_details)
            {
                printf("scale_index == %zi\n", scale_index);
                printf("min_corridor_abs_margin == %.3f\n", min_corridor_abs_margin);
                printf("min_margin_corridor_bottom_v == %zi\n", min_margin_corridor_bottom_v);
                printf("estimated_ground_plane_corridor[min_margin_corridor_bottom_v] == %i\n",
                       estimated_ground_plane_corridor[min_margin_corridor_bottom_v]);
                printf("Search corridor top, bottom == %i, %i\n",
                       search_corridor_top, search_corridor_bottom);
                printf("Original search_range.min_y/max_y == %i, %i\n",
                       original_min_y, original_max_y);
                printf("search_range.min_y/max_y == %i, %i\n",
                       search_range.min_y, search_range.max_y);

                printf("Something went terribly wrong inside update_search_range, "
                       "reseting the search range to original values\n");
            }

            if(first_call)
            {
                log.warning() << "At scale index " << scale_index
                              << " the detection window size is larger than the biggest ground plane corridor. "
                              << "Setting the detection search to a single line."
                              << std::endl;
            }

            if(original_min_y == 0 and original_max_y == 0)
            {
                search_range.min_y = 0;
            }
            else
            {
                search_range.min_y = std::max(0, static_cast<int>(search_range.max_y) - 1);
            }

            if(search_range.min_y > search_range.max_y)
            {
                printf("Original search_range.min_y/max_y == %i, %i\n", original_min_y, original_max_y);
                printf("search_range.min_y/max_y == %i, %i\n", search_range.min_y, search_range.max_y);
                throw std::runtime_error("Something went terribly wrong inside update_search_range");
            }
        } // end of "if search range is non-valid"


        const int
                updated_width = (search_range.max_x - search_range.min_x),
                updated_height = (search_range.max_y - search_range.min_y),
                max_reasonable_width_or_height = 1000*channels_resizing_factor;


        if(updated_width > max_reasonable_width_or_height) // sanity check
        {
            printf("scale index == %zi, updated_width == %i\n", scale_index, updated_width);
            throw std::runtime_error("updated_width seems unreasonably high");
        }

        if(updated_height > max_reasonable_width_or_height) // sanity check
        {
            printf("scale index == %zi, updated_height == %i\n", scale_index, updated_height);
            throw std::runtime_error("updated_height seems unreasonably high");
        }

        if(first_call)
        {
            printf("scale_index == %zi, original_height == %i, updated_height == %i\n",
                   scale_index, original_height, updated_height);
        }

        pixels_count_original += original_height*original_width;
        pixels_count_updated += updated_height*original_width;

    } // end of "for each scale index"


    if(first_call)
    {
        printf("Expected speed gain == %.2fx (num pixels original/updated)\n",
               pixels_count_original / ( pixels_count_updated + 1));
    }

    first_call = false;


    //throw std::runtime_error("Please debug me.");
    return;
}


void BaseIntegralChannelsDetector::set_ground_plane_corridor(const ground_plane_corridor_t &corridor)
{
    static bool first_call = true;

    // FIXME do we actually need a copy ? this only impacts the search range...
    // we store a copy
    estimated_ground_plane_corridor = corridor;

    if(estimated_ground_plane_corridor.size() != static_cast<size_t>(get_input_height() - 2*additional_border))
    {
        printf("estimated_ground_plane_corridor.size() == %zi\n", estimated_ground_plane_corridor.size());
        printf("get_input_height() == %zi\n", get_input_height());
        printf("(get_input_height() - 2*additional_border) == %zi\n", get_input_height() - 2*additional_border);

        throw std::invalid_argument("BaseIntegralChannelsDetector::set_ground_plane_corridor expects to "
                                    "receive a vector covering the whole image height");
    }

    if(additional_border != 0)
    {
        throw std::runtime_error("update_search_range for additional_border !=0 not yet implemented");
    }

    // FIXME how to avoid this recomputation ?
    // we recompute the extra data (since it contains the scaled search ranges)
    compute_extra_data_per_scale(get_input_width(), get_input_height());

    if((dynamic_cast<BaseVeryFastIntegralChannelsDetector *>(this) != NULL))
    {
        update_search_range(estimated_ground_plane_corridor,
                            get_input_height(),
                            stixels_vertical_margin,
                            scale_one_detection_window_size,
                            original_detection_window_scales,
                            extra_data_per_scale);

        if(first_call)
        {
            // re-run sanity check
            check_extra_data_per_scale();
        }
    }
    else
    {
        throw std::runtime_error("update_search_range does not (yet) work with methods other than *_very_fast. "
                                 "To be fixed...");
    }

    first_call = false;
    return;
}


void BaseIntegralChannelsDetector::compute_scaled_detection_cascades()
{
    //printf("BaseIntegralChannelsDetector::compute_scaled_detection_cascades\n");
    assert(cascade_model_p);

    detection_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    detection_window_size_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges_data.size();

    detection_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);


    for(size_t scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        DetectorSearchRangeMetaData &search_range_data = search_ranges_data[scale_index];
        original_detection_window_scales.push_back(search_range_data.detection_window_scale);

        const float relative_scale = 1.0f; // we rescale the images, not the features
        const cascade_stages_t cascade_stages = cascade_model_p->get_rescaled_stages(relative_scale);
        detection_cascade_per_scale.push_back(cascade_stages);

        detector_cascade_relative_scale_per_scale.push_back(relative_scale);
        detection_window_size_per_scale.push_back(scale_one_detection_window_size);
    } // end of "for each search range"

    return;
}


void BaseIntegralChannelsDetector::compute_extra_data_per_scale(
        const size_t input_width, const size_t input_height)
{
    static bool first_call = true;
    using boost::math::iround;

    extra_data_per_scale.clear();
    extra_data_per_scale.reserve(search_ranges_data.size());

    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const float channels_resizing_factor = 1.0f/IntegralChannelsForPedestrians::get_shrinking_factor();

    max_search_range_width = 0;
    max_search_range_height = 0;
    for(size_t scale_index = 0; scale_index < search_ranges_data.size(); scale_index += 1)
    {
        const DetectorSearchRangeMetaData &original_search_range_data = search_ranges_data[scale_index];

        ScaleData extra_data;

        const float
                input_to_input_scaled = 1.0f/original_search_range_data.detection_window_scale,
                input_to_input_scaled_ratio = 1.0f/original_search_range_data.detection_window_ratio,
                input_to_input_scaled_x = input_to_input_scaled * input_to_input_scaled_ratio;

        // update the scaled input sizes
        {
            // ratio is defined  as width/height; we apply the "inverse ratio"
            const size_t
                    scaled_x = std::ceil(input_width*input_to_input_scaled_x),
                    scaled_y = std::ceil(input_height*input_to_input_scaled);


            // FIXME move the size checks from GpuIntegralChannelsDetector::resize_input_and_compute_integral_channels into a function here
            extra_data.scaled_input_image_size = image_size_t(scaled_x, scaled_y);
        }


        // update the scaled search ranges and strides
        {
            const float
                    detection_window_scale = original_detection_window_scales[scale_index],
                    input_to_channel_scale = input_to_input_scaled*channels_resizing_factor,
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

            // the search range needs to be recomputed,
            // since the image is rescaled, but the model stays of fixed size
            extra_data.scaled_search_range = compute_scaled_search_range(scale_index);
        }

        // update the scaled detection window sizes
        {
            const detection_window_size_t &original_detection_window_size = detection_window_size_per_scale[scale_index];
            const float
                    original_window_scale = original_search_range_data.detection_window_scale,
                    original_window_ratio = original_search_range_data.detection_window_ratio,
                    original_window_scale_x = original_window_scale*original_window_ratio;

            // ceiling is important to avoid being "off by one pixel" in the search range
            const detection_window_size_t::coordinate_t
                    detection_width = std::ceil(original_detection_window_size.x()*original_window_scale_x),
                    detection_height = std::ceil(original_detection_window_size.y()*original_window_scale);

            extra_data.scaled_detection_window_size = detection_window_size_t(detection_width, detection_height);
        }


        { // update max search range width/height
            const DetectorSearchRange &scaled_search_range = extra_data.scaled_search_range;
            max_search_range_width = std::max(max_search_range_width,
                                              scaled_search_range.max_x - scaled_search_range.min_x);
            max_search_range_height = std::max(max_search_range_height,
                                               scaled_search_range.max_y - scaled_search_range.min_y);
        }

        extra_data_per_scale.push_back(extra_data);
    } // end of "for each search range"


    if(false and (dynamic_cast<BaseVeryFastIntegralChannelsDetector *>(this) != NULL))
    {
        // uses centered windows, so the current sanity check does not apply
        printf("VeryFast detector skips the sanity checks...\n");
    }
    else
    {
        // sanity check
        check_extra_data_per_scale();
    }

    first_call = false;
    return;
} // end of BaseIntegralChannelsDetector::compute_extra_data_per_scale


void BaseIntegralChannelsDetector::check_extra_data_per_scale()
{

    static size_t num_warnings = 0;
    const size_t max_num_warnings = search_ranges_data.size() * 2;

    if(extra_data_per_scale.size() != search_ranges_data.size())
    {
        throw std::runtime_error("BaseIntegralChannelsDetector::check_extra_data_per_scale "
                                 "(extra_data_per_scale.size() != search_ranges.size())");
    }


    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();
    const float channels_resizing_factor = 1.0f/shrinking_factor;

    for(size_t scale_index = 0; scale_index < search_ranges_data.size(); scale_index += 1)
    {

        const ScaleData &extra_data = extra_data_per_scale[scale_index];
        const image_size_t &scaled_input_size = extra_data.scaled_input_image_size;
        const DetectorSearchRange &scaled_search_range = extra_data.scaled_search_range;
        //const detection_window_size_t &scaled_detection_window_size = extra_data.scaled_detection_window_size;

        const detection_window_size_t &detection_window_size = detection_window_size_per_scale[scale_index];
        const float detector_relative_scale = detector_cascade_relative_scale_per_scale[scale_index];

        detection_window_size_t scaled_detection_window_size;
        // detection_window_size is relative to the original image dimensions
        scaled_detection_window_size.x(detection_window_size.x()*detector_relative_scale*channels_resizing_factor);
        scaled_detection_window_size.y(detection_window_size.y()*detector_relative_scale*channels_resizing_factor);


        if(scaled_search_range.detector_occlusion_level == 0)
        { // no occlusion

            if((scaled_search_range.max_y > 0) and (scaled_search_range.max_x > 0))
            {   // assert <= is correct, since we want to check the data access

                const int
                        scaled_height = std::ceil(scaled_input_size.y()*channels_resizing_factor),
                        scaled_width = std::ceil(scaled_input_size.x()*channels_resizing_factor),
                        delta_y = scaled_height -(scaled_search_range.max_y + scaled_detection_window_size.y()),
                        delta_x = scaled_width -(scaled_search_range.max_x + scaled_detection_window_size.x()),
                        max_abs_delta = 10;

                if(delta_y < 0)
                {
                    printf("scale index == %zi, scale == %.3f\n",
                           scale_index, search_ranges_data[scale_index].detection_window_scale);

                    printf("(scaled_search_range.max_y + scaled_detection_window_size.y()) == %i\n",
                           (scaled_search_range.max_y + scaled_detection_window_size.y()));
                    printf("resized_input_size.y() == %i\n", scaled_height);
                    throw std::runtime_error("BaseIntegralChannelsDetector::check_extra_data_per_scale "
                                             "failed the y axis safety check");
                }

                if(delta_x < 0)
                {
                    printf("(scaled_search_range.max_x + scaled_detection_window_size.x()) == %i\n",
                           (scaled_search_range.max_x + scaled_detection_window_size.x()));
                    printf("resized_input_size.x() == %i\n", scaled_width);
                    throw std::runtime_error("BaseIntegralChannelsDetector::compute_scaled_search_range_and_strides "
                                             "failed the x axis safety check");
                }

                const bool not_using_stixels = estimated_ground_plane_corridor.empty();
                if(not_using_stixels and (delta_y > max_abs_delta))
                {
                    // the margin between the y border is too wide, something smells wrong

                    if(false)
                    {
                        throw std::runtime_error("BaseIntegralChannelsDetector::compute_scaled_search_range_and_strides "
                                                 "failed the max_delta_y sanity check");
                    }
                    else if(false or (num_warnings < max_num_warnings))
                    {
                        log.warning() << "The y-margin between search_range + detection_window_size "
                                         "and the image border is suspiciously large (" << delta_y << " pixels)" << std::endl;

                        num_warnings += 1;
                    }
                }

            }

            const bool do_strict_check = true;
            if(do_strict_check)
            {
                const variant_stages_t &cascade = detection_cascade_per_scale[scale_index];
                const int
                        //shrunk_width = get_input_width() / shrinking_factor,
                        //shrunk_height = get_input_height() / shrinking_factor;
                        shrunk_width = extra_data.scaled_input_image_size.x() / shrinking_factor,
                        shrunk_height = extra_data.scaled_input_image_size.y() / shrinking_factor;

                check_stages_and_range_visitor visitor(scale_index, scaled_search_range, shrunk_width, shrunk_height);
                bool everything_is_fine = boost::apply_visitor(visitor, cascade);
                //hack
                everything_is_fine = true;
                if(not everything_is_fine)
                {
                    printf("Model with no occlusion at scale %.3f (number %zi out of %zi) failed the safety checks\n",
                           search_ranges_data[scale_index].detection_window_scale,
                           scale_index,
                           search_ranges_data.size());
                    throw std::runtime_error("BaseIntegralChannelsDetector::check_extra_data_per_scale "
                                             "one of the non-occluded models failed the (strict) safety checks");
                }
            } // end of "do strict check"
        }
        else // (scaled_search_range.detector_occlusion_level != 0)
        { // some occlusion, we do a more detailed check

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
                throw std::runtime_error("BaseIntegralChannelsDetector::check_extra_data_per_scale "
                                         "one of the occluded models failed the safety checks");
            }
        }

    } // end of "for each search range"

    return;
}


/// We search for the next occlusion level,
/// should have same scale, same occlusion type, but have a smaller occlusion value
float get_next_occlusion_level(const size_t scale_index,
                               const detector_search_ranges_data_t &search_ranges_data)
{

    const DetectorSearchRangeMetaData &current_data = search_ranges_data[scale_index];

    const DetectorSearchRangeMetaData::occlusion_type_t
            current_occlusion_type = current_data.detector_occlusion_type;

    const float
            current_scale = current_data.detection_window_scale,
            current_occlusion_level = current_data.detector_occlusion_level;

    float next_occlusion_level = 0, min_delta_level = std::numeric_limits<float>::max();

    BOOST_FOREACH(const DetectorSearchRangeMetaData &other_data, search_ranges_data)
    {
        const bool
                same_occlusion = other_data.detector_occlusion_type == current_occlusion_type,
                same_scale = other_data.detection_window_scale == current_scale,
                lower_occlusion = other_data.detector_occlusion_level < current_occlusion_level;

        if(same_occlusion and same_scale and lower_occlusion)
        {
            const float delta_occlusion_level = std::abs(current_occlusion_level - other_data.detector_occlusion_level);
            if(delta_occlusion_level < min_delta_level)
            {
                min_delta_level = delta_occlusion_level;
                next_occlusion_level = other_data.detector_occlusion_level;
            }
        } // end of "if same scale, occlusion type and has lower occlusion level"

    } // end of "for each range data"


    // if we could not find any other range with same scale or occlusion type,
    // we assume this was the last occlusion level for this detector type,
    // and leave next_occlusion_level == 0

    //printf("Found next_occlusion_level == %.3f\n", next_occlusion_level);

    return next_occlusion_level;
}


DetectorSearchRange BaseIntegralChannelsDetector::compute_scaled_search_range(const size_t scale_index) const
{
    const DetectorSearchRangeMetaData &search_data = search_ranges_data[scale_index];

    const int
            input_width = get_input_width(),
            input_height = get_input_height(),
            shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    const float channels_resizing_factor = 1.0f/shrinking_factor;


    const float
            input_to_input_scaled = 1.0f/search_data.detection_window_scale,
            input_to_input_scaled_ratio = 1.0f/search_data.detection_window_ratio,
            input_to_channel_scale = input_to_input_scaled*channels_resizing_factor,
            input_to_input_scaled_x = input_to_input_scaled * input_to_input_scaled_ratio;

    const size_t
            scaled_input_width = std::ceil(input_width*input_to_input_scaled_x),
            scaled_input_height = std::ceil(input_height*input_to_input_scaled);


    DetectorSearchRange scaled_range;

    scaled_range = search_data; // copy all meta data

    scaled_range.range_scaling *= input_to_channel_scale;
    scaled_range.range_ratio *= input_to_input_scaled_ratio;
    scaled_range.detection_window_scale *= input_to_channel_scale;
    scaled_range.detection_window_ratio *= input_to_input_scaled_ratio; // ratio is invariant to scaling

    const detection_window_size_t &detection_window_size = detection_window_size_per_scale[scale_index];

    if(false)
    {
        printf("Scale %zu detection window size (x,y) == (%i, %i)\n",
               scale_index, detection_window_size.x(), detection_window_size.y());
    }

    // notice that the detection window size is _not_ rescaled
    // ceiling is important to avoid being "off by one pixel" in the search range
    const int
            shrunk_input_width = scaled_input_width*channels_resizing_factor,
            shrunk_input_height = scaled_input_height*channels_resizing_factor,
            detection_window_width = std::ceil(detection_window_size.x()*channels_resizing_factor),
            detection_window_height = std::ceil(detection_window_size.y()*channels_resizing_factor);

    const DetectorSearchRangeMetaData::occlusion_type_t
            occlusion_type = scaled_range.detector_occlusion_type;


    if (occlusion_type == SoftCascadeOverIntegralChannelsModel::NoOcclusion)
    {
        scaled_range.min_x = 0;
        scaled_range.min_y = 0;
        scaled_range.max_x = std::max<int>(0, shrunk_input_width - detection_window_width);
        scaled_range.max_y = std::max<int>(0, shrunk_input_height - detection_window_height);
    }
    else
    {
        const float
                next_occlusion_level = get_next_occlusion_level(scale_index, search_ranges_data),
                current_occlusion_level = scaled_range.detector_occlusion_level;

        assert(current_occlusion_level > 0);
        assert(next_occlusion_level < current_occlusion_level);

        if((scaled_range.detection_window_ratio != 1) or (scaled_range.range_ratio != 1))
        {
            throw std::runtime_error("Occluded detectors do not handle (yet) ratio != 1");
        }

        const float
                scale = scaled_range.detection_window_scale,
                ratio = scaled_range.detection_window_ratio,
                x_scale = scale*ratio, // we multiply so that (w/h)*current_h = new_w
                y_scale = scale;

        const float
                no_occlusion_width = detection_window_size.x()*x_scale,
                no_occlusion_height = detection_window_size.y()*y_scale;

        printf("Scale %zu no occlusion window size (x,y) == (%.3f, %.3f)\n",
               scale_index, no_occlusion_width, no_occlusion_height);

        // ceiling is important to avoid being "off by one pixel" in the search range
        const int
                detection_window_width = std::ceil(detection_window_size.x()*x_scale),
                detection_window_height = std::ceil(detection_window_size.y()*y_scale);

        if (occlusion_type == SoftCascadeOverIntegralChannelsModel::BottomOcclusion)
        {
            const float
                    // it is assumed that next occlusion is lower than current occlusion
                    // when there is less occlusion, the height is higher
                    min_occluded_height = no_occlusion_height*(1 - next_occlusion_level),
                    max_occluded_height = no_occlusion_height*(1 - current_occlusion_level);

            assert(min_occluded_height > max_occluded_height);
            scaled_range.min_y = std::max(0.0f, shrunk_input_height -min_occluded_height);
            scaled_range.max_y = std::max(0.0f, shrunk_input_height -max_occluded_height);

            // x is unaffected by the Bottom occlusion,
            // same as AbstractObjectsDetector::compute_search_ranges
            scaled_range.min_x = 0;
            scaled_range.max_x = std::max<int>(0, shrunk_input_width -detection_window_width);

        }
        else if (occlusion_type == SoftCascadeOverIntegralChannelsModel::LeftOcclusion)
        {
            const float
                    // it is assumed that next occlusion is lower than current occlusion
                    // when there is less occlusion, the width is wider
                    // (right and bottom occlusion are similar, left occlusion case is different)
                    min_occluded_width = no_occlusion_width*(1 - next_occlusion_level),
                    max_occluded_width = no_occlusion_width*(1 - current_occlusion_level),
                    delta_width = min_occluded_width - max_occluded_width;

            assert(min_occluded_width > max_occluded_width);

            scaled_range.min_x = 0;
            scaled_range.max_x = std::ceil(delta_width);

            // y is unaffected by the Left/Right occlusions,
            // same as AbstractObjectsDetector::compute_search_ranges
            scaled_range.min_y = 0;
            scaled_range.max_y = std::max<int>(0, shrunk_input_height -(detection_window_height));
        }
        else if (occlusion_type == SoftCascadeOverIntegralChannelsModel::RightOcclusion)
        {
            // ceiling is important to avoid being "off by one pixel" in the search range
            const int
                    // it is assumed that next occlusion is lower than current occlusion
                    // when there is less occlusion, the width is wider
                    min_occluded_width = std::floor(no_occlusion_width*(1 - next_occlusion_level)),
                    max_occluded_width = std::ceil(no_occlusion_width*(1 - current_occlusion_level));

            assert(min_occluded_width > max_occluded_width);

            scaled_range.min_x = std::max<int>(0, shrunk_input_width -min_occluded_width);
            scaled_range.max_x = std::max<int>(0, shrunk_input_width -max_occluded_width);

            // y is unaffected by the Left/Right occlusions,
            // same as AbstractObjectsDetector::compute_search_ranges
            scaled_range.min_y = 0;
            scaled_range.max_y = std::max<int>(0, shrunk_input_height -detection_window_height);
        }
        else
        {
            throw std::runtime_error("BaseIntegralChannelsDetector::compute_scaled_search_range "
                                     "received an unhandled occlusion type");
        }

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

    return  scaled_range;
}


// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


void add_detection(
        const detection_t::coordinate_t detection_col, const detection_t::coordinate_t detection_row,
        const detection_t::coordinate_t detection_width, const detection_t::coordinate_t detection_height,
        const float detection_score,
        AbstractObjectsDetector::detections_t &detections)
{
    detection_t detection;

    // set the detection_window
    detection_t::rectangle_t &box = detection.bounding_box;
    box.min_corner().x(detection_col);
    box.min_corner().y(detection_row);
    box.max_corner().x(detection_col + std::max<detection_t::coordinate_t>(1, detection_width));
    box.max_corner().y(detection_row + std::max<detection_t::coordinate_t>(1, detection_height));

    const bool print_detection_area = false;
    if(print_detection_area)
    {
        printf("rectangle_area(detection.bounding_box) == %.3f\n",
               rectangle_area(detection.bounding_box));
    }

    // set the detection score
    detection.score = detection_score;
    detection.object_class = detection_t::Pedestrian;

    detections.push_back(detection); // will copy the detection instance
    return;
}


/// occlusion_type == LeftOcclusion
void add_left_detection(
        const detection_t::coordinate_t detection_col, const detection_t::coordinate_t detection_row,
        const detection_t::coordinate_t detection_width, const detection_t::coordinate_t detection_height,
        const float detector_occlusion_level,
        const float detection_score,
        AbstractObjectsDetector::detections_t &detections)
{
    detection_t detection;

    // set the detection_window
    detection_t::rectangle_t &box = detection.bounding_box;
    const float min_x = static_cast<float>(detection_col) - detector_occlusion_level*detection_width;
    box.min_corner().x(min_x);
    box.min_corner().y(detection_row);
    box.max_corner().x(min_x + std::max<detection_t::coordinate_t>(1, detection_width));
    box.max_corner().y(detection_row + std::max<detection_t::coordinate_t>(1, detection_height));

    const bool print_detection_area = false;
    if(print_detection_area)
    {
        printf("rectangle_area(detection.bounding_box) == %.3f\n",
               rectangle_area(detection.bounding_box));
    }

    // set the detection score
    detection.score = detection_score;
    detection.object_class = detection_t::Pedestrian;

    detections.push_back(detection); // will copy the detection instance
    return;
}


void add_detection(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const ScaleData &scale_data,
        detections_t &detections)
{
    using boost::math::iround;

    const DetectorSearchRange &scaled_search_range = scale_data.scaled_search_range;
    const detection_window_size_t &scaled_detection_window_size = scale_data.scaled_detection_window_size;

    detection_t::coordinate_t original_col, original_row;

    // map the detection point back the input image coordinates
    original_col = iround(detection_col/(scaled_search_range.range_scaling*scaled_search_range.range_ratio));
    original_row = iround(detection_row/scaled_search_range.range_scaling);

    //printf("Detection at coordinates %i,%i\n", col, row); // just for debugging

    if((scaled_search_range.detector_occlusion_type == SoftCascadeOverIntegralChannelsModel::NoOcclusion)
       or (scaled_search_range.detector_occlusion_type == SoftCascadeOverIntegralChannelsModel::RightOcclusion)
       or (scaled_search_range.detector_occlusion_type == SoftCascadeOverIntegralChannelsModel::BottomOcclusion))
    { // right and bottom occlusion need no special handling
        add_detection(original_col, original_row,
                      scaled_detection_window_size.x(), scaled_detection_window_size.y(),
                      detection_score, detections);
    }
    else if(scaled_search_range.detector_occlusion_type == SoftCascadeOverIntegralChannelsModel::LeftOcclusion)
    {
        add_left_detection(original_col, original_row,
                           scaled_detection_window_size.x(), scaled_detection_window_size.y(),
                           scaled_search_range.detector_occlusion_level,
                           detection_score, detections);
    }
    else
    {
        throw std::runtime_error("add_detection received a unknown occlusion type");
    }

    return;
}


void add_detection_for_bootstrapping(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const AbstractObjectsDetector::detection_window_size_t &original_detection_window_size,
        AbstractObjectsDetector::detections_t &detections)
{
    detection_t::coordinate_t original_col, original_row, detection_width, detection_height;

    // for the bootstrapping_lib we need the detections in the rescaled image coordinates
    // not in the input image coordinates
    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    // in the boostrapping case, we simply ignore the scale and ratio information
    original_col = detection_col;
    original_row = detection_row;
    detection_width = original_detection_window_size.x()/shrinking_factor;
    detection_height = original_detection_window_size.y()/shrinking_factor;

    add_detection(original_col, original_row, detection_width, detection_height,
                  detection_score, detections);
    return;
}






} // end of namespace doppia
