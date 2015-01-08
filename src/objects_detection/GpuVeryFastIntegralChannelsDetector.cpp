#include "GpuVeryFastIntegralChannelsDetector.hpp"

#include "MultiScalesIntegralChannelsModel.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include "cudatemplates/hostmemoryheap.hpp"
#include "cudatemplates/copy.hpp"

#include <boost/foreach.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <omp.h>

#include <omp.h>


namespace doppia {

MODULE_LOG_MACRO("GpuVeryFastIntegralChannelsDetector")

GpuVeryFastIntegralChannelsDetector::GpuVeryFastIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   non_maximal_suppression_p, score_threshold, additional_border),
      GpuIntegralChannelsDetector(
          options,
          boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
          non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseVeryFastIntegralChannelsDetector(options, detector_model_p)
{
    const bool ignore_cascade = get_option_value<bool>(options, "objects_detector.ignore_soft_cascade");
    use_the_detector_model_cascade = (ignore_cascade == false) and detector_model_p->has_soft_cascade();

    if(use_the_detector_model_cascade)
    {
        log.info() << "Will use the model soft cascade at run time" << std::endl;
    }
    else
    {
        log.info() << "Will not use a soft cascade at run time" << std::endl;
    }

    { // set the helper variables

        scale_logarithmic_step = 0;
        log_min_detection_window_scale = std::log(min_detection_window_scale);
        if(num_scales > 1)
        {
            scale_logarithmic_step = (std::log(max_detection_window_scale) - log_min_detection_window_scale) / (num_scales -1);
        }

        typedef MultiScalesIntegralChannelsModel::detector_t detector_t;
        object_to_detection_window_height_ratio = 1.0f;
        BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
        {
            if(detector.get_scale() == 1.0f)
            {
                const float
                        object_height =
                        detector.get_object_window().max_corner().y() - detector.get_object_window().min_corner().y(),
                        detection_window_height = detector.get_model_window_size().y();

                // we assume the object is centered in the detection window
                object_to_detection_window_height_ratio = detection_window_height / object_height;
            }
        } // end of "for each detector"
    }

    // we setup stixels_max_search_range_height since it is fixed at runtime --
    {
        const int
                shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor(),
                shrunk_vertical_margin = stixels_vertical_margin / shrinking_factor;

        stixels_max_search_range_height = 2*shrunk_vertical_margin;
    }
    return;
}


GpuVeryFastIntegralChannelsDetector::~GpuVeryFastIntegralChannelsDetector()
{
    // nothing to do here
    return;
}


void GpuVeryFastIntegralChannelsDetector::compute()
{

    const bool use_v2 = true;
    //const bool use_v2 = false;
    if(use_v2)
    {
        compute_v2();
    }
    else
    {   // we use vanilla GPU compute
        // with the new windows centering, this version is deprecated
        GpuIntegralChannelsDetector::compute();
    }

    return;
}


/// This function is identical to GpuIntegralChannelsDetector::set_image except it loads
/// additional per-scale information into the GPU
void GpuVeryFastIntegralChannelsDetector::set_image(const boost::gil::rgb8c_view_t &input_view,
                                                    const std::string &image_file_path)
{
    const bool input_dimensions_changed =
            ((input_gpu_mat.cols != input_view.width()) or (input_gpu_mat.rows != input_view.height()));

    GpuIntegralChannelsDetector::set_image(input_view);

    if(false)
    {
        printf("GpuVeryFastIntegralChannelsDetector::set_image received an image of dimensions (%li, %li)\n",
               input_view.width(), input_view.height());
    }

    if(input_dimensions_changed)
    {
        set_gpu_scales_data();
        set_gpu_half_window_widths();
    } // end of "if input dimensions changed"

    return;
}

void GpuVeryFastIntegralChannelsDetector::set_ground_plane_corridor(const ground_plane_corridor_t &corridor)
{
    BaseIntegralChannelsDetector::set_ground_plane_corridor(corridor);

    // since we have new search ranges, we need to update the gpu scales data
    set_gpu_scales_data();
    return;
}

void GpuVeryFastIntegralChannelsDetector::set_stixels(const stixels_t &stixels)
{
    BaseIntegralChannelsDetector::set_stixels(stixels);

    // we move the stixels data to GPU
    set_gpu_stixels();

    set_gpu_half_window_widths();
    return;
}


/// updates the GPU data about the scales
void GpuVeryFastIntegralChannelsDetector::set_gpu_scales_data()
{
    if(search_ranges_data.empty())
    {
        throw std::runtime_error("GpuVeryFastIntegralChannelsDetector::set_gpu_scales_data received "
                                 "search_ranges.empty() == true");
    }

    const size_t num_scales = search_ranges_data.size();
    Cuda::HostMemoryHeap1D<gpu_scale_datum_t> cpu_scales_data(num_scales);

    { // we initialize max_search_range
        const ScaleData &scale_data = extra_data_per_scale[0];
        const DetectorSearchRange &scaled_search_range = scale_data.scaled_search_range;

        max_search_range.min_corner().x( scaled_search_range.min_x  );
        max_search_range.min_corner().y( scaled_search_range.min_y  );
        max_search_range.max_corner().x( scaled_search_range.max_x  );
        max_search_range.max_corner().y( scaled_search_range.max_y  );

        max_search_range_width = scaled_search_range.max_x - scaled_search_range.min_x;
        max_search_range_height = scaled_search_range.max_y - scaled_search_range.min_y;
    }

    for(size_t scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        const ScaleData &scale_data = extra_data_per_scale[scale_index];
        const DetectorSearchRange &scaled_search_range = scale_data.scaled_search_range;

        gpu_scale_datum_t &datum = cpu_scales_data[scale_index];

        datum.search_range.min_corner().x( scaled_search_range.min_x );
        datum.search_range.min_corner().y( scaled_search_range.min_y );
        datum.search_range.max_corner().x( scaled_search_range.max_x );
        datum.search_range.max_corner().y( scaled_search_range.max_y );

        { // we update max_search_range
            // max_search_range keeps the "bigger" range,
            // which means the smallest min_corner and the larger max_corner
            max_search_range.min_corner().x(
                        std::min(max_search_range.min_corner().x(), datum.search_range.min_corner().x() ));
            max_search_range.min_corner().y(
                        std::min(max_search_range.min_corner().y(), datum.search_range.min_corner().y() ));
            max_search_range.max_corner().x(
                        std::max(max_search_range.max_corner().x(), datum.search_range.max_corner().x() ));
            max_search_range.max_corner().y(
                        std::max(max_search_range.max_corner().y(), datum.search_range.max_corner().y() ));


            max_search_range_width = std::max(max_search_range_width,
                                              scaled_search_range.max_x - scaled_search_range.min_x);
            max_search_range_height = std::max(max_search_range_height,
                                               scaled_search_range.max_y - scaled_search_range.min_y);

        }

        datum.stride.x( scale_data.stride.x() );
        datum.stride.y( scale_data.stride.y() );

        if(false)
        {
            printf("Scale index %zi, search_range.max_corner().y() == %i\n",
                   scale_index, datum.search_range.max_corner().y());
        }

    } // end of "for each scale index"

    if(false)
    {
        printf("GpuVeryFastIntegralChannelsDetector::set_gpu_scales_data max_search_range_width/height == %i, %i\n",
               max_search_range_width, max_search_range_height);
    }


    // move data to GPU
    if(gpu_scales_data.getNumElements() != num_scales)
    {  // lazy allocations
        gpu_scales_data.alloc(num_scales);
    }
    Cuda::copy(gpu_scales_data, cpu_scales_data);

    return;
}


void GpuVeryFastIntegralChannelsDetector::set_gpu_stixels()
{
    assert((estimated_stixels.size() + 2*additional_border) == get_input_width());

    const int
            // shrinking_factor is 1, 2 or 4
            shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    // we have as many stixels as columns in the shrunk image
    const size_t num_gpu_stixels = get_input_width() / shrinking_factor;

    Cuda::HostMemoryHeap1D<gpu_stixel_t> cpu_stixels(num_gpu_stixels);

    const size_t
            border = std::max<size_t>(0, additional_border),
            shrunk_border = border / shrinking_factor,
            shrunk_image_width = (border*2+get_input_width()) / shrinking_factor;

    // we set the left area
    for(size_t u = 0; u < shrunk_border; u+=1)
    {
        cpu_stixel_to_gpu_stixel(estimated_stixels.front(), cpu_stixels[u]);
    } // end of "for each image column, in the left area"


    // we the area on the original image
    for(size_t u = shrunk_border; u < (shrunk_image_width - shrunk_border); u+=1)
    {

        Stixel aggregated_stixel;
        {
            const size_t stixel_index = u*shrinking_factor;
            aggregated_stixel = estimated_stixels[stixel_index + 0];
            bool all_occluded = (aggregated_stixel.type == Stixel::Occluded);
            for(int i = 1; i < shrinking_factor; i += 1)
            {
                const Stixel &t_stixel = estimated_stixels[stixel_index + i];
                all_occluded &= (t_stixel.type == Stixel::Occluded);

                aggregated_stixel.top_y = std::min(aggregated_stixel.top_y, t_stixel.top_y);
                aggregated_stixel.bottom_y = std::max(aggregated_stixel.bottom_y, t_stixel.bottom_y);

            } // end of "for each stixel that overlaps a shrunk pixel"

            if(not all_occluded)
            {
                aggregated_stixel.type = Stixel::Pedestrian;
            }
        }

        cpu_stixel_to_gpu_stixel(aggregated_stixel, cpu_stixels[u]);

    } // end of "for each image column, in the center area"


    // we set the right area
    for(size_t u = (shrunk_image_width - shrunk_border); u < shrunk_image_width; u+=1)
    {
        cpu_stixel_to_gpu_stixel(estimated_stixels.back(), cpu_stixels[u]);
    } // end of "for each image column, in the right area"


    if(gpu_stixels.getNumElements() != num_gpu_stixels)
    {  // lazy allocations
        gpu_stixels.alloc(num_gpu_stixels);
    }
    Cuda::copy(gpu_stixels, cpu_stixels);
    return;
}


void GpuVeryFastIntegralChannelsDetector::cpu_stixel_to_gpu_stixel(const Stixel &cpu_stixel, gpu_stixel_t &gpu_stixel)
{
    // in the GPU the x,y coordinates correspond to upper, left corner
    // FIXME for stixel+very fast detections, we may want to change this to be centered in the detection window
    // (big change, is it worth it ?)

    const int
            stixel_center_y = (cpu_stixel.top_y + cpu_stixel.bottom_y) / 2,
            stixel_height = cpu_stixel.bottom_y - cpu_stixel.top_y,
            detection_height = stixel_height*object_to_detection_window_height_ratio,
            detection_top_y = stixel_center_y - detection_height/2,
            // shrinking_factor is 1, 2 or 4
            shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor(),
            shrunk_stixel_detection_top_y = detection_top_y / shrinking_factor,
            //shrunk_stixel_bottom_y = cpu_stixel.bottom_y / shrinking_factor,
            shrunk_image_height = get_input_height() / shrinking_factor,
            shrunk_vertical_margin = stixels_vertical_margin / shrinking_factor;

    // set the top stixel limit
    if(cpu_stixel.type == Stixel::Occluded)
    {
        // the bottom of the stixel is constrained, but the top is not (will use only ground plane top constraints)
        gpu_stixel.min_y = 0;
    }
    else
    {
        gpu_stixel.min_y = std::max(0, shrunk_stixel_detection_top_y - shrunk_vertical_margin);
    }

    // set the bottom limit (still relative to the upper, left detection corner)
    gpu_stixel.max_y = std::max(0,
                                std::min(shrunk_image_height, shrunk_stixel_detection_top_y + shrunk_vertical_margin));

    const size_t max_scale_index = search_ranges_data.size() - 1;

    const bool use_stixel_scale_range = (stixels_scales_margin > 0);
    if(use_stixel_scale_range)
    {

        const bool use_vertical_margin_for_scales_range = false;
        if(use_vertical_margin_for_scales_range)
        {
            const int
                    min_stixel_height = detection_height - 2*stixels_vertical_margin,
                    max_stixel_height = detection_height + 2*stixels_vertical_margin;

            gpu_stixel.min_scale_index = get_scale_index_from_height(min_stixel_height);
            gpu_stixel.reference_scale_index = get_scale_index_from_height(detection_height);
            gpu_stixel.max_scale_index = get_scale_index_from_height(max_stixel_height);
        }
        else
        {
            const int reference_scale = get_scale_index_from_height(detection_height);

            gpu_stixel.min_scale_index = std::max(0, reference_scale - stixels_scales_margin);
            gpu_stixel.reference_scale_index = reference_scale;
            gpu_stixel.max_scale_index = std::min<int>(max_scale_index, reference_scale + stixels_scales_margin);
        }

        //printf("gpu_stixel min/max scale_index == %zi / %zi\n", gpu_stixel.min_scale_index, gpu_stixel.max_scale_index);
    }
    else
    {
        gpu_stixel.min_scale_index = 0;
        gpu_stixel.reference_scale_index = max_scale_index / 2;
        gpu_stixel.max_scale_index = max_scale_index;
    }

    return;
}


size_t GpuVeryFastIntegralChannelsDetector::get_scale_index_from_height(const float height)
{
    if(height <= 0)
    {
        // we assume the first scale is the smallest
        return 0;
    }
    const float scale = height / scale_one_detection_window_size.y();
    const int scale_index = (std::log(scale) - log_min_detection_window_scale) / scale_logarithmic_step;

    const size_t max_scale_index = search_ranges_data.size() - 1;
    return std::min<size_t>(std::max<int>(0, scale_index), max_scale_index);
}


void GpuVeryFastIntegralChannelsDetector::set_gpu_half_window_widths()
{

    if(search_ranges_data.empty())
    {
        throw std::runtime_error("GpuVeryFastIntegralChannelsDetector::set_gpu_half_window_widths received "
                                 "search_ranges.empty() == true");
    }

    const size_t num_scales = search_ranges_data.size();
    Cuda::HostMemoryHeap1D<gpu_half_window_widths_t::Type> cpu_half_window_widths(num_scales);

    // shrinking_factor is 1, 2 or 4
    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    for(size_t scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        const ScaleData &scale_data = extra_data_per_scale[scale_index];

        cpu_half_window_widths[scale_index] = scale_data.scaled_detection_window_size.x() / 2;
        cpu_half_window_widths[scale_index] /= shrinking_factor; // FIXME

        //printf("CPU scale index %zi, cpu_half_window_widths == %i\n",
        //       scale_index, cpu_half_window_widths[scale_index]);

    } // end of "for each scale index"

    // move data to GPU
    if(gpu_half_window_widths.getNumElements() != num_scales)
    {  // lazy allocations
        gpu_half_window_widths.alloc(num_scales);
    }
    Cuda::copy(gpu_half_window_widths, cpu_half_window_widths);

    return;
}


void print_gpu_scales_data(GpuVeryFastIntegralChannelsDetector::gpu_scales_data_t &gpu_scales_data )
{
    static bool first_call = true;

    if(not first_call)
    {
        return;
    }

    typedef GpuVeryFastIntegralChannelsDetector::gpu_scale_datum_t gpu_scale_datum_t;
    const size_t num_scales = gpu_scales_data.getNumElements();
    Cuda::HostMemoryHeap1D<gpu_scale_datum_t> cpu_scales_data(num_scales);

    Cuda::copy(cpu_scales_data, gpu_scales_data);

    for(size_t scale_index = 0; scale_index < num_scales; scale_index += 1)
    {
        const gpu_scale_datum_t &datum = cpu_scales_data[scale_index];

        const int corridor_height = datum.search_range.max_corner().y() - datum.search_range.min_corner().y();

        printf("Scale index %zi, corridor_height == %i\n",
               scale_index, corridor_height);

    } // end of "for each scale"

    first_call = false;
    return;
}




class invoke_integral_channels_detector_over_all_scales: public boost::static_visitor<void>
{
protected:

    doppia::objects_detection::gpu_integral_channels_t &integral_channels;
    doppia::objects_detection::gpu_scale_datum_t::search_range_t &max_search_range;
    const int max_search_range_width;
    const int max_search_range_height;
    doppia::objects_detection::gpu_scales_data_t &scales_data;
    //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale;
    const float score_threshold;
    const bool use_the_model_cascade;
    doppia::objects_detection::gpu_detections_t& gpu_detections;
    size_t &num_detections;

public:

    invoke_integral_channels_detector_over_all_scales(
            doppia::objects_detection::gpu_integral_channels_t &integral_channels_,
            doppia::objects_detection::gpu_scale_datum_t::search_range_t &max_search_range_,
            const int max_search_range_width_, const int max_search_range_height_,
            doppia::objects_detection::gpu_scales_data_t &scales_data_,
            //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
            const float score_threshold_,
            const bool use_the_model_cascade_,
            doppia::objects_detection::gpu_detections_t& gpu_detections_,
            size_t &num_detections_)
        : integral_channels(integral_channels_),
          max_search_range(max_search_range_),
          max_search_range_width(max_search_range_width_),
          max_search_range_height(max_search_range_height_),
          scales_data(scales_data_),
          //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
          score_threshold(score_threshold_),
          use_the_model_cascade(use_the_model_cascade_),
          gpu_detections(gpu_detections_),
          num_detections(num_detections_)
    {
        // nothing to do here
        return;
    }

    template<typename T>
    void operator()(T &gpu_detection_cascade_per_scale) const;

protected:

    template<typename T>
    void compute(T &gpu_detection_cascade_per_scale) const;

}; // end of visitor class invoke_v0_integral_channels_detector



template<typename T>
void invoke_integral_channels_detector_over_all_scales::operator()(T &gpu_detection_cascade_per_scale) const
{
    // gpu_detection_stump_cascade_per_scale_t (and others)
    throw std::runtime_error("integral_channels_detector_over_all_scales with a gpu detection cascade per scale type "
                             "that is not yet managed");
    return;
}


template<typename T>
inline
void invoke_integral_channels_detector_over_all_scales::compute(T &gpu_detection_cascade_per_scale) const
{
    doppia::objects_detection::integral_channels_detector_over_all_scales(
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                gpu_detection_cascade_per_scale,
                score_threshold, use_the_model_cascade,
                gpu_detections, num_detections);
    return;
}


template<>
void invoke_integral_channels_detector_over_all_scales::operator()
<GpuIntegralChannelsDetector::gpu_detection_cascade_per_scale_t>
(GpuIntegralChannelsDetector::gpu_detection_cascade_per_scale_t &gpu_detection_cascade_per_scale) const
{
    compute(gpu_detection_cascade_per_scale);
    return;
}
/*
template<>
void invoke_integral_channels_detector_over_all_scales::operator()
<GpuIntegralChannelsDetector::gpu_detection_stump_cascade_per_scale_t>
(GpuIntegralChannelsDetector::gpu_detection_stump_cascade_per_scale_t &gpu_detection_cascade_per_scale) const
{
    compute(gpu_detection_cascade_per_scale);
    return;
}*/


template<>
void invoke_integral_channels_detector_over_all_scales::operator()
<GpuIntegralChannelsDetector::gpu_fractional_detection_cascade_per_scale_t>
(GpuIntegralChannelsDetector::gpu_fractional_detection_cascade_per_scale_t &gpu_detection_cascade_per_scale) const
{
    compute(gpu_detection_cascade_per_scale);
    return;
}



class invoke_integral_channels_detector_over_all_scales_with_stixels: public boost::static_visitor<void>
{
protected:

    doppia::objects_detection::gpu_integral_channels_t &integral_channels;
    doppia::objects_detection::gpu_scale_datum_t::search_range_t &max_search_range;
    const int max_search_range_width;
    const int max_search_range_height;
    const int num_scales_to_evaluate;
    doppia::objects_detection::gpu_scales_data_t &scales_data;
    doppia::objects_detection::gpu_stixels_t &stixels;
    doppia::objects_detection::gpu_half_window_widths_t &gpu_half_window_widths;
    //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale;
    const float score_threshold;
    const bool use_the_model_cascade;
    doppia::objects_detection::gpu_detections_t& gpu_detections;
    size_t &num_detections;

public:

    invoke_integral_channels_detector_over_all_scales_with_stixels(
            doppia::objects_detection::gpu_integral_channels_t &integral_channels_,
            doppia::objects_detection::gpu_scale_datum_t::search_range_t &max_search_range_,
            const int max_search_range_width_, const int max_search_range_height_,
            const int num_scales_to_evaluate_,
            doppia::objects_detection::gpu_scales_data_t &scales_data_,
            doppia::objects_detection::gpu_stixels_t &stixels_,
            doppia::objects_detection::gpu_half_window_widths_t &gpu_half_window_widths_,
            //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
            const float score_threshold_,
            const bool use_the_model_cascade_,
            doppia::objects_detection::gpu_detections_t& gpu_detections_,
            size_t &num_detections_)
        : integral_channels(integral_channels_),
          max_search_range(max_search_range_),
          max_search_range_width(max_search_range_width_),
          max_search_range_height(max_search_range_height_),
          num_scales_to_evaluate(num_scales_to_evaluate_),
          scales_data(scales_data_),
          stixels(stixels_),
          gpu_half_window_widths(gpu_half_window_widths_),
          //gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
          score_threshold(score_threshold_),
          use_the_model_cascade(use_the_model_cascade_),
          gpu_detections(gpu_detections_),
          num_detections(num_detections_)
    {
        // nothing to do here
        return;
    }


    template<typename T>
    void operator()(T &gpu_detection_cascade_per_scale) const;

protected:

    template<typename T>
    void compute(T &gpu_detection_cascade_per_scale) const;

}; // end of visitor class invoke_v0_integral_channels_detector



template<typename T>
void invoke_integral_channels_detector_over_all_scales_with_stixels::operator()(T &gpu_detection_cascade_per_scale) const
{
    // gpu_detection_stump_cascade_per_scale_t, gpu_fractional_detection_cascade_per_scale_t
    // (and others)
    throw std::runtime_error("invoke_integral_channels_detector_over_all_scales_with_stixels "
                             "with a gpu detection cascade per scale type "
                             "that is not yet managed");
    return;
}


template<typename T>
inline
void invoke_integral_channels_detector_over_all_scales_with_stixels::compute(T &gpu_detection_cascade_per_scale) const
{
    doppia::objects_detection::integral_channels_detector_over_all_scales(
                integral_channels,
                max_search_range,
                max_search_range_width, max_search_range_height,
                num_scales_to_evaluate,
                scales_data,
                stixels, gpu_half_window_widths,
                gpu_detection_cascade_per_scale, score_threshold, use_the_model_cascade,
                gpu_detections, num_detections);
    return;
}


template<>
void invoke_integral_channels_detector_over_all_scales_with_stixels::operator()
<GpuIntegralChannelsDetector::gpu_detection_cascade_per_scale_t>
(GpuIntegralChannelsDetector::gpu_detection_cascade_per_scale_t &gpu_detection_cascade_per_scale) const
{
    compute(gpu_detection_cascade_per_scale);
    return;
}


void GpuVeryFastIntegralChannelsDetector::compute_v2()
{

#if defined(BOOTSTRAPPING_LIB)
    throw std::runtime_error("GpuVeryFastIntegralChannelsDetector::compute_v2 "
                             "should not be used inside bootstrapping_lib, "
                             "use GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v0 instead");
#endif

    // some debugging variables
    static bool first_call = true;

    assert(integral_channels_computer_p);
    //assert(gpu_detection_variant_cascade_per_scale.getBuffer() != NULL);


    if(first_call)
    {
        printf("GpuVeryFastIntegralChannelsDetector::compute_v2 " \
               "max search range (min_x, min_y; max_x, max_y) == (%i, %i; %i, %i)\n",
               max_search_range.min_corner().x(),  max_search_range.min_corner().y(),
               max_search_range.max_corner().x(),  max_search_range.max_corner().y());
    }
    //print_gpu_scales_data(gpu_scales_data); // just for debugging

    const int num_calls_for_timing = 250;
    static int num_calls = 0;
    static double cumulated_integral_channels_computation_time = 0;

    detections.clear();
    num_gpu_detections = 0; // no need to clean the buffer

    {
        double start_wall_time = omp_get_wtime();

        // we only need to compute the integral images for one image size,
        // since all other scales will have the same resized size
        const size_t search_range_index = 0;
        doppia::objects_detection::gpu_integral_channels_t &integral_channels =
                resize_input_and_compute_integral_channels(search_range_index, first_call);

        cumulated_integral_channels_computation_time += omp_get_wtime() - start_wall_time;

        // compute the detections, and keep the results on the gpu memory --
        if(estimated_stixels.empty())
        {
            invoke_integral_channels_detector_over_all_scales visitor(
                        integral_channels,
                        max_search_range, max_search_range_width, max_search_range_height,
                        gpu_scales_data,
                        score_threshold, use_the_detector_model_cascade,
                        gpu_detections, num_gpu_detections);

            boost::apply_visitor(visitor, gpu_detection_variant_cascade_per_scale);
        }
        else
        { // stixels available for the detection
            const int num_scales_to_evaluate =
                    (stixels_scales_margin > 0)? 2*stixels_scales_margin: search_ranges_data.size();

            invoke_integral_channels_detector_over_all_scales_with_stixels visitor(
                        integral_channels,
                        max_search_range,  max_search_range_width, stixels_max_search_range_height,
                        num_scales_to_evaluate,
                        gpu_scales_data,
                        gpu_stixels, gpu_half_window_widths,
                        score_threshold, use_the_detector_model_cascade,
                        gpu_detections, num_gpu_detections);

            boost::apply_visitor(visitor, gpu_detection_variant_cascade_per_scale);
        }

        // retrieve the gpu detections --
        collect_the_gpu_detections();
    }

    //const bool print_num_detections = true;
    const bool print_num_detections = false;
    if(print_num_detections)
    {
        log.info() << "number of raw (before non maximal suppression) detections on this frame == "
                   << detections.size() << std::endl;
    }


    num_calls += 1;
    if(true and ((num_calls % num_calls_for_timing) == 0))
    {
        printf("Average integral channels features computation speed %.4lf [Hz] (in the last %i iterations)\n",
               num_calls / cumulated_integral_channels_computation_time, num_calls );
    }

    //recenter_detections(detections); // FIXME just for testing

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    compute_non_maximal_suppresion();

    first_call = false;
    return;
}



} // end of namespace doppia
