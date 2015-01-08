#include "VeryFastDetectorScaleStatisticsApplication.hpp"

#include "VeryFastDetectorScaleStatisticsApplication.hpp"

#include "objects_detection/ObjectsDetectorFactory.hpp"
#include "objects_detection/VeryFastIntegralChannelsDetector.hpp"

#include "video_input/ImagesFromDirectory.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/for_each_multi_array.hpp"

#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>

#include <boost/test/test_tools.hpp>

#include <list>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace doppia {

using namespace boost;

typedef AbstractObjectsDetector::detection_t detection_t;


AccumulatorsPerDeltaScale::AccumulatorsPerDeltaScale()
    : accumulator_one(),
      very_low_quantile(quantile_probability=0.01),
      low_quantile(quantile_probability=0.05),
      high_quantile(quantile_probability=0.95),
      very_high_quantile(quantile_probability=0.99)
{
    // nothing to do here
    return;
}

void AccumulatorsPerDeltaScale::add_sample(const float sample_value)
{
    accumulator_one(sample_value);
    very_low_quantile(sample_value);
    low_quantile(sample_value);
    high_quantile(sample_value);
    very_high_quantile(sample_value);
    return;
}


std::string VeryFastDetectorScaleStatisticsApplication::get_application_title()
{
    return "very_fast_detector_scale_statistics_test_application";
}


VeryFastDetectorScaleStatisticsApplication::VeryFastDetectorScaleStatisticsApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}

VeryFastDetectorScaleStatisticsApplication::~VeryFastDetectorScaleStatisticsApplication()
{
    // nothing to do here
    return;
}


/// helper method used by the user interfaces when recording screenshots
/// this number is expected to change with that same frequency that update_gui is called
int VeryFastDetectorScaleStatisticsApplication::get_current_frame_number() const
{
    // no need to provide true numbers
    return 0;
}

void VeryFastDetectorScaleStatisticsApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(BaseApplication::get_options_description(get_application_title()));

    desc.add(ObjectsDetectorFactory::get_args_options());

    //desc.add(AbstractObjectsDetector::get_args_options());
    //desc.add(IntegralChannelsDetector::get_args_options());
    //desc.add(VeryFastIntegralChannelsDetector::get_args_options());


    //desc.add(ObjectsDetectionApplication::get_args_options());
    //desc.add(ObjectsDetectionGui::get_args_options());
    //desc.add(VideoInputFactory::get_args_options());

    // Subset of the options from ObjectsDetectionApplication
    desc.add_options()

            ("additional_border",
             program_options::value<int>()->default_value(0),
             "when using process_folder, will add border to the image to enable detection of cropped pedestrians. "
             "Value is in pixels (e.g. 50 pixels)")

            ;

    // Subset of ObjectsDetectionApplication
    desc.add_options()
            ("process_folder",
             program_options::value<string>(),
             "for evaluation purposes, will process images on a folder. Normal video input will be ignored")
            ;

    return;
}


void VeryFastDetectorScaleStatisticsApplication::setup_problem(const program_options::variables_map &options)
{
    const filesystem::path folder_to_process = get_option_value<string>(options, "process_folder");
    directory_input_p.reset(new ImagesFromDirectory(folder_to_process));

    AbstractObjectsDetector *abstract_detector_p = ObjectsDetectorFactory::new_instance(options);

    VeryFastIntegralChannelsDetector *very_fast_detector_p =
            dynamic_cast<VeryFastIntegralChannelsDetector *>(abstract_detector_p);

    if(very_fast_detector_p == NULL)
    {
        delete abstract_detector_p;
        throw std::runtime_error("VeryFastDetectorScaleStatisticsApplication expects ObjectsDetectorFactory "
                                 "to create a detector of type VeryFastIntegralChannelsDetector");
    }

    objects_detector_p.reset(very_fast_detector_p);

    return;
}



AbstractGui* VeryFastDetectorScaleStatisticsApplication::create_gui(const program_options::variables_map &/*options*/)
{
    // no gui
    return NULL;
}


void VeryFastDetectorScaleStatisticsApplication::main_loop()
{

    int num_iterations = 0;
    const int num_iterations_for_timing = 10;
    double cumulated_processing_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = false;
    video_input_is_available = directory_input_p->next_frame();

    // for each input image
    while(video_input_is_available)
    {
        // update video input --
        const AbstractVideoInput::input_image_view_t &
                input_view = directory_input_p->get_image();

        const double start_processing_wall_time = omp_get_wtime();

        printf("Processing image %s\n", directory_input_p->get_image_name().c_str());

        // compute the channel responses and comparison statistics --
        process_frame(input_view);

        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        // print timing information --
        num_iterations += 1;
        if((num_iterations % num_iterations_for_timing) == 0)
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        // retrieve the next test image --
        video_input_is_available = directory_input_p->next_frame();

    } // end of "while video input"

    // print global timing information --
    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average objects detection speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    // compute test results --
    printf("\n");
    save_computed_statistics();
    save_computed_centered_statistics();

    return;
}


void VeryFastDetectorScaleStatisticsApplication::save_computed_statistics() const
{
    // save the per_scale_per_channel_statistics in a format compatible with
    // http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt
    const string statistics_filename = "very_fast_detector_scale_statistics.txt";
    FILE *fout = fopen(statistics_filename.c_str(), "w");

    fprintf(fout,
            "# The columns are: "
            "delta_scale, delta_score_mean, delta_score_min, delta_score_max, "
            "delta_score_1%%, delta_score_5%%, delta_score_50%%, delta_score_95%%,  delta_score_99%%, count\n");


    assert(objects_detector_p->num_scales > 0);
    const float
            scale_logarithmic_step =
            (log(objects_detector_p->max_detection_window_scale)
             - log(objects_detector_p->min_detection_window_scale)) / (objects_detector_p->num_scales -1);

    // we skip the "zero delta" row, thus scale_index = 1
    for(size_t scale_index=1; scale_index < statistics_per_delta_scale.size(); scale_index +=1)
    {
        const AccumulatorsPerDeltaScale &statistics = statistics_per_delta_scale[scale_index];

        const size_t count = accumulators::count(statistics.accumulator_one);
        if(count == 0)
        {
            // no data points collected here, skipping
            continue;
        }

        const float delta_scale = exp(scale_index*scale_logarithmic_step);

        fprintf(fout, "%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %zi\n",
                delta_scale,
                accumulators::mean(statistics.accumulator_one),
                accumulators::min(statistics.accumulator_one),
                accumulators::max(statistics.accumulator_one),
                accumulators::p_square_quantile(statistics.very_low_quantile),
                accumulators::p_square_quantile(statistics.low_quantile),
                accumulators::median(statistics.accumulator_one),
                accumulators::p_square_quantile(statistics.high_quantile),
                accumulators::p_square_quantile(statistics.very_high_quantile),
                count);
    } // end of "for each scale"

    fclose(fout);

    printf("Created file %s\n", statistics_filename.c_str());
    return;
}

void VeryFastDetectorScaleStatisticsApplication::save_computed_centered_statistics() const
{
    const string statistics_filename = "very_fast_detector_scale_centered_statistics.txt";
    FILE *fout = fopen(statistics_filename.c_str(), "w");

    fprintf(fout,
            "# The columns are: "
            "delta_scale, delta_score_mean, delta_score_min, delta_score_max, "
            "delta_score_1%%, delta_score_5%%, delta_score_50%%, delta_score_95%%,  delta_score_99%%, count\n");


    assert(objects_detector_p->num_scales > 0);
    const float
            scale_logarithmic_step =
            (log(objects_detector_p->max_detection_window_scale)
             - log(objects_detector_p->min_detection_window_scale)) / (objects_detector_p->num_scales -1);

    const size_t num_scales = statistics_per_delta_scale.size();

    for(size_t scale_index=0; scale_index < centered_statistics_per_delta_scale.size(); scale_index +=1)
    {
        const AccumulatorsPerDeltaScale &statistics = centered_statistics_per_delta_scale[scale_index];

        const int delta_scale_index = scale_index - num_scales;
        const float
                sign = (delta_scale_index >= 0)? 1 : -1,
                delta_scale = sign*exp(abs(delta_scale_index)*scale_logarithmic_step);

        const size_t count = accumulators::count(statistics.accumulator_one);
        if(count == 0)
        {
            // no data points collected here, skipping
            printf("Skipping scale index %zi (delta_scale == %.3f) because got zero samples there\n",
                   scale_index, delta_scale);
            continue;
        }


        fprintf(fout, "%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %zi\n",
                delta_scale,
                accumulators::mean(statistics.accumulator_one),
                accumulators::min(statistics.accumulator_one),
                accumulators::max(statistics.accumulator_one),
                accumulators::p_square_quantile(statistics.very_low_quantile),
                accumulators::p_square_quantile(statistics.low_quantile),
                accumulators::median(statistics.accumulator_one),
                accumulators::p_square_quantile(statistics.high_quantile),
                accumulators::p_square_quantile(statistics.very_high_quantile),
                count);
    } // end of "for each delta scale"


    fclose(fout);

    printf("Created file %s\n", statistics_filename.c_str());
    return;
}


void VeryFastDetectorScaleStatisticsApplication::update_statistics(
        const multi_array<float, 2> &scores_per_pixel,
        std::vector<AccumulatorsPerDeltaScale> &statistics_per_delta_scale) const
{

    if(scores_per_pixel.shape()[1] != statistics_per_delta_scale.size())
    {
        throw std::runtime_error("scores_per_pixel.shape()[1] != statistics_per_delta_scale.size()");
    }

    const size_t num_pixels = scores_per_pixel.shape()[0];

    printf("Updating the statistics with the image data of %zi pixels\n", num_pixels);
    boost::progress_display progress(num_pixels);

    for(size_t pixel_index=0; pixel_index < num_pixels; pixel_index +=1)
    {
        multi_array<float, 2>::const_reference pixel_scores = scores_per_pixel[pixel_index];


        for(size_t delta_scale_index = 0; delta_scale_index < statistics_per_delta_scale.size(); delta_scale_index +=1)
        {
            AccumulatorsPerDeltaScale &accumulators = statistics_per_delta_scale[delta_scale_index];

            // map from scores to delta scores
            for(size_t a=0, b=delta_scale_index; b < pixel_scores.size(); a+=1, b+=1)
            {
                const float
                        score_a = pixel_scores[a],
                        score_b = pixel_scores[b];

                //const bool check_min_score = true;
                const bool check_min_score = false;
                if(check_min_score)
                {
                    //const float min_score = 0;
                    const float min_score = -0.05;
                    //const float min_score = -0.2;
                    if((score_a < min_score) or (score_b < min_score))
                    {
                        continue;
                    }
                }

                float delta_score = 0;
                const bool use_fraction = false;
                if(use_fraction)
                {
                    //const float min_abs_score = 0;
                    //const float min_abs_score = 0.001;
                    //const float min_abs_score = 0.01;
                    //if((abs(score_a) <= min_abs_score) or (abs(score_b) <= min_abs_score))
                    //{ continue; }
                    //else
                    {
                        delta_score = score_a / score_b;
                    }
                }
                else
                {
                    delta_score = score_a - score_b;
                    if(delta_score == 0)
                    {
                        // we try to make things a little bit faster, we skip the non-interesting bits
                        continue;
                    }
                }

                accumulators.add_sample(delta_score);
            }

        } // end of "for each delta scale"

        ++progress;
    } // end of "for each pixel"

    return;
}


void VeryFastDetectorScaleStatisticsApplication::update_centered_statistics(
        const boost::multi_array<float, 2> &scores_per_pixel,
        std::vector<AccumulatorsPerDeltaScale> &centered_statistics_per_delta_scale) const
{
    const size_t num_scales = scores_per_pixel.shape()[1];

    if((num_scales*2 + 1) != centered_statistics_per_delta_scale.size())
    {
        throw std::runtime_error("centered_statistics_per_delta_scale.size() does not have the expected value");
    }

    // this score threshold allows the multiscale models to reach the 1 FPPI line in the INRIA dataset
    const float score_threshold = 0.01;

    const size_t num_pixels = scores_per_pixel.shape()[0];

    printf("Updating the centered statistics with the image data of %zi pixels\n", num_pixels);
    boost::progress_display progress(num_pixels);

    for(size_t pixel_index=0; pixel_index < num_pixels; pixel_index +=1)
    {
        multi_array<float, 2>::const_reference pixel_scores = scores_per_pixel[pixel_index];

        float max_score = pixel_scores[0];
        size_t max_score_scale_index = 0;

        for(size_t scale_index = 0; scale_index < pixel_scores.size(); scale_index+=1)
        {
            const float &score = pixel_scores[scale_index];

            if(score > max_score)
            {
                max_score = score;
                max_score_scale_index = scale_index;
            }
        } // end of "for each scale"


        if(max_score >= score_threshold)
        { // found a valid detection

            for(size_t scale_index = 0; scale_index < pixel_scores.size(); scale_index+=1)
            {

                const size_t centered_index = num_scales + scale_index - max_score_scale_index;
                AccumulatorsPerDeltaScale &accumulators = centered_statistics_per_delta_scale[centered_index];

                const float &score = pixel_scores[scale_index];
                accumulators.add_sample(score);
            } // end of "for each scale"

        }
        else
        {
            // nothing to do, we skip this pixel
        }

        ++progress;
    } // end of "for each pixel"

    return;
}



/// Helper function that saves the best score and the
void save_some_scores(const multi_array<float, 2> &scores_per_pixel)
{

    // save the per_scale_per_channel_statistics in a format compatible with
    // http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt
    FILE *fout = fopen("some_pixel_scores.txt", "w");
    fprintf(fout, "# The columns are indexed by scale index, the rows by pixel\n");

    const size_t
            num_scales = scores_per_pixel.shape()[1],
            num_pixels = scores_per_pixel.shape()[0];

    // we search the max score --
    {
        size_t max_score_pixel_index = 0;
        float max_score = scores_per_pixel[0][0];
        for(size_t pixel_index = 0; pixel_index < num_pixels; pixel_index +=1)
        {
            for(size_t scale_index = 0; scale_index < num_scales; scale_index +=1)
            {
                const float score = scores_per_pixel[pixel_index][scale_index];
                if(score > max_score)
                {
                    max_score = score;
                    max_score_pixel_index = pixel_index;
                }

            } // end of "for each scale"

        } // end of "for each pixel"

        // we save the max score pixel --
        for(size_t scale_index = 0; scale_index < num_scales; scale_index +=1)
        {
            const float score = scores_per_pixel[max_score_pixel_index][scale_index];
            fprintf(fout, "%.5f ", score);
        } // end of "for each scale"
        fprintf(fout, "\n");
    }

    // we save high score detections --
    {
        //const float score_threshold = 0.001;

        for(size_t pixel_index = 0; pixel_index < num_pixels; pixel_index +=1)
        {
            //bool found_detection = false;
            float max_score = scores_per_pixel[pixel_index][0];
            size_t max_score_pixel_index = 0;

            for(size_t scale_index = 0; scale_index < num_scales; scale_index +=1)
            {
                const float score = scores_per_pixel[pixel_index][scale_index];
                if(score >= max_score)
                {
                    max_score = score;
                    max_score_pixel_index = pixel_index;
                }

            } // end of "for each scale"

            for(size_t scale_index = 0; scale_index < num_scales; scale_index +=1)
            {
                const float score = scores_per_pixel[max_score_pixel_index][scale_index];
                fprintf(fout, "%.5f ", score);
            } // end of "for each scale"
            fprintf(fout, "\n");

        } // end of "for each pixel"

    }


    // we save some random pixels --
    {
        boost::mt19937 random_generator;
        boost::uniform_int<size_t> random_index(0, (num_pixels - 1));

        const int num_random_pixels = 10;

        for(int c=0; c < num_random_pixels; c+=1)
        {
            const size_t random_pixel_index = random_index(random_generator);

            for(size_t scale_index = 0; scale_index < num_scales; scale_index +=1)
            {
                const float score = scores_per_pixel[random_pixel_index][scale_index];
                fprintf(fout, "%.5f ", score);
            } // end of "for each scale"
            fprintf(fout, "\n");
        } // end of "for each random pixel"

    }

    fclose(fout);
    throw std::runtime_error("Stopping everything so you can inspect the some_pixel_scores.txt file");
    return;
}

// underline at the end to avoid double definition, too lazy to move to an anonymous namespace
// waiting to use the cool C++11 [](const detection_t &a, const detection_t &b) -> bool {a.score > b.score }
bool has_higher_score_(const detection_t &a, const detection_t &b)
{
    return a.score > b.score;
}


void save_best_detections(const AbstractObjectsDetector::detections_t &detections)
{

    if(detections.empty())
    {
        throw std::invalid_argument("save_best_detections received detections.empty()");
    }

    // we find the best detection --
    typedef AbstractObjectsDetector::detection_t detection_t;
    typedef std::list<detection_t> candidate_detections_t;
    candidate_detections_t candidate_detections(detections.begin(), detections.end());
    candidate_detections.sort(has_higher_score_);

    const detection_t::rectangle_t &best_box = candidate_detections.front().bounding_box;

    const float
            best_box_x = (best_box.max_corner().x() + best_box.min_corner().x())/2.0f,
            best_box_y = (best_box.max_corner().y() + best_box.min_corner().y())/2.0f;

    // we remove all detections that are "too far" --
    const float max_center_distance = 5; // [pixels]
    size_t num_removed_items = 0;
    candidate_detections_t::iterator detections_it = candidate_detections.begin();
    while(detections_it != candidate_detections.end())
    {
        const detection_t::rectangle_t &box = detections_it->bounding_box;
        const float
                box_x = (box.max_corner().x() + box.min_corner().x())/2.0f,
                box_y = (box.max_corner().y() + box.min_corner().y())/2.0f,
                delta_x = box_x - best_box_x,
                delta_y = box_y - best_box_y,
                distance = std::sqrt((delta_x*delta_x) + (delta_y*delta_y));

        const bool box_is_too_far = (distance >= max_center_distance);
        if(box_is_too_far)
        { // we remove detections that are too far from the best one
            detections_it = candidate_detections.erase(detections_it);
            num_removed_items += 1;
        }
        else
        {
            // not too far, nothing to do, we move to next sample
            ++detections_it;
        }
    } // end of "for each detection"

    printf("Removed %zi detections that were too far from the best one\n", num_removed_items);
    printf("Number initial detections %zi, number selected detections %zi\n",
           detections.size(), candidate_detections.size());

    // we save the data
    {

        // save the per_scale_per_channel_statistics in a format compatible with
        // http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt
        FILE *fout = fopen("best_detections.txt", "w");
        fprintf(fout, "# The columns are score, x, y, distance and the rows are detections\n");

        BOOST_FOREACH(const detection_t &detection, candidate_detections)
        {

            const detection_t::rectangle_t &box = detection.bounding_box;
            const float
                    box_x = (box.max_corner().x() + box.min_corner().x())/2.0f,
                    box_y = (box.max_corner().y() + box.min_corner().y())/2.0f,
                    delta_x = box_x - best_box_x,
                    delta_y = box_y - best_box_y,
                    distance = std::sqrt((delta_x*delta_x) + (delta_y*delta_y));

            fprintf(fout, "%.5f %.5f %.5f %.5f\n", detection.score, box_x, box_y, distance);
        } // end of "for each detection"

        fclose(fout);
        throw std::runtime_error("Stopping everything so you can inspect the best_detections.txt file");
    }

    return;
}


void VeryFastDetectorScaleStatisticsApplication::process_frame(const AbstractVideoInput::input_image_view_t &input_view)
{
    static bool first_call = true;
    const bool save_score_image = false; //, false_first_call = false;

    objects_detector_p->detections.clear();
    objects_detector_p->set_image(input_view);
    const IntegralChannelsDetector::detections_scores_t &detections_scores = objects_detector_p->detections_scores;

    BOOST_REQUIRE(objects_detector_p->search_ranges_data.empty() == false);

    // we will only consider the pixels covered by all scales
    size_t
            min_max_x = detections_scores.shape()[1],
            min_max_y =  detections_scores.shape()[0];

    for(size_t scale_index=0; scale_index < objects_detector_p->search_ranges_data.size(); scale_index +=1)
    {
        const ScaleData &scale_data = objects_detector_p->extra_data_per_scale[scale_index];
        const DetectorSearchRange &search_range = scale_data.scaled_search_range;
        BOOST_REQUIRE(search_range.min_x == 0);
        BOOST_REQUIRE(search_range.min_y == 0);
        min_max_x = std::min<size_t>(min_max_x, search_range.max_x);
        min_max_y = std::min<size_t>(min_max_y, search_range.max_y);
    }

    if(first_call)
    {
        printf("min_max_x/y == %zi, %zi\n", min_max_x, min_max_y);
    }

    // we collect all the scores in one image
    multi_array<float, 2> scores_per_pixel;

    const size_t
            num_score_pixels = min_max_x*min_max_y,
            num_scales = objects_detector_p->search_ranges_data.size();
    scores_per_pixel.resize(boost::extents[num_score_pixels][num_scales]);

    statistics_per_delta_scale.resize(num_scales); // lazy allocation
    centered_statistics_per_delta_scale.resize(2*num_scales+1);

    // for each range search
    for(size_t scale_index=0; scale_index < objects_detector_p->search_ranges_data.size(); scale_index +=1)
    {
        objects_detector_p->compute_detections_at_specific_scale(scale_index,
                                                                 save_score_image, first_call);

        const IntegralChannelsDetector::detections_scores_t &detections_scores = objects_detector_p->detections_scores;
        size_t pixel_index = 0;
        for(size_t row=0; row < min_max_y; row+=1)
        {
            for(size_t col=0; col < min_max_x; col+=1)
            {
                scores_per_pixel[pixel_index][scale_index] = detections_scores[row][col];
                pixel_index += 1;
            } // end of "for each score col"

        } // end of "for each score row"

    } // end of "for each search range"


    const bool do_save_best_detections = false;
    //const bool do_save_best_detections = true;
    if(do_save_best_detections)
    {
        save_best_detections(objects_detector_p->get_detections());
    }

    const bool do_save_some_scores = false;
    //const bool do_save_some_scores = true;
    if(do_save_some_scores)
    {
        save_some_scores(scores_per_pixel);
    }

    update_statistics(scores_per_pixel, statistics_per_delta_scale);
    update_centered_statistics(scores_per_pixel, centered_statistics_per_delta_scale);


    first_call = false;
    return;
}


} // end of namespace doppia
