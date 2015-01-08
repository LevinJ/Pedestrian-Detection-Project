#include "DetectorsComparisonTestApplication.hpp"

#include "objects_detection/ObjectsDetectorFactory.hpp"
#include "objects_detection/non_maximal_suppression/AbstractNonMaximalSuppression.hpp"
#include "objects_detection/FastestPedestrianDetectorInTheWest.hpp"
#include "objects_detection/IntegralChannelsDetector.hpp"
#include "objects_detection/SlidingIntegralFeature.hpp"
#include "objects_detection/detector_model.pb.h"

#include "video_input/ImagesFromDirectory.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/for_each_multi_array.hpp"

#include <boost/foreach.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/get.hpp>

#include <boost/test/test_tools.hpp>

#include <limits>
#include <algorithm>
#include <omp.h>

namespace doppia {

using namespace boost;

typedef DetectorsComparisonTestApplication::second_moment_accumulator_t second_moment_accumulator_t;
typedef DetectorsComparisonTestApplication::channel_statistics_accumulator_t channel_statistics_accumulator_t;
typedef DetectorsComparisonTestApplication::per_scale_per_channel_statistics_t per_scale_per_channel_statistics_t;
typedef DetectorsComparisonTestApplication::feature_values_t feature_values_t;


std::string DetectorsComparisonTestApplication::get_application_title()
{
    return "detectors_comparison_test_application";
}


DetectorsComparisonTestApplication::DetectorsComparisonTestApplication()
    : BaseApplication()
{
    global_channel_statistics.resize(10); // IntegralChannelsForPedestrians has 10 channels
    return;
}

DetectorsComparisonTestApplication::~DetectorsComparisonTestApplication()
{
    // nothing to do here
    return;
}


/// helper method used by the user interfaces when recording screenshots
/// this number is expected to change with that same frequency that update_gui is called
int DetectorsComparisonTestApplication::get_current_frame_number() const
{
    // no need to provide true numbers
    return 0;
}

void DetectorsComparisonTestApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(BaseApplication::get_options_description(get_application_title()));

    desc.add(AbstractObjectsDetector::get_args_options());
    desc.add(IntegralChannelsDetector::get_args_options());
    desc.add(FastestPedestrianDetectorInTheWest::get_args_options());


    //desc.add(ObjectsDetectionApplication::get_args_options());
    //desc.add(ObjectsDetectionGui::get_args_options());
    //desc.add(VideoInputFactory::get_args_options());

    // Subset of the options from ObjectsDetectorFactory
    desc.add_options()

            ("objects_detector.score_threshold",
             program_options::value<float>()->default_value(0.5),
             "minimum score needed to validate a detection." \
             "The score is assumed normalized across classes and models.")

            ("objects_detector.non_maximal_suppression",
             program_options::value<bool>()->default_value(true),
             "apply (greedy) non maximal supression to the detections")
            ;

    // Subset of ObjectsDetectionApplication
    desc.add_options()
            ("process_folder",
             program_options::value<string>(),
             "for evaluation purposes, will process images on a folder. Normal video input will be ignored")
            ;

    return;
}

/// Creates a model where there first feature of each stage is of size 64x128 and
void create_channels_test_model_instance(boost::shared_ptr<doppia_protobuf::DetectorModel> &detector_model_p)
{

    detector_model_p.reset(new doppia_protobuf::DetectorModel());

    detector_model_p->set_detector_name("Synthetic detector from DetectorsComparisonTestApplication "
                                        "for testing the channels scaling");
    detector_model_p->set_training_dataset_name("no data was used");
    detector_model_p->set_detector_type(doppia_protobuf::DetectorModel::SoftCascadeOverIntegralChannels);

    const int shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

    // FIXME 64x128 is a hardcoded default, should be put in the test model
    const int feature_width = 64/shrinking_factor, feature_height = 128/shrinking_factor;

    const int num_channels = 10; // FIXME hardcoded value, this should come from IntegralChannelsForPedestrians
    for(int channel_index=0; channel_index < num_channels; channel_index+=1)
    {
        doppia_protobuf::SoftCascadeOverIntegralChannelsStage *stage_p =
                detector_model_p->mutable_soft_cascade_model()->add_stages();

        stage_p->set_feature_type(doppia_protobuf::SoftCascadeOverIntegralChannelsStage::Level2DecisionTree);
        stage_p->set_weight(0);
        stage_p->set_cascade_threshold(-1);

        doppia_protobuf::IntegralChannelBinaryDecisionTree *level2_decision_tree_p =
                stage_p->mutable_level2_decision_tree();

        // set root node -
        {
            doppia_protobuf::IntegralChannelBinaryDecisionTreeNode *root_node_p =
                    level2_decision_tree_p->add_nodes();

            root_node_p->set_id(0);
            root_node_p->set_parent_id(root_node_p->id());
            doppia_protobuf::IntegralChannelDecisionStump *root_stump_p =
                    root_node_p->mutable_decision_stump();

            root_stump_p->set_feature_threshold(0);
            doppia_protobuf::IntegralChannelsFeature *root_feature_p =
                    root_stump_p->mutable_feature();

            root_feature_p->set_channel_index(channel_index);
            doppia_protobuf::Box *root_box_p = root_feature_p->mutable_box();
            root_box_p->mutable_min_corner()->set_x(0);
            root_box_p->mutable_min_corner()->set_y(0);
            root_box_p->mutable_max_corner()->set_x(feature_width-1);
            root_box_p->mutable_max_corner()->set_y(feature_height-1);
        }

        // set the level two nodes, we do not care about their content
        {
            doppia_protobuf::IntegralChannelBinaryDecisionTreeNode
                    *child_true_node_p = level2_decision_tree_p->add_nodes(),
                    *child_false_node_p = level2_decision_tree_p->add_nodes();

            child_true_node_p->set_id(1);
            child_true_node_p->set_parent_id(0);
            child_true_node_p->set_parent_value(true);

            child_false_node_p->set_id(2);
            child_false_node_p->set_parent_id(0);
            child_false_node_p->set_parent_value(false);

            doppia_protobuf::Box
                    *true_box_p = child_true_node_p->mutable_decision_stump()->mutable_feature()->mutable_box(),
                    *false_box_p = child_false_node_p->mutable_decision_stump()->mutable_feature()->mutable_box();

            true_box_p->mutable_min_corner()->set_x(0);
            true_box_p->mutable_min_corner()->set_y(0);
            true_box_p->mutable_max_corner()->set_x(feature_width-1);
            true_box_p->mutable_max_corner()->set_y(feature_height-1);

            false_box_p->mutable_min_corner()->set_x(0);
            false_box_p->mutable_min_corner()->set_y(0);
            false_box_p->mutable_max_corner()->set_x(feature_width-1);
            false_box_p->mutable_max_corner()->set_y(feature_height-1);
        }

    } // end of "for each channel"

    return;
}

void DetectorsComparisonTestApplication::setup_problem(const program_options::variables_map &options)
{
    const filesystem::path folder_to_process = get_option_value<string>(options, "process_folder");
    directory_input_p.reset(new ImagesFromDirectory(folder_to_process));

    const float score_threshold =
            get_option_value<float>(options, "objects_detector.score_threshold");

    const int additional_border = 0; //get_option_value<int>(options, "additional_border");


    boost::shared_ptr<doppia_protobuf::DetectorModel> detector_model_data_p;

    create_channels_test_model_instance(detector_model_data_p);


    boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>
            cascade_model_p(new SoftCascadeOverIntegralChannelsModel(*detector_model_data_p));

    boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p;
    chnftrs_detector_p.reset(new IntegralChannelsDetector(
                                 options,
                                 cascade_model_p, non_maximal_suppression_p,
                                 score_threshold, additional_border));

    fpdw_detector_p.reset( new FastestPedestrianDetectorInTheWest(
                               options,
                               cascade_model_p, non_maximal_suppression_p,
                               score_threshold, additional_border));
    return;
}




AbstractGui* DetectorsComparisonTestApplication::create_gui(const program_options::variables_map &/*options*/)
{
    // no gui
    return NULL;
}

void DetectorsComparisonTestApplication::main_loop()
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
    check_computed_statistics();
    save_channel_statistics();

    return;
}


void DetectorsComparisonTestApplication::check_computed_statistics()
{

    const float mean_tolerance = 0.1;
    //const float mean_tolerance = 5.0; // FIXME ridiculuous number, just for testing
    const float max_std_deviation = 0.5;
    //const float max_std_deviation = 5.0; // ridiculuous number, just for testing

    for(size_t scale_index=0; scale_index < chnftrs_detector_p->search_ranges_data.size(); scale_index +=1)
    {

        const float detection_window_scale = chnftrs_detector_p->search_ranges_data[scale_index].detection_window_scale;

        for(size_t channel_index = 0; channel_index < this->per_scale_per_channel_statistics.shape()[0]; channel_index +=1)
        {
            const channel_statistics_accumulator_t &statistics =
                    per_scale_per_channel_statistics[channel_index][scale_index];
            const float feature_ratio_mean = accumulators::mean(statistics);
            const float feature_ratio_standard_deviation = sqrt(accumulators::variance(statistics));

            const float relative_scale = detection_window_scale;
            float expected_ratio_mean = get_channel_scaling_factor(channel_index, relative_scale);
            if(expected_ratio_mean < 1.0)
            {
                expected_ratio_mean = 1.0 / expected_ratio_mean;
            }

            printf("Channel %zi, scale\t%.3f, chnftrs/fpdw (all channels together) is in range (%.3f, %.3f)\t"
                   "with an average of\t%.4f (expected %.4f) and a standard deviation of\t%.4f\n",
                   channel_index,
                   detection_window_scale,
                   accumulators::min(statistics),
                   accumulators::max(statistics),
                   feature_ratio_mean,
                   expected_ratio_mean,
                   feature_ratio_standard_deviation);

            // we expect a mean ~1
            BOOST_CHECK( std::abs((feature_ratio_mean/expected_ratio_mean) - 1) < mean_tolerance );

            // we expect a standard deviation of ~0.2
            BOOST_CHECK( feature_ratio_standard_deviation < max_std_deviation );

        } // end of "for each channel"
        printf("\n");
    } // end of "for each scale"

    printf("\n\n");

    for(size_t scale_index=0; scale_index < chnftrs_detector_p->search_ranges_data.size(); scale_index +=1)
    {
        const channel_statistics_accumulator_t &statistics = global_channel_statistics[scale_index];
        const float feature_ratio_mean = accumulators::mean(statistics);
        const float feature_ratio_standard_deviation = sqrt(accumulators::variance(statistics));

        printf("Scale %.3f, chnftrs/fpdw (all channels together) is in range (%.3f, %.3f)\t"
               "with an average of\t%.4f and a standard deviation of\t%.4f\n",
               chnftrs_detector_p->search_ranges_data[scale_index].detection_window_scale,
               accumulators::min(statistics),
               accumulators::max(statistics),
               feature_ratio_mean,
               feature_ratio_standard_deviation);

        // we expect a mean ~1
        //BOOST_CHECK( std::abs(feature_ratio_mean - 1) < mean_tolerance );

        // we expect a standard deviation of ~0.2
        //BOOST_CHECK( feature_ratio_standard_deviation < max_std_deviation );

    } // end of "for each scale"

    return;
}


void DetectorsComparisonTestApplication::save_channel_statistics()
{
    // save the per_scale_per_channel_statistics in a format compatible with
    // http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt

    const string statistics_filename = "channel_statistics.txt";
    FILE *fout = fopen(statistics_filename.c_str(), "w");

    fprintf(fout,
            "# The rows are: scales, "
            "channel_0_mean, channel_0_std_deviation, channel_1_mean, channel_1_std_deviation, etc..\n");

    for(size_t scale_index=0; scale_index < chnftrs_detector_p->search_ranges_data.size(); scale_index +=1)
    {
        fprintf(fout, "%.5f ", chnftrs_detector_p->search_ranges_data[scale_index].detection_window_scale);
    }
    fprintf(fout, "\n");

    for(size_t channel_index = 0; channel_index < this->per_scale_per_channel_statistics.shape()[0]; channel_index +=1)
    {
        for(size_t scale_index=0; scale_index < chnftrs_detector_p->search_ranges_data.size(); scale_index +=1)
        {
            const channel_statistics_accumulator_t &statistics =
                    per_scale_per_channel_statistics[channel_index][scale_index];
            const float feature_ratio_mean = accumulators::mean(statistics);
            fprintf(fout, "%.5f ", feature_ratio_mean);
        } // end of "for each scale"
        fprintf(fout, "\n");

        for(size_t scale_index=0; scale_index < chnftrs_detector_p->search_ranges_data.size(); scale_index +=1)
        {
            const channel_statistics_accumulator_t &statistics =
                    per_scale_per_channel_statistics[channel_index][scale_index];
            const float feature_ratio_standard_deviation = sqrt(accumulators::variance(statistics));

            fprintf(fout, "%.5f ", feature_ratio_standard_deviation);
        } // end of "for each scale"
        fprintf(fout, "\n");


    } // end of "for each channel"

    fclose(fout);

    printf("Created file %s\n", statistics_filename.c_str());
    return;
}

void DetectorsComparisonTestApplication::compute_feature_values(const IntegralChannelsDetector &detector,
                                                                const IntegralChannelsFeature &feature,
                                                                feature_values_t &feature_values) const
{
    const size_t
            max_y = detector.scaled_search_range.max_y,
            max_x = detector.scaled_search_range.max_x;

    feature_values.resize(extents[max_y][max_x]);

    BOOST_REQUIRE(detector.actual_integral_channels_p != NULL);

    // for each pixel in the larger image
    const SlidingIntegralFeature::integral_channels_t &integral_channels =
            *detector.actual_integral_channels_p;

    const int start_col = 0, x_stride = 1;

#pragma omp parallel for
    for(size_t y=0; y < max_y; y+=1)
    {
        SlidingIntegralFeature sliding_feature(feature,integral_channels,
                                               y, start_col, x_stride);
        feature_values_t::reference feature_values_row = feature_values[y];
        for(size_t x=start_col; x < max_x; x+=1, sliding_feature.slide())
        {
            feature_values_row[x] = sliding_feature.get_value();
        } // end of "for each col"
    } // end of "for each row"

    return;
}


/// mini helper function for debuggging, add a rectangle to an existing file
void draw_box(const IntegralChannelsFeature::rectangle_t &box, string png_filename )
{
    gil::rgb8_image_t image;
    gil::png_read_image(png_filename, image);

    draw_rectangle(gil::view(image), rgb8_colors::pink, box);

    gil::png_write_view(png_filename, gil::const_view(image));
    return;
}


class stages_size_visitor: public boost::static_visitor<size_t>
{
public:

    template<typename T>
    size_t operator()(const T &stages) const
    {
        return stages.size();
    }
}; // end of visitor class stages_size_visitor


void DetectorsComparisonTestApplication::update_channels_statistics(
    const size_t current_scale_index,
    const IntegralChannelsDetector &chnftrs_detector,
    const FastestPedestrianDetectorInTheWest &fpdw_detector,
    per_scale_per_channel_statistics_t &per_scale_per_channel_statistics,
    vector<channel_statistics_accumulator_t> &global_channel_statistics) const
{

    BOOST_REQUIRE(chnftrs_detector.actual_ystride == fpdw_detector.actual_ystride);
    BOOST_REQUIRE(chnftrs_detector.actual_xstride == fpdw_detector.actual_xstride);

    // since in the config.ini files has xstride/ystride =~= 1E-10
    BOOST_REQUIRE(chnftrs_detector.actual_ystride == 1);
    BOOST_REQUIRE(chnftrs_detector.actual_xstride == 1);

    BOOST_REQUIRE(chnftrs_detector.scaled_search_range.min_x == 0);
    BOOST_REQUIRE(chnftrs_detector.scaled_search_range.min_y == 0);

    BOOST_REQUIRE(fpdw_detector.scaled_search_range.min_x == 0);
    BOOST_REQUIRE(fpdw_detector.scaled_search_range.min_y == 0);

    const float epsilon = 1E-5;
    // smallest feature value we will use to compute a ratio
    //const float smallest_value_of_interest = 0;
    const float smallest_value_of_interest = 10;



    // both classifiers should have the same artificial cascade
    const SoftCascadeOverIntegralChannelsModel::variant_stages_t  &cascade_stages =
            chnftrs_detector.cascade_model_p->get_stages();

    const size_t cascade_stages_size = boost::apply_visitor(stages_size_visitor(), cascade_stages);
    BOOST_REQUIRE(cascade_stages_size == 10);

    const bool chnftrs_is_bigger_than_fpdw =
            (chnftrs_detector.scaled_search_range.max_y > fpdw_detector.scaled_search_range.max_y);

    float small_to_large_ratio =
            chnftrs_detector.scaled_search_range.range_scaling/fpdw_detector.scaled_search_range.range_scaling;

    if(small_to_large_ratio < 1.0f)
    {
        small_to_large_ratio = 1.0f/small_to_large_ratio;
    }

    printf("scale_index == %zi, "
           "chnftrs range_scaling == %.3f, fpdw range_scaling == %.3f, "
           "small_to_large_ratio == %.3f, chnftrs_is_bigger_than_fpdw == %i\n",
           current_scale_index,
           chnftrs_detector.scaled_search_range.range_scaling,
           fpdw_detector.scaled_search_range.range_scaling,
           small_to_large_ratio,
           chnftrs_is_bigger_than_fpdw);


    // the search radius used for each pixel in the large image (which all map to a single pixel in the small image)
    const int search_radius = static_cast<int>(ceil(small_to_large_ratio/2));

    for(size_t channel_index = 0; channel_index < cascade_stages_size; channel_index += 1)
    {
        // compute the feature values ---
        feature_values_t chnftrs_feature_values, fpdw_feature_values;

        typedef SoftCascadeOverIntegralChannelsModel::plain_stages_t plain_stages_t;
        const plain_stages_t
                &chnftrs_stages = boost::get<plain_stages_t>(chnftrs_detector.actual_cascade_stages),
                &fpdw_stages = boost::get<plain_stages_t>(fpdw_detector.actual_cascade_stages);

        const IntegralChannelsFeature
                &chnftrs_feature = chnftrs_stages[channel_index].weak_classifier.level1_node.feature,
                &fpdw_feature = fpdw_stages[channel_index].weak_classifier.level1_node.feature;

        compute_feature_values(chnftrs_detector, chnftrs_feature, chnftrs_feature_values);
        compute_feature_values(fpdw_detector, fpdw_feature, fpdw_feature_values);

        // compare the values and keep statistics --
        feature_values_t
                *large_image_p, *small_image_p;

        size_t considered_values = 0;

        if(chnftrs_is_bigger_than_fpdw)
        {
            large_image_p = &chnftrs_feature_values;
            small_image_p = &fpdw_feature_values;
        }
        else
        {
            large_image_p = &fpdw_feature_values;
            small_image_p = &chnftrs_feature_values;
        }

        // for each pixel in the small image
        for(size_t y=0; y < small_image_p->shape()[0]; y+=1 )
        {
            const size_t large_y = y*small_to_large_ratio;

            if(large_y >= large_image_p->shape()[0])
            {
                continue;
            }

            for(size_t x=0; x < small_image_p->shape()[1]; x+=1 )
            {

                float small_image_value = (*small_image_p)[y][x];

                if(small_image_value < smallest_value_of_interest)
                {
                    // must skip
                    continue;
                }

                if(small_image_value == 0)
                {
                    small_image_value = epsilon;
                }

                const size_t large_x = x*small_to_large_ratio;

                if(large_x >= large_image_p->shape()[1])
                {
                    continue;
                }

                float best_ratio = std::numeric_limits<float>::max();

                // update best_ratio
                {
                    float large_image_value = (*large_image_p)[large_y][large_x];
                    if(large_image_value > smallest_value_of_interest)
                    {
                        if(large_image_value == 0)
                        {
                            large_image_value = epsilon;
                        }
                        float ratio = large_image_value/small_image_value;
                        if(ratio < 1)
                        {
                            ratio = small_image_value/large_image_value;
                        }

                        // all ratios are bigger or equal to 1
                        best_ratio = std::min(ratio, best_ratio);
                    }
                    else
                    {
                        // we continue searching
                    }
                }

                // explore the corresponding pixels in the large image
                for(int t_y=large_y-search_radius; t_y < static_cast<int>(large_y+search_radius); t_y+=1)
                {
                    if((t_y < 0) or (t_y >= static_cast<int>(large_image_p->shape()[0])))
                    {
                        continue;
                    }

                    for(int t_x=large_x-search_radius; t_x < (large_x+search_radius); t_x+=1)
                    {
                        if((t_x < 0) or (t_x >= static_cast<int>(large_image_p->shape()[1])))
                        {
                            continue;
                        }

                        // update best_ratio
                        {
                            float large_image_value = (*large_image_p)[t_y][t_x];
                            if(large_image_value < smallest_value_of_interest)
                            {
                                // must skip
                                continue;
                            }

                            if(large_image_value == 0)
                            {
                                large_image_value = epsilon;
                            }
                            float ratio = large_image_value/small_image_value;
                            if(ratio < 1)
                            {
                                ratio = small_image_value/large_image_value;
                            }

                            // all ratios are bigger or equal to 1
                            best_ratio = std::min(ratio, best_ratio);
                        }

                    } // end of "for each large_x around"
                } // end of "for each large_y around"

                if(best_ratio != std::numeric_limits<float>::max())
                {

                    if(best_ratio > 1000)
                    {
                        printf("Found a best_ratio fail\n");

                        const float small_image_value = (*small_image_p)[y][x];
                        const float large_image_value = (*large_image_p)[large_y][large_x];
                        printf("At small (%zi,%zi) -> large (%zi, %zi) the values are %.3f -> %.3f, providing a best_ratio == %.3f\n",
                               x,y, large_x, large_y, small_image_value, large_image_value, best_ratio);


                        IntegralChannelsFeature::rectangle_t small_box, large_box;

                        const string
                                small_filename = "small_image_integral_channels.png",
                                large_filename = "large_image_integral_channels.png";


                        // compare the values and keep statistics --
                        if(chnftrs_is_bigger_than_fpdw)
                        {
                            small_box.min_corner().x(x);
                            small_box.min_corner().y(y);
                            small_box.max_corner().x(x + fpdw_feature.box.max_corner().x());
                            small_box.max_corner().y(y + fpdw_feature.box.max_corner().y());

                            large_box.min_corner().x(large_x);
                            large_box.min_corner().y(large_y);
                            large_box.max_corner().x(large_x + chnftrs_feature.box.max_corner().x());
                            large_box.max_corner().y(large_y + chnftrs_feature.box.max_corner().y());

                            save_integral_channels_to_file(*(chnftrs_detector.actual_integral_channels_p),
                                                           large_filename);

                            save_integral_channels_to_file(*(fpdw_detector.actual_integral_channels_p),
                                                           small_filename);
                        }
                        else
                        { // fpdw is bigger than chnftrs

                            small_box.min_corner().x(x);
                            small_box.min_corner().y(y);
                            small_box.max_corner().x(x + chnftrs_feature.box.max_corner().x());
                            small_box.max_corner().y(y + chnftrs_feature.box.max_corner().y());

                            large_box.min_corner().x(large_x);
                            large_box.min_corner().y(large_y);
                            large_box.max_corner().x(large_x + fpdw_feature.box.max_corner().x());
                            large_box.max_corner().y(large_y + fpdw_feature.box.max_corner().y());

                            save_integral_channels_to_file(*(fpdw_detector.actual_integral_channels_p),
                                                           large_filename);

                            save_integral_channels_to_file(*(chnftrs_detector.actual_integral_channels_p),
                                                           small_filename);
                        }

                        printf("small (%i,%i; %i, %i), large (%i,%i; %i, %i)\n",
                               small_box.min_corner().x(), small_box.min_corner().y(),
                               small_box.max_corner().x(), small_box.max_corner().y(),
                               large_box.min_corner().x(), large_box.min_corner().y(),
                               large_box.max_corner().x(), large_box.max_corner().y());

                        draw_box(small_box, small_filename);
                        draw_box(large_box, large_filename);

                        printf("Created large and small_image_integral_channels.png\n");
                        throw std::runtime_error("Stopping computation so we can debug the issue with the ratio values");
                    }

                    per_scale_per_channel_statistics[channel_index][current_scale_index](best_ratio);
                    global_channel_statistics[current_scale_index](best_ratio);

                    considered_values += 1;
                }
                else
                {
                    // could not find a big enough matching value
                    // omited value
                }


            } // end of "for each column in the small image"
        } // end of "for each row in the small image"

        const int omitted_values = small_image_p->num_elements() - considered_values;
        printf("Channel index == %zi, omitted %.3f%% of values (%i out of %zi values)\n",
               channel_index, (100.0f*omitted_values)/small_image_p->num_elements(), omitted_values, small_image_p->num_elements());

    } // end "for each channel"

    return;
}

void DetectorsComparisonTestApplication::process_frame(const AbstractVideoInput::input_image_view_t &input_view)
{
    static bool first_call = true;
    const bool save_score_image = false, false_first_call = false;

    chnftrs_detector_p->detections.clear();
    fpdw_detector_p->detections.clear();

    chnftrs_detector_p->set_image(input_view);
    fpdw_detector_p->set_image(input_view);

    // check that the search ranges are identical
    BOOST_REQUIRE_MESSAGE(chnftrs_detector_p->search_ranges_data == fpdw_detector_p->search_ranges_data,
                          "The search ranges of the two detectors is not identical, but should have been");

    BOOST_REQUIRE(chnftrs_detector_p->search_ranges_data.empty() == false);

    BOOST_REQUIRE(fpdw_detector_p->integral_channels_scales.size() == 1);
    BOOST_REQUIRE(fpdw_detector_p->integral_channels_scales[0] == 1.0f);

    if((per_scale_per_channel_statistics.shape()[0] != 10) or
            (per_scale_per_channel_statistics.shape()[1] < chnftrs_detector_p->search_ranges_data.size()))
    {
        per_scale_per_channel_statistics.resize(boost::extents[10][chnftrs_detector_p->search_ranges_data.size()]);
    }

    if(first_call and chnftrs_detector_p->search_ranges_data.size() < 10)
    {
        printf("Going to search for detection window scales: ");
        BOOST_FOREACH(const DetectorSearchRangeMetaData &search_range_data, chnftrs_detector_p->search_ranges_data)
        {
            printf("%.3f, ", search_range_data.detection_window_scale);
        }
        printf("\n");
    }

    fpdw_detector_p->compute_integral_channels();

    // already checked chnftrs_detector_p->search_ranges_data == fpdw_detector_p->search_ranges_data

    int current_scale_index = 0;
    // for each range search
    for(size_t search_range_index=0; search_range_index < chnftrs_detector_p->search_ranges_data.size(); search_range_index +=1)
    {
        // run on both methods, for this specific scale --
        chnftrs_detector_p->compute_detections_at_specific_scale(search_range_index,
                                                                 save_score_image, false_first_call);

        fpdw_detector_p->compute_detections_at_specific_scale(search_range_index,
                                                              save_score_image, false_first_call);

        // obtain statistics about the channels ratios --
        update_channels_statistics(
                    current_scale_index,
                    *chnftrs_detector_p, *fpdw_detector_p,
                    per_scale_per_channel_statistics,
                    global_channel_statistics);

        current_scale_index += 1;
    } // end of "for each search range"

    printf("\n");

    first_call = false;
    return;
}



} // end of namespace doppia
