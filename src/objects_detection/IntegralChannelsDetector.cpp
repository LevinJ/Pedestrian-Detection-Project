#include "IntegralChannelsDetector.hpp"

#include "non_maximal_suppression/AbstractNonMaximalSuppression.hpp"
#include "integral_channels/IntegralChannelsForPedestrians.hpp"
#include "SoftCascadeOverIntegralChannelsModel.hpp"

#include "SlidingIntegralFeature.hpp"

#if defined(TESTING)
#include "BaseVeryFastIntegralChannelsDetector.hpp"
#endif

#include "helpers/fill_multi_array.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <boost/gil/image_view_factory.hpp>
//#include <boost/gil/utilities.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <limits>
#include <cmath>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "IntegralChannelsDetector");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "IntegralChannelsDetector");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "IntegralChannelsDetector");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef IntegralChannelsDetector::cascade_stages_t cascade_stages_t;
//typedef cascade_stages_t::value_type cascade_stage_t; // old code
typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t cascade_stage_t;

typedef AbstractObjectsDetector::detections_t detections_t;
typedef AbstractObjectsDetector::detection_t detection_t;
typedef IntegralChannelsDetector::detection_window_size_t detection_window_size_t;
typedef IntegralChannelsDetector::detections_scores_t detections_scores_t;
typedef IntegralChannelsDetector::stages_left_t stages_left_t;
typedef IntegralChannelsDetector::stages_left_in_the_row_t stages_left_in_the_row_t;

typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;

typedef BaseIntegralChannelsDetector::stride_t stride_t;



IntegralChannelsDetector::IntegralChannelsDetector(
        const variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border)
    : BaseIntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p,
                                   score_threshold, additional_border)
{


    max_score_last_frame = score_threshold * 2;

    // create the integral channels computer
    integral_channels_computer_p.reset(new IntegralChannelsForPedestrians());

    return;
}


IntegralChannelsDetector::~IntegralChannelsDetector()
{
    // nothing to do here
    return;
}


void IntegralChannelsDetector::set_image(const boost::gil::rgb8c_view_t &input_view_, const string &image_file_path)
{
    const bool input_dimensions_changed = (input_image.dimensions() != input_view_.dimensions());
    input_image.recreate(input_view_.dimensions());
    input_view = gil::const_view(input_image);
    gil::copy_pixels(input_view_, gil::view(input_image));


    // set default search range --
    if(input_dimensions_changed or search_ranges_data.empty())
    {
        log_debug() << boost::str(boost::format("::set_image resizing the search_ranges using input size (%i,%i)")
                                  % input_view.width() % input_view.height()) << std::endl;

        compute_search_ranges_meta_data(search_ranges_data);

        // update the detection cascades
        compute_scaled_detection_cascades();

        // update additional, input size dependent, data
        compute_extra_data_per_scale(input_view.width(), input_view.height());
    } // end of "set default search range"


    {
        // the first corresponds the smallest detection_window_scale octave,
        // which corresponds to the largest image used to compute integral_channels

        // max_scale is slightly over estimated for the FPDW case, but it is ok,
        // we are not memory constrained
        const float max_scale = 1.0f/min_detection_window_scale;
        //const float max_scale = integral_channels_scales[0]; // FPDW specific case

        const size_t max_y= input_view.height()*max_scale, max_x=input_view.width()*max_scale;
        stages_left_in_the_row.resize(max_y);
        stages_left.resize(boost::extents[max_y][max_x]);
        detections_scores.resize(boost::extents[max_y][max_x]);
    }

    return;
}

// FIXME complete hack, using global variables !
boost::multi_array<int, 2> num_weak_classifiers;
size_t current_y;

// useful for debugging (see also SlidingIntegralFeature.hpp)
const bool print_each_feature_value = false;

template <typename StageType>
inline
bool compute_cascade_stage_on_row(
        const DetectorSearchRange &search_range,
        const StageType &stage, const int stage_index,
        const integral_channels_t &integral_channels,
        const size_t row_index,
        const int xstride,
        const bool use_the_detector_model_cascade,
        detections_scores_t::reference &detections_scores,
        stages_left_t::reference &stages_left)
{
    throw std::runtime_error("Received a call to IntegralChannelsDetector compute_cascade_stage_on_row "
                             "for a type of stage not yet implemented");
    return false;
}


/// @return true if there are still detections left (unresolved) in the row
/// this version is slightly slower than the version using SoftCascadeOverIntegralChannelsModel::fast_stage_t
/// @see SoftCascadeOverIntegralChannelsFastStage
template <>
inline
bool compute_cascade_stage_on_row<SoftCascadeOverIntegralChannelsModel::plain_stage_t>(
        const DetectorSearchRange &search_range,
        const SoftCascadeOverIntegralChannelsModel::plain_stage_t &stage, const int stage_index,
        const integral_channels_t &integral_channels,
        const size_t row_index,
        const int xstride,
        const bool use_the_detector_model_cascade,
        detections_scores_t::reference &detections_scores,
        stages_left_t::reference &stages_left)
{

    assert(use_the_detector_model_cascade == false);

    const bool print_cascade_scores = false; // just for debugging
    bool detections_left_unresolved = false;

    const Level2DecisionTree &weak_classifier = stage.weak_classifier;
    const size_t start_col = search_range.min_x;
    SlidingIntegralFeature
            level1_feature(weak_classifier.level1_node.feature,
                           integral_channels, row_index, start_col, xstride),
            level2_true_feature(weak_classifier.level2_true_node.feature,
                                integral_channels, row_index, start_col, xstride),
            level2_false_feature(weak_classifier.level2_false_node.feature,
                                 integral_channels, row_index, start_col, xstride);

    // we expect search_range.max_x to protect us of reading out of the bounds the image row
    // for each element in row check if score is already too high
    for(size_t col=start_col; col < search_range.max_x;
        col += xstride,
        level1_feature.slide(), level2_true_feature.slide(), level2_false_feature.slide())
    {
        if(stages_left[col] == false)
        {
            // the classifier has already made a decision for this candidate window (position, scale)
            continue;
        }

        detections_scores_t::element &detection_score = detections_scores[col];

        if(print_each_feature_value)
        { // useful for debugging
            printf("stage_index %i ", stage_index);
        }
        bool weak_classifier_result = false;


        if( weak_classifier.level1_node(level1_feature.get_value()) == true)
        {
            weak_classifier_result = weak_classifier.level2_true_node(level2_true_feature.get_value());
        }
        else
        {
            weak_classifier_result = weak_classifier.level2_false_node(level2_false_feature.get_value());
        }

        if(weak_classifier_result == true)
        {
            detection_score += stage.weight;
        }
        else
        {
            // the weak classifier ouput is +1 or -1
            detection_score -= stage.weight;

            // FIXME this is not like
            // {
            //   the detection score stays the same
            //   (see "Robust Object Detection Via Soft Cascade", Bourdev and Brandt, CVPR 2005; figure 2)
            // }
        }

        // (16,16) is the INRIA training positive pedestrians position
        // 1/4 is the integral_image scaling factor
        if(print_cascade_scores
           //and detection_score > 0
           //and (row_index == 16/4) and (col == 16/4)
           //and (row_index == 4) and (col == 4)
           //and (row_index == 255/4) and (col == 423/4)
           //and (row_index == 64) and (col == 106)
           and (row_index == 52) and (col == 4)
           //and (row_index == 203) and (col == 101)
           )
        {
            //log_info()
            cout
                    << str(format("Cascade score at (%i, %i),\tstage %i == %2.3f,\tthreshold == %.3f,\t"
                                  "level1_feature == %.3f,\tlevel2_true_feature == %4.3f,\tlevel2_false_feature == %4.3f")
                           % col % row_index
                           % stage_index
                           % detection_score
                           % stage.cascade_threshold
                           % level1_feature.get_value()
                           % level2_true_feature.get_value()
                           % level2_false_feature.get_value()
                           ) << std::endl;


            cout <<
                    str(format("level1 threshold %.3f, level2_true threshold %.3f, level2_false threshold %.3f")
                        % weak_classifier.level1_node.feature_threshold
                        % weak_classifier.level2_true_node.feature_threshold
                        % weak_classifier.level2_false_node.feature_threshold
                        ) << std::endl;
        }

        if(detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            stages_left[col] = false; // FIXME only for debugging
            //stages_left[col] = true; // no early stop cascade
        }
        //else if(detection_score > detection_threshold)
        //{
        // even if the detection score is above the threshold, we still want to compute the next stages
        // so we can choose the best window in the non maximal suppression stage
        //}
        else
        {
            // stages_left[col] = true; // already set
            // if one detection is left unresolved, then we should revisit the whole row
            detections_left_unresolved = true;
        }

    } // end of "for each column in the row"

    return detections_left_unresolved;
}



/// @return true if there are still detections left (unresolved) in the row
template<>
inline
bool compute_cascade_stage_on_row<SoftCascadeOverIntegralChannelsModel::stump_stage_t>(
        const DetectorSearchRange &search_range,
        const SoftCascadeOverIntegralChannelsModel::stump_stage_t &stage, const int stage_index,
        const integral_channels_t &integral_channels,
        const size_t row_index,
        const int xstride,
        const bool use_the_detector_model_cascade,
        detections_scores_t::reference &detections_scores,
        stages_left_t::reference &stages_left)
{
    bool detections_left_unresolved = false;

    typedef SoftCascadeOverIntegralChannelsModel::stump_stage_t stage_t;
    const stage_t::weak_classifier_t &weak_classifier = stage.weak_classifier;
    const size_t start_col = search_range.min_x;
    SlidingIntegralFeature
            the_feature(weak_classifier.feature,
                        integral_channels, row_index, start_col, xstride);

    // we expect search_range.max_x to protect us of reading out of the bounds the image row
    // for each element in row check if score is already too high
    for(size_t col=start_col; col < search_range.max_x;
        col += xstride,
        the_feature.slide())
    {
        if(stages_left[col] == false)
        {
            // the classifier has already made a decision for this candidate window (position, scale)
            continue;
        }

        detections_scores_t::element &detection_score = detections_scores[col];

        num_weak_classifiers[current_y][col] += 1; // FIXME complete hack

        // update the detection score --
        detection_score += weak_classifier(the_feature.get_value());

        if(use_the_detector_model_cascade and detection_score < stage.cascade_threshold)
        {
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

            // this is not an object of the class we are looking for
            stages_left[col] = false; // FIXME only for debugging
            //stages_left[col] = true; // no early stop cascade
        }
        //else if(detection_score > detection_threshold)
        //{
        // even if the detection score is above the threshold, we still want to compute the next stages
        // so we can choose the best window in the non maximal suppression stage
        //}
        else
        {
            // stages_left[col] = true; // already set
            // if one detection is left unresolved, then we should revisit the whole row
            detections_left_unresolved = true;
        }

    } // end of "for each column in the row"

    return detections_left_unresolved;
}



/// @return true if there are still detections left (unresolved) in the row
template<>
inline
bool compute_cascade_stage_on_row<SoftCascadeOverIntegralChannelsModel::fast_stage_t>(
        const DetectorSearchRange &search_range,
        const SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage, const int stage_index,
        const integral_channels_t &integral_channels,
        const size_t row_index,
        const int xstride,
        const bool use_the_detector_model_cascade,
        detections_scores_t::reference &detections_scores,
        stages_left_t::reference &stages_left)
{

    const bool print_cascade_scores = false; // just for debugging
    bool detections_left_unresolved = false;

    typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t stage_t;
    const stage_t::weak_classifier_t &weak_classifier = stage.weak_classifier;
    const size_t start_col = search_range.min_x;
    SlidingIntegralFeature
            level1_feature(weak_classifier.level1_node.feature,
                           integral_channels, row_index, start_col, xstride),
            level2_true_feature(weak_classifier.level2_true_node.feature,
                                integral_channels, row_index, start_col, xstride),
            level2_false_feature(weak_classifier.level2_false_node.feature,
                                 integral_channels, row_index, start_col, xstride);

    // we expect search_range.max_x to protect us of reading out of the bounds the image row
    // for each element in row check if score is already too high
    for(size_t col=start_col; col < search_range.max_x;
        col += xstride,
        level1_feature.slide(), level2_true_feature.slide(), level2_false_feature.slide())
    {
        if(stages_left[col] == false)
        {
            // the classifier has already made a decision for this candidate window (position, scale)
            continue;
        }

        detections_scores_t::element &detection_score = detections_scores[col];

        num_weak_classifiers[current_y][col] += 1; // FIXME complete hack

        if(print_each_feature_value)
        { // useful for debugging
            printf("stage_index %i ", stage_index);
        }

        // level 1 nodes return a boolean value,
        // level 2 nodes return directly the float value to add to the score
        if( weak_classifier.level1_node(level1_feature.get_value()) == true)
        {
            detection_score += weak_classifier.level2_true_node(level2_true_feature.get_value());
        }
        else
        {
            detection_score += weak_classifier.level2_false_node(level2_false_feature.get_value());
        }

        // (16,16) is the INRIA training positive pedestrians position
        // 1/4 is the integral_image scaling factor
        if(print_cascade_scores
           //and detection_score > 0
           //and (row_index == 16/4) and (col == 16/4)
           //and (row_index == 4) and (col == 4)
           //and (row_index == 255/4) and (col == 423/4)
           //and (row_index == 64) and (col == 106)
           and (row_index == 52) and (col == 4)
           //and (row_index == 203) and (col == 101)
           )
        {
            //log_info()
            cout
                    << str(format("Cascade score at (%i, %i),\tstage %i == %2.3f,\tthreshold == %.3f,\t"
                                  "level1_feature == %.3f,\tlevel2_true_feature == %4.3f,\tlevel2_false_feature == %4.3f")
                           % col % row_index
                           % stage_index
                           % detection_score
                           % stage.cascade_threshold
                           % level1_feature.get_value()
                           % level2_true_feature.get_value()
                           % level2_false_feature.get_value()
                           ) << std::endl;


            cout <<
                    str(format("level1 threshold %.3f, level2_true threshold %.3f, level2_false threshold %.3f")
                        % weak_classifier.level1_node.feature_threshold
                        % weak_classifier.level2_true_node.feature_threshold
                        % weak_classifier.level2_false_node.feature_threshold
                        ) << std::endl;
        }

        if(use_the_detector_model_cascade and detection_score < stage.cascade_threshold)
        {
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

            // this is not an object of the class we are looking for
            stages_left[col] = false; // FIXME only for debugging
            //stages_left[col] = true; // no early stop cascade
        }
        //else if(detection_score > detection_threshold)
        //{
        // even if the detection score is above the threshold, we still want to compute the next stages
        // so we can choose the best window in the non maximal suppression stage
        //}
        else
        {
            // stages_left[col] = true; // already set
            // if one detection is left unresolved, then we should revisit the whole row
            detections_left_unresolved = true;
        }

    } // end of "for each column in the row"

    return detections_left_unresolved;
}



#if defined(TESTING)
// Yes this is ugly, but it is only used for a specific task, in a specific context
// y,x, stages
boost::multi_array<float, 3> scores_trajectories;
size_t current_detector_index = 0;
#endif


void collect_the_detections(
        const ScaleData &scale_data,
        const detection_window_size_t &original_detection_window_size,
        const detections_scores_t &detections_scores,
        const float detection_score_threshold,
        detections_t &detections,
        detections_t *non_rescaled_detections_p)
{

    const DetectorSearchRange &search_range = scale_data.scaled_search_range;
    const stride_t &stride = scale_data.stride;

    assert(search_range.range_scaling > 0);
    //log_debug() << "search_range.range_scaling == " << search_range.range_scaling << std::endl;

    //printf("original_col == col*%.3f\n", 1/search_range.range_scaling); // just for debugging
    //printf("About to collect detections with score >= %.3f\n", detection_score_threshold); // just for debugging

    for(uint16_t row=search_range.min_y; row < search_range.max_y; row += stride.y())
    {
        detections_scores_t::const_reference scores_row = detections_scores[row];
        for(uint16_t col=search_range.min_x; col < search_range.max_x; col += stride.x())
        {
            const detections_scores_t::element &detection_score = scores_row[col];

            // >= to be consistent with Markus's code
            if(detection_score >= detection_score_threshold
               //and (num_weak_classifiers[row][col] > 300) // FIXME complete hack
               and (num_weak_classifiers[row][col] > 3000) // FIXME complete hack
               )
            { // we got a detection, yey !

                add_detection(col, row, detection_score, scale_data, detections);

#if defined(TESTING)
                const size_t num_stages = scores_trajectories[row][col].size();
                detection_t &detection = detections.back();
                detection.score_trajectory.resize(num_stages);
                for(size_t stage_index=0; stage_index < num_stages; stage_index+=1)
                {
                    detection.score_trajectory[stage_index] = scores_trajectories[row][col][stage_index];
                } // end of "for each stage index"
                detection.detector_index = current_detector_index;
#endif

#if defined(BOOTSTRAPPING_LIB)
                if(non_rescaled_detections_p != NULL)
                {
                    add_detection_for_bootstrapping(col, row, detection_score,
                                                    original_detection_window_size,
                                                    *non_rescaled_detections_p);
                }
#endif
            }
            else
            {
                // not a detection, nothing to do
            }

        } // end of "for each column in the search range"
    } // end of "for each row in the search range"

    return;
}


template<typename CascadeStagesType>
void compute_detections_at_specific_scale_impl(
        stages_left_in_the_row_t &stages_left_in_the_row,
        stages_left_t &stages_left,
        detections_scores_t &detections_scores,
        const integral_channels_t &integral_channels,
        const detection_window_size_t &original_detection_window_size,
        const float original_detection_window_scale,
        detections_t &detections, detections_t *non_rescaled_detections_p,
        const CascadeStagesType &cascade_stages,
        const float score_threshold,
        const ScaleData &scale_data,
        const bool print_stages,
        const bool print_cascade_statistics,
        const bool save_score_image,
        const bool use_the_detector_model_cascade)
{

    typedef typename CascadeStagesType::value_type cascade_stage_t;
    const DetectorSearchRange &scaled_search_range = scale_data.scaled_search_range;
    const stride_t &actual_stride = scale_data.stride;

    if((scaled_search_range.max_y == 0) or (scaled_search_range.max_x == 0))
    {
        // nothing to do here
        return;
    }

    std::fill(stages_left_in_the_row.begin(), stages_left_in_the_row.end(), true);
    fill(stages_left, true);
    fill(detections_scores, 0.0f);

    if(num_weak_classifiers.size() != detections_scores.size() )
    {
        num_weak_classifiers.resize(boost::extents[detections_scores.shape()[0]][detections_scores.shape()[1]]);
    }
    fill(num_weak_classifiers, 0);


#if not defined(NDEBUG)
    const size_t input_view_height = integral_channels.shape()[1];
    assert(scaled_search_range.max_y < input_view_height);
#endif

#if defined(TESTING)
    scores_trajectories.resize(
                boost::extents[scaled_search_range.max_y][scaled_search_range.max_x][cascade_stages.size()]);
#endif

    //printf("PING 0 input_view_height == %zi, scaled_search_range.max_y == %i, cascade_stages.size == %zi \n",
    //       input_view_height, scaled_search_range.max_y, cascade_stages.size()); // just for debugging

    // FIXME hardcoded parameter
    //const bool use_partial_detectors = false;
    const bool use_partial_detectors = true; // will try to detect objects that are partially outside of the image frame

    size_t stage_index = 0;
    BOOST_FOREACH(const cascade_stage_t &stage, cascade_stages)
    {
        // we process each row in parallel
#pragma omp parallel for
        for(size_t y=scaled_search_range.min_y; y < scaled_search_range.max_y; y+=actual_stride.y())
        {
            if(stages_left_in_the_row[y] == false)
            {
                // we can safely skip this row
                continue;
            }

            detections_scores_t::reference detections_scores_row = detections_scores[y];
            stages_left_t::reference stages_left_row = stages_left[y];
            current_y = y; // FIXME complete hack

            const bool stages_left = \
                    compute_cascade_stage_on_row(
                        scaled_search_range, stage, stage_index, integral_channels,
                        y, actual_stride.x(), use_the_detector_model_cascade,
                        detections_scores_row, stages_left_row);

            stages_left_in_the_row[y] = stages_left;

        } // end of "for each row in search range"


        size_t max_row = scaled_search_range.max_y;
        if(use_partial_detectors)
        {
            // we will also search in areas where the detection window gets out of the input image
            max_row = integral_channels.shape()[1];
            const size_t max_col = integral_channels.shape()[2];

            DetectorSearchRange search_range_fixed_max_x = scaled_search_range;
            search_range_fixed_max_x.max_x = max_col -1 -stage.get_bounding_box().max_corner().x();

            //printf("stages_left_in_the_row.size() == %zi, max_row == %zi\n",
            //       stages_left_in_the_row.size(), max_row);
            //printf("scaled_search_range.max_y == %zi\n", scaled_search_range.max_y);

#pragma omp parallel for
            for(size_t y=scaled_search_range.max_y; y < max_row; y+=actual_stride.y())
            {
                if(stages_left_in_the_row[y] == false)
                {
                    // we can safely skip this row
                    continue;
                }

                const size_t bottom_y = y + stage.get_bounding_box().max_corner().y();
                if(bottom_y < max_row)
                {
                    detections_scores_t::reference detections_scores_row = detections_scores[y];
                    stages_left_t::reference stages_left_row = stages_left[y];
                    current_y = y; // FIXME complete hack

                    const bool stages_left = \
                            compute_cascade_stage_on_row(
                                search_range_fixed_max_x, stage, stage_index, integral_channels,
                                y, actual_stride.x(), use_the_detector_model_cascade,
                                detections_scores_row, stages_left_row);

                    stages_left_in_the_row[y] = stages_left;
                }
                else
                {
                    // we simply skip this stage
                }

            } // end of "for each row at the bottom of search range"



        } // end of "if use partial detectors"

        int num_rows_left = 0;
        for(size_t y=scaled_search_range.min_y; y < max_row; y+=actual_stride.y())
        {
            if(stages_left_in_the_row[y])
            {
                num_rows_left += 1;
            }
        }

        if(num_rows_left == 0)
        {
            break;
        }

#if defined(TESTING)
        // we copy scores into the scores_trajectories
        for(size_t y=scaled_search_range.min_y; y < scaled_search_range.max_y; y+=actual_stride.y())
        {
            for(size_t x=scaled_search_range.min_x; x < scaled_search_range.max_x; x+=actual_stride.x())
            {
                scores_trajectories[y][x][stage_index] = detections_scores[y][x];
            } // end of "for each column"
        } // end of "for each row"
#endif

        stage_index += 1;
    } // end of "for each cascade stage"

    if(print_stages or print_cascade_statistics)
    {
        log_info() << str(format("Finished all detections of scale %.3f  at cascade stage %i out of %i")
                          % original_detection_window_scale
                          % stage_index
                          % cascade_stages.size()) << std::endl;

        if(stage_index == cascade_stages.size())
        {
            int num_rows_without_stages_left = 0;
            for(size_t i=scaled_search_range.min_y;
                i < min<size_t>(stages_left_in_the_row.size(), scaled_search_range.max_y);
                i+=1)
            {
                if(stages_left_in_the_row[i] == false)
                {
                    num_rows_without_stages_left += 1;
                }
            }

            log_info() << str(format("%i rows out of %i where finished before computing the full scores")
                              % num_rows_without_stages_left
                              % (scaled_search_range.max_y - scaled_search_range.min_y)) << std::endl;
        }
    } // end of "print detections information"


    if(stage_index == cascade_stages.size())
    { // at least one detection reached the last detection stage


        collect_the_detections(scale_data, original_detection_window_size,
                               detections_scores, score_threshold,
                               detections, non_rescaled_detections_p);

        if(use_partial_detectors)
        {
            ScaleData scale_data_fixed = scale_data;

            const size_t
                    max_row = integral_channels.shape()[1],
                    max_col = integral_channels.shape()[2];
            scale_data_fixed.scaled_search_range.max_y = max_row -1;
            scale_data_fixed.scaled_search_range.max_x = max_col -1;

            collect_the_detections(scale_data_fixed, original_detection_window_size,
                                   detections_scores, score_threshold,
                                   detections, non_rescaled_detections_p);
        }
    }

    if(save_score_image)
    {
        cv::Mat scores_mat(
                    //detections_scores.shape()[0], detections_scores.shape()[1],
                    scaled_search_range.max_y, scaled_search_range.max_x,
                    CV_32FC1, detections_scores.origin(),
                    detections_scores.strides()[0]*sizeof(detections_scores_t::element));

        cv::Mat normalized_scores;
        double min_score, max_score;
        cv::minMaxLoc(scores_mat, &min_score, &max_score);
        cv::normalize(scores_mat, normalized_scores, 255, 0, cv::NORM_MINMAX);

        const string filename = str(format("scores_at_%.2f.png") % original_detection_window_scale);
        cv::imwrite(filename, normalized_scores);
        log_info() << "Created debug file " << filename << std::endl;
        log_info() << str(format("Scores in %s are in the range (min, max) == (%.3f, %.3f)")
                          % filename % min_score % max_score) << std::endl;

        if(false and min_score > 0)
        {
            throw std::runtime_error("Min score should be 0 sometimes. Something is fishy in this test.");
        }

    } // end of "if save_score_image"

    return;
} // end of compute_detections_at_specific_scale_impl


/// This class is ugly as hell, but I did not know how to do better (other than creating an arguments structure)
class compute_detections_at_specific_scale_visitor: public boost::static_visitor<void>
{

protected:

    stages_left_in_the_row_t &stages_left_in_the_row;
    stages_left_t &stages_left;
    detections_scores_t &detections_scores;
    const integral_channels_t &integral_channels;
    const detection_window_size_t &original_detection_window_size;
    const float original_detection_window_scale;
    detections_t &detections;
    detections_t *non_rescaled_detections_p;
    //const CascadeStagesType &cascade_stages,
    const float score_threshold;
    const ScaleData &scale_data;
    const bool print_stages;
    const bool print_cascade_statistics;
    const bool save_score_image;
    const bool use_the_detector_model_cascade;

public:

    compute_detections_at_specific_scale_visitor(
            stages_left_in_the_row_t &stages_left_in_the_row_,
            stages_left_t &stages_left_,
            detections_scores_t &detections_scores_,
            const integral_channels_t &integral_channels_,
            const detection_window_size_t &original_detection_window_size_,
            const float original_detection_window_scale_,
            detections_t &detections_, detections_t *non_rescaled_detections_p_,
            //const CascadeStagesType &cascade_stages,
            const float score_threshold_,
            const ScaleData &scale_data_,
            const bool print_stages_,
            const bool print_cascade_statistics_,
            const bool save_score_image_,
            const bool use_the_detector_model_cascade_)
        :
          stages_left_in_the_row(stages_left_in_the_row_),
          stages_left(stages_left_),
          detections_scores(detections_scores_),
          integral_channels(integral_channels_),
          original_detection_window_size(original_detection_window_size_),
          original_detection_window_scale(original_detection_window_scale_),
          detections(detections_),
          non_rescaled_detections_p(non_rescaled_detections_p_),
          //const CascadeStageType &cascade_stages,
          score_threshold(score_threshold_),
          scale_data(scale_data_),
          print_stages(print_stages_),
          print_cascade_statistics(print_cascade_statistics_),
          save_score_image(save_score_image_),
          use_the_detector_model_cascade(use_the_detector_model_cascade_)
    {
        // nothing to do here
        return;
    }

    typedef SoftCascadeOverIntegralChannelsModel::plain_stages_t plain_stages_t;
    typedef SoftCascadeOverIntegralChannelsModel::fast_fractional_stages_t fast_fractional_stages_t;

    template<typename CascadeStagesType>
    void operator()(const CascadeStagesType &cascade_stages) const;

}; // end of visitor class compute_detections_at_specific_scale_visitor


template<typename CascadeStagesType>
void compute_detections_at_specific_scale_visitor::operator()(const CascadeStagesType &cascade_stages) const
{
    doppia::compute_detections_at_specific_scale_impl(
                stages_left_in_the_row,
                stages_left,
                detections_scores,
                integral_channels,
                original_detection_window_size,
                original_detection_window_scale,
                detections, non_rescaled_detections_p,
                cascade_stages,
                score_threshold,
                scale_data,
                print_stages,
                print_cascade_statistics,
                save_score_image,
                use_the_detector_model_cascade);
    return;
}


template<>
void compute_detections_at_specific_scale_visitor::operator()
<compute_detections_at_specific_scale_visitor::plain_stages_t>(
        const compute_detections_at_specific_scale_visitor::plain_stages_t &) const
{
    throw std::runtime_error("plain_stages_t not yet supported");
    return;
}

template<>
void compute_detections_at_specific_scale_visitor::operator()
<compute_detections_at_specific_scale_visitor::fast_fractional_stages_t>(
        const compute_detections_at_specific_scale_visitor::fast_fractional_stages_t &) const
{
    throw std::runtime_error("fast_fractional_stages_t not yet supported");
    return;
}

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
        const bool use_the_detector_model_cascade)
{

    compute_detections_at_specific_scale_visitor visitor(
                stages_left_in_the_row,
                stages_left,
                detections_scores,
                integral_channels,
                detection_window_size,
                original_detection_window_scale,
                detections, non_rescaled_detections_p,
                //cascade_stages,
                score_threshold,
                scale_data,
                print_stages,
                print_cascade_statistics,
                save_score_image,
                use_the_detector_model_cascade);

    boost::apply_visitor(visitor, cascade_stages);
    /*
    compute_detections_at_specific_scale_impl(
                stages_left_in_the_row,
                stages_left,
                detections_scores,
                integral_channels,
                detection_window_size,
                original_detection_window_scale,
                detections, non_rescaled_detections_p,
                cascade_stages,
                score_threshold,
                scale_data,
                print_stages,
                print_cascade_statistics,
                save_score_image,
                use_the_detector_model_cascade);*/
    return;
}


const IntegralChannelsForPedestrians::integral_channels_t &
IntegralChannelsDetector::resize_input_and_compute_integral_channels(const size_t search_range_index,
                                                                     const bool first_call)
{

    IntegralChannelsForPedestrians &integral_channels_computer = *integral_channels_computer_p;

    // rescale the image --
    cv::Mat scaled_input;
    gil::rgb8c_view_t scaled_input_view;
    {
        const image_size_t &scaled_input_image_size = extra_data_per_scale[search_range_index].scaled_input_image_size;

        if(first_call)
        {
            const DetectorSearchRangeMetaData &original_search_range_data = search_ranges_data[search_range_index];
            log_debug() << "resize_input_and_compute_integral_channels "
                        << "scale index == " << search_range_index
                        << ", detection window scale == " << original_search_range_data.detection_window_scale
                        << ", scaled x,y == "
                        << scaled_input_image_size.x() << ", " << scaled_input_image_size.y() << std::endl;
        }

        if(search_range_index > 0)
        { // if not on the first DetectorSearchRange
            // (when search_range_index ==0 we assume a new picture is being treated)

            const image_size_t &previous_image_size = extra_data_per_scale[search_range_index - 1].scaled_input_image_size;

            //printf("%zi =?= %zi; %zi =?= %zi\n",
            //        scaled_input_image_size.x(), previous_image_size.x(),
            //        scaled_input_image_size.y(), previous_image_size.y());

            if((scaled_input_image_size.x() == previous_image_size.x())
               and (scaled_input_image_size.y() == previous_image_size.y()))
            {
                if(first_call)
                {
                    log_debug() << "Skipped integral channels computation for search range index "
                                << search_range_index
                                << " (since redundant with previous computed one)"<< std::endl;
                }
                // current required scale, match the one already computed in the integral_channels_computer
                // no need to recompute the integral_channels, we provide the current result
                return integral_channels_computer.get_integral_channels();
            }
        }

        const gil::opencv::ipl_image_wrapper input_ipl = gil::opencv::create_ipl_image(input_view);
        const cv::Mat input_mat(input_ipl.get());

        int interpolation = cv::INTER_LINEAR;
        if(static_cast<size_t>(input_view.width()) > scaled_input_image_size.x())
        {
            // when shrinking the image, should use a decimation specific interpolation method
            interpolation = cv::INTER_AREA;
        }

        cv::resize(input_mat, scaled_input,
                   cv::Size(scaled_input_image_size.x(), scaled_input_image_size.y()),
                   0, 0, interpolation);

        scaled_input_view =
                gil::interleaved_view(scaled_input.cols, scaled_input.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(scaled_input.data),
                                      static_cast<size_t>(scaled_input.step));
    }

    // compute the integral channels --
    {
        // set image will copy the image content
        integral_channels_computer.set_image(scaled_input_view);
        integral_channels_computer.compute();

        const bool save_integral_channels = false;
        if(save_integral_channels
           and (scaled_input_view.height() > 100))
        {
            integral_channels_computer.save_channels_to_file();
            // stop everything
            throw std::runtime_error("Stopped the program so we can debug it. "
                                     "See the result of integral_channels_computer.save_channels_to_file()");
        }

    }

    return integral_channels_computer.get_integral_channels();
}


size_t IntegralChannelsDetector::get_input_width() const
{
    return input_view.width();
}


size_t IntegralChannelsDetector::get_input_height() const
{
    return input_view.height();
}



void IntegralChannelsDetector::compute_detections_at_specific_scale(
        const size_t search_range_index,
        const bool save_score_image,
        const bool first_call)
{
    // some debugging variables
    const bool
            print_stages = false,
            print_cascade_statistics = false;

    const integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const DetectorSearchRangeMetaData &original_search_range_data = search_ranges_data[search_range_index];
    const variant_stages_t &cascade_stages = detection_cascade_per_scale[search_range_index];
    const detection_window_size_t &detection_window_size = detection_window_size_per_scale[search_range_index];
    const ScaleData &scale_data = extra_data_per_scale[search_range_index];

    // run the cascade classifier and collect the detections --
#if defined(BOOTSTRAPPING_LIB)
    current_image_scale = 1.0f/original_search_range_data.detection_window_scale;
    detections_t *non_rescaled_detections_p = &non_rescaled_detections;
#else
    detections_t *non_rescaled_detections_p = NULL;
#endif

#if defined(TESTING)
    {
        const BaseVeryFastIntegralChannelsDetector *detector_p =
                dynamic_cast<BaseVeryFastIntegralChannelsDetector *>(this);
        if(detector_p)
        {
            current_detector_index = detector_p->detector_index_per_scale[search_range_index];
        }
    }
#endif

    compute_detections_at_specific_scale_visitor visitor(
                stages_left_in_the_row,
                stages_left,
                detections_scores,
                integral_channels,
                detection_window_size,
                original_search_range_data.detection_window_scale, // at original scale
                detections, non_rescaled_detections_p,
                //cascade_stages,
                score_threshold,
                scale_data,
                print_stages,
                print_cascade_statistics,
                save_score_image,
                use_the_detector_model_cascade);

    boost::apply_visitor(visitor, cascade_stages);

#if defined(TESTING)
    // store some key values for testing via DetectorsComparisonTestApplication
    this->scaled_search_range = scaled_search_range;
    this->actual_xstride = actual_xstride;
    this->actual_ystride = actual_ystride;
    this->actual_cascade_stages = cascade_stages;
    this->actual_integral_channels_p = &integral_channels;
#endif

    return;
}


/// filter a detection based on the ground plane constraints
/// @returns true if the detection should be accepted, also returns an updated score
inline
bool filter_detection(const AbstractObjectsDetector::detection_t &detection,
                      const AbstractObjectsDetector::ground_plane_corridor_t &ground_plane_corridor,
                      const int vertical_margin,
                      const int additional_border,
                      float &score)
{

    if(ground_plane_corridor.empty())
    {
        // no information, every detection is valid
        score = detection.score;
        return true;
    }


    const detection_t::rectangle_t &box = detection.bounding_box;
    const int max_y = ground_plane_corridor.size() - 1;

    const int
            bottom_y = std::max(0, box.max_corner().y() - additional_border),
            top_y = std::max(0, box.min_corner().y() - additional_border),
            min_bottom_y = std::min(max_y, std::max(0, bottom_y - vertical_margin)),
            max_bottom_y = std::min(max_y, bottom_y + vertical_margin);
    int
            min_top_y = ground_plane_corridor[min_bottom_y],
            max_top_y = ground_plane_corridor[max_bottom_y];

    if(max_top_y < min_top_y)
    {
        std::swap(max_top_y, min_top_y);
    }

    bool should_be_accepted = true; // by default, we accept

    //log_info() << boost::str( boost::format(" %i <=? top_y == %i  <=? %i")
    //                         % min_top_y % top_y % max_top_y) << std::endl;

    if(min_top_y > 0) // both values are well defined
    {
        should_be_accepted = (top_y >= min_top_y) and (top_y <= max_top_y);

        // for now, we do nothing to the scores
        score = detection.score;
    }

    return should_be_accepted;
}



/// filter a detection based on the ground plane constraints
/// @returns false if the detection should be reject, also returns an updated score and information with the stixel is occluded
inline
bool filter_detection(const AbstractObjectsDetector::detection_t &detection,
                      const stixels_t &estimated_stixels,
                      const int vertical_margin,
                      const int additional_border,
                      float &score,
                      bool &occluded_stixel)
{

    const detection_t::rectangle_t &box = detection.bounding_box;

    const int middle_column_with_border = (box.max_corner().x() + box.min_corner().x()) / 2;
    // additional_border >= 0
    const int middle_column = std::max(0, middle_column_with_border - additional_border);
    const Stixel &center_stixel = estimated_stixels[middle_column];
    const int
            top_margin = std::abs<int>(center_stixel.top_y - std::max(0, box.min_corner().y() - additional_border)),
            bottom_margin = std::abs<int>(center_stixel.bottom_y - std::max(0, box.max_corner().y() - additional_border));

    occluded_stixel = (center_stixel.type == Stixel::Occluded);

    score = detection.score;

    bool should_be_accepted = true;

    if(occluded_stixel)
    {
        // no information to reject the detection
        should_be_accepted = true;
    }
    else if((top_margin <= vertical_margin)
            and (bottom_margin <= vertical_margin))
    { // not occluded stixel

        should_be_accepted = true; // fits in the stixel margins

        const bool adjust_scores = true;
        //const bool adjust_scores = false;
        if(adjust_scores)
        {
            // adjust the score
            const float margin_fraction = (top_margin + bottom_margin)*0.5f / vertical_margin;

            // FIXME hardcoded value
            const float stixels_scores_influence = 1.0;
            // 0 means zero influence, 1.0 means 100% influence
            // 0 if perfect match -> 2x original score
            // 1 would be "almost reject" -> 0.5x original score
            const float
                    perfect_score_scaling = 1.0 + stixels_scores_influence,
                    almost_reject_score_scaling = 1.0f/perfect_score_scaling,
                    score_scaling =
                    (perfect_score_scaling - almost_reject_score_scaling)*std::max(0.0f, 1.0f - margin_fraction)
                    + almost_reject_score_scaling;

            score *= score_scaling; // update the score
        }
    }
    else
    {
        // does not fits in the stixel margins
        should_be_accepted = false;
    }

    return should_be_accepted;
}


/// slow and dummy way to use ground plane constraints
void filter_detections(AbstractObjectsDetector::detections_t &detections,
                       const AbstractObjectsDetector::ground_plane_corridor_t &ground_plane_corridor,
                       const int vertical_margin,
                       const int additional_border)
{
    typedef AbstractObjectsDetector::detections_t detections_t;
    typedef AbstractObjectsDetector::detection_t detection_t;
    detections_t filtered_detections;
    filtered_detections.reserve(detections.size());

    BOOST_FOREACH(detection_t &detection, detections)
    {

        float new_score = detection.score;
        const bool should_be_accepted =
                filter_detection(detection, ground_plane_corridor,
                                 vertical_margin, additional_border,
                                 new_score);

        // the detection fits the ground plane constraints
        if(should_be_accepted)
        {
            detection.score = new_score;
            filtered_detections.push_back(detection);
        }
    } // end of "for each detection"

    detections = filtered_detections;
    return;
}

// FIXME should be moved to BaseIntegralChannelsDetector
/// slow and dummy way to use stixel and ground plane constraints
void filter_detections(AbstractObjectsDetector::detections_t &detections,
                       const stixels_t &estimated_stixels,
                       const AbstractObjectsDetector::ground_plane_corridor_t &ground_plane_corridor,
                       const int vertical_margin,
                       const int additional_border)
{

    if(estimated_stixels.empty() and ground_plane_corridor.empty())
    {
        // nothing to do here
        return;
    }

    if(estimated_stixels.empty() and (not ground_plane_corridor.empty()))
    {
        // only ground plane information is available
        filter_detections(detections, ground_plane_corridor,
                          vertical_margin, additional_border);
        return;
    }

    // stixels and ground plane information is available

    // we already checked inside set_stixels,
    // that the length of estimated_stixels matches the image width

    typedef AbstractObjectsDetector::detections_t detections_t;
    typedef AbstractObjectsDetector::detection_t detection_t;
    detections_t filtered_detections;
    filtered_detections.reserve(detections.size());

    BOOST_FOREACH(detection_t &detection, detections)
    {

        float new_score;
        bool occluded_stixel;
        const bool should_be_accepted =
                filter_detection(detection, estimated_stixels,
                                 vertical_margin, additional_border,
                                 new_score, occluded_stixel);

        // the detection fits the stixels constraints
        if(should_be_accepted and (not occluded_stixel))
        {
            detection.score = new_score;
            filtered_detections.push_back(detection);
        }
        else if(occluded_stixel)
        {
            // occluded stixels are treated using ground plane constraints only
            const bool should_be_accepted =
                    filter_detection(detection, ground_plane_corridor,
                                     vertical_margin, additional_border,
                                     new_score);

            // the detection fits the ground plane constraints
            if(should_be_accepted)
            {
                detection.score = new_score;
                filtered_detections.push_back(detection);
            }
        }
        else
        {
            // not occluded, not inside the stixels margin
            // we just throw away this detection
        }

    } // end of "for each detection"

    detections = filtered_detections;
    return;
}


void IntegralChannelsDetector::compute()
{
    detections.clear();

    // some debugging variables
    const bool save_score_image = false;
    //const bool save_score_image = true;
    static bool first_call = true;

    assert(integral_channels_computer_p);
    assert(search_ranges_data.size() == detection_cascade_per_scale.size());

    // for each range search
    for(size_t search_range_index=0; search_range_index < search_ranges_data.size(); search_range_index +=1)
    {
        compute_detections_at_specific_scale(search_range_index,
                                             save_score_image, first_call);

    } // end of "for each search range"


    if(save_score_image)
    {
        // stop everything
        throw std::runtime_error("Stopped the program so we can debug it. "
                                 "See the scores_at_*.png score images");
    }

    process_raw_detections();

    first_call = false;

    return;
}



void IntegralChannelsDetector::process_raw_detections()
{

    const size_t num_raw_detections = detections.size();

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    // filter the detections based on the previous frame stixels
    filter_detections(detections,
                      estimated_stixels, estimated_ground_plane_corridor,
                      stixels_vertical_margin, additional_border);

    log_info() << "number of detections (before non maximal suppression)  on this frame == "
               << num_raw_detections << " (raw) / " <<  detections.size() << " (after filtering)" << std::endl;

    compute_non_maximal_suppresion();

    return;
}



} // end of namespace doppia
