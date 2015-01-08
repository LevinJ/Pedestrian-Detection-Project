#include "GreedyNonMaximalSuppression.hpp"

#include "helpers/get_option_value.hpp"

//#include <boost/geometry/algorithms/union.hpp>
//#include <boost/geometry/algorithms/intersection.hpp>
//#include <boost/geometry/algorithms/area.hpp>

#include <boost/foreach.hpp>


#if defined(TESTING)
#include <boost/multi_array.hpp>
#include "helpers/fill_multi_array.hpp"
#include <cstdio>
#endif

namespace doppia {

using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef GreedyNonMaximalSuppression::detection_t detection_t;

options_description
GreedyNonMaximalSuppression::get_args_options()
{
    options_description desc("GreedyNonMaximalSuppression options");

    desc.add_options()

            ("objects_detector.minimal_overlap_threshold", value<float>()->default_value(0.65),
             "overlap allowed between two detections be considered different. "
             "The non maximal suppression is based on greddy* variant, "
             "see Section 1.2.1 P. Dollar, Integral Channel Features - Addendum, 2009. "
             "This overlap is _not_ the PASCAL VOC overlap criterion.")
            // 0.65 fixed based on the results of P. Dollar 2009 addendum, figure 2
            ("objects_detector.greedy_overlap_method", value<string>()->default_value("dollar"),
             "defines overlap criterion used:  <dollar> <pascal>"
             "dollar: The non maximal suppression is based on greddy* variant, "
             "see Section 1.2.1 P. Dollar, Integral Channel Features - Addendum, 2009. "
             "pascal: PASCALVOC overlap over union; please adjust objects_detector.minimal_overlap_threshold")
            ;

    return desc;
}


GreedyNonMaximalSuppression::GreedyNonMaximalSuppression(const variables_map &options)
    : minimal_overlap_threshold(
          get_option_value<float>(options, "objects_detector.minimal_overlap_threshold"))
          , overlap_method(get_option_value<string>(options, "objects_detector.greedy_overlap_method"))
{
    // nothing to do here
    return;
}


GreedyNonMaximalSuppression::GreedyNonMaximalSuppression(const float minimal_overlap_threshold_, const string overlap_method_)
    : minimal_overlap_threshold(minimal_overlap_threshold_), overlap_method(overlap_method_)
{
    // nothing to do here
    return;
}


GreedyNonMaximalSuppression::~GreedyNonMaximalSuppression()
{
    // nothing to do here
    return;
}


void GreedyNonMaximalSuppression::set_detections(const detections_t &detections)
{
    candidate_detections.clear();

    // convert vector into list, and copy data
    candidate_detections.assign(detections.begin(), detections.end());

    return;
}


bool has_higher_score(const detection_t &a, const detection_t &b)
{
    return a.score > b.score;
}


float area(const detection_t::rectangle_t &a)
{
    const float delta_x = a.max_corner().x() - a.min_corner().x();
    const float delta_y = a.max_corner().y() - a.min_corner().y();

    const float area = delta_x*delta_y;
    return area;
}


inline
float overlapping_area(const detection_t::rectangle_t &a,
                       const detection_t::rectangle_t &b)
{
    // a and b are expected to be tuples of the type (x1, y1, x2, y2)
    // code adapted from http://visiongrader.sf.net

    const float w =
            std::min(a.max_corner().x(), b.max_corner().x()) -
            std::max(a.min_corner().x(), b.min_corner().x());
    const float h =
            std::min(a.max_corner().y(), b.max_corner().y()) -
            std::max(a.min_corner().y(), b.min_corner().y());
    if (w < 0 or h < 0)
    {
        return 0;
    }
    else
    {
        return w * h;
    }
}

inline
float union_area(const detection_t::rectangle_t &a,
                       const detection_t::rectangle_t &b)
{
    // a and b are expected to be tuples of the type (x1, y1, x2, y2)
    // code adapted from http://visiongrader.sf.net

    const float a_area = area(a);
    const float b_area = area(b);
    const float inter = overlapping_area(a,b);
    return a_area + b_area - inter;

}

// compute the overlap between two detections, using the P. Dollar overlap criterion
// this is _not_ the PASCAL VOC overlap criterion
// (_inlined is a dirty trick to have a local fast version and a "slow" version accessible from other files)
inline
float compute_overlap_inlined(const detection_t &a, const detection_t &b, const string & method = "dollar")
{
    if (method == "dollar"){

        const float intersection_area = overlapping_area(a.bounding_box, b.bounding_box);
        const float area_a = area(a.bounding_box), area_b = area(b.bounding_box);
        const float min_area = std::min(area_a, area_b);
        return intersection_area / min_area;

    }else if (method =="pascal"){
        //using namespace boost::geometry;

        const float union_area_value = union_area( a.bounding_box, b.bounding_box );
        const float intersection_area = overlapping_area(a.bounding_box, b.bounding_box);
        return intersection_area / union_area_value;

    }else{

        throw std::runtime_error("overlap method must be 'dollar' or 'pascal'");
    }

}


float compute_overlap(const detection_t &a, const detection_t &b, const string & method)
{
    return compute_overlap_inlined(a, b, method);
}


#if defined(TESTING)

void GreedyNonMaximalSuppression::compute()
{
    candidate_detections.sort(has_higher_score);
    maximal_detections.clear();
    maximal_detections.reserve(42); // we do not expect more than 42 pedestrians per scene

    static boost::multi_array<float, 2> soft_cascades_thresholds;
    const size_t max_num_detector = 10;
    static size_t max_detector_index = 0;
    //const size_t max_k = 1;
    const size_t max_k = 3;
    //const size_t max_k = 5;

    if(soft_cascades_thresholds.empty() and (candidate_detections.size() > 0))
    {
        soft_cascades_thresholds.resize(boost::extents[max_num_detector][candidate_detections.front().score_trajectory.size()]);
        fill(soft_cascades_thresholds, std::numeric_limits<float>::max());
    }

    candidate_detections_t::iterator detections_it = candidate_detections.begin();
    for(; detections_it != candidate_detections.end(); ++detections_it)
    {
        const detection_t &detection = *detections_it;

        // this detection passed the test
        maximal_detections.push_back(detection);

        size_t k = 0; // how many score trajectories we have already considered
        for(size_t stage_index=0; stage_index < soft_cascades_thresholds.shape()[1];  stage_index+=1)
        {
            float &threshold = soft_cascades_thresholds[detection.detector_index][stage_index];
            threshold = std::min(threshold, detection.score_trajectory[stage_index]);
            max_detector_index = std::max(max_detector_index, detection.detector_index);
        }
        k+=1;


        candidate_detections_t::iterator lower_score_detection_it = detections_it;
        ++lower_score_detection_it; // = detections_it + 1
        for(; lower_score_detection_it != candidate_detections.end(); )
        {
            const float overlap = compute_overlap_inlined(detection, *lower_score_detection_it);

            if(overlap > minimal_overlap_threshold)
            {
                if(k < max_k)
                {
                    for(size_t stage_index=0; stage_index < soft_cascades_thresholds.shape()[1];  stage_index+=1)
                    {
                        float &threshold = soft_cascades_thresholds[lower_score_detection_it->detector_index][stage_index];
                        threshold = std::min(threshold, lower_score_detection_it->score_trajectory[stage_index]);
                        max_detector_index = std::max(max_detector_index, lower_score_detection_it->detector_index);
                    }
                    k+=1;
                } // end of "k not yet reached max_k"


                // this detection seems to overlap too much, we should remove it
                lower_score_detection_it = candidate_detections.erase(lower_score_detection_it);
            }
            else
            {
                // we keep this detection in the candidates list
                ++lower_score_detection_it;
            }

        } // end of "for each lower score detection"


    } // end of "for each candidate detection"


    static size_t num_calls = 0;
    num_calls += 1;

    if(num_calls > 10)
    {
        const char filename[] = "cascade_threshold.txt";
        FILE *fout = fopen(filename, "w");

        for(size_t detector_index=0; detector_index <= max_detector_index; detector_index +=1)
        {
            for(size_t stage_index=0; stage_index < soft_cascades_thresholds.shape()[1]; stage_index+=1)
            {
                fprintf(fout, "%.8f ", soft_cascades_thresholds[detector_index][stage_index]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);

        printf("Wrote current results in %s (num_cascade_stages == %zi, num_detectors == %zi)\n",
               filename, soft_cascades_thresholds.shape()[1], max_detector_index + 1);
        num_calls = 0;
    }

    return;
}

#else

void GreedyNonMaximalSuppression::compute()
{
    candidate_detections.sort(has_higher_score);
    maximal_detections.clear();
    maximal_detections.reserve(64); // we do not expect more than 64 pedestrians per scene

    candidate_detections_t::iterator detections_it = candidate_detections.begin();
    for(; detections_it != candidate_detections.end(); ++detections_it)
    {
        const detection_t &detection = *detections_it;

        // this detection passed the test
        maximal_detections.push_back(detection);

        candidate_detections_t::iterator lower_score_detection_it = detections_it;
        ++lower_score_detection_it; // = detections_it + 1
        for(; lower_score_detection_it != candidate_detections.end(); )
        {
            const float overlap = compute_overlap_inlined(detection, *lower_score_detection_it, overlap_method);

            if(overlap > minimal_overlap_threshold)
            {
                // this detection seems to overlap too much, we should remove it
                lower_score_detection_it = candidate_detections.erase(lower_score_detection_it);
            }
            else
            {
                // we keep this detection in the candidates list
                ++lower_score_detection_it;
            }

        } // end of "for each lower score detection"


    } // end of "for each candidate detection"

    return;
}

#endif // if defined TESTING




} // end of namespace doppia
