#include "FixedWindowNonMaximalSuppression.hpp"

#include "GreedyNonMaximalSuppression.hpp"

#include <boost/foreach.hpp>

namespace doppia {

FixedWindowNonMaximalSuppression::FixedWindowNonMaximalSuppression()
{
    // nothing to do here
    return;
}


FixedWindowNonMaximalSuppression::~FixedWindowNonMaximalSuppression()
{
    // nothing to do here
    return;
}


void FixedWindowNonMaximalSuppression::set_detections(const detections_t &detections)
{
    candidate_detections = detections;
    return;
}

void FixedWindowNonMaximalSuppression::compute()
{
    detection_t the_detection;
    maximal_detections.clear();

    const int object_width = 48, object_height = 96;
    the_detection.bounding_box.min_corner().x(20 +  8 + 0);
    the_detection.bounding_box.min_corner().y(20 + 16 + 0);
    the_detection.bounding_box.max_corner().x(20 + 8 + object_width);
    the_detection.bounding_box.max_corner().y(20 + 16 + object_height);

    the_detection.object_class = detection_t::Pedestrian;

    float best_overlap = 0;
    BOOST_FOREACH(const detection_t &detection, candidate_detections)
    {
        const float overlap_value = compute_overlap(detection, the_detection);

        const int detection_width = detection.bounding_box.max_corner().x() -detection.bounding_box.min_corner().x(),
                 detection_height = detection.bounding_box.max_corner().y() -detection.bounding_box.min_corner().y();
        if((detection_width != object_width) or (detection_height != object_height))
        {
                throw std::runtime_error("Unexpected window size in FixedWindowNonMaximalSuppression::compute");
        }

        if(overlap_value > best_overlap)
        {
            // we keep the score of the best overlapping window
            the_detection.score = detection.score;

            best_overlap = overlap_value;
        }
    } // end of "for each detection"

    // only one window in the output
    maximal_detections.push_back(the_detection);
    return;
}


} // namespace doppia
