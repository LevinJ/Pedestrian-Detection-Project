#include "TrackedDetection2d.hpp"

#include <boost/foreach.hpp>

#include <Eigen/Dense>

#include <cmath>

namespace doppia {


TrackedDetection2d::TrackedDetection2d(const int id, const TrackedDetection2d::detection_t &detection,
                                       const int max_extrapolation_length_)
    :
      object_class(detection.object_class),
      track_id(id),
      max_extrapolation_length(max_extrapolation_length_)
{
    current_bounding_box = detection.bounding_box;
    detections_in_time.push_back(detection);

    max_detection_score = detection.score;
    num_extrapolated_detections = 0;
    num_true_detections_in_time = 0;
    num_consecutive_detections = 0;
    max_consecutive_detections = 0;
    return;
}


TrackedDetection2d::~TrackedDetection2d()
{
    // nothing to do here
    return;
}


void TrackedDetection2d::add_matched_detection(const TrackedDetection2d::detection_t &detection)
{
    assert(detection.object_class == this->object_class);
    current_bounding_box = detection.bounding_box;
    max_detection_score = std::max<float>(max_detection_score, detection.score);

    // we use the sign to mark extrapolated detections,
    // thus we require only positive scores for detections
    //assert(detection.score >= 0);
    detections_in_time.push_back(detection);
    num_true_detections_in_time += 1;

    const bool reweight_score = true;
    if(reweight_score)
    {
        // we set the detection score
        detection_t &last_detection = detections_in_time.back();

        // we consider the max score
        last_detection.score = max_detection_score;

        // we consider the ratio of true detections versus total detections (how many points extrapolated?)
        last_detection.score *= static_cast<float>(num_true_detections_in_time) / detections_in_time.size();

        // and we make longer tracks more confident
        //last_detection.score *= num_true_detections_in_time; // this failed miserably
        //last_detection.score *= std::min(max_consecutive_detections, 5); // this failed miserably too
        last_detection.score *= 2; // this is the magic number
    }

    num_extrapolated_detections = 0;
    num_consecutive_detections += 1;

    max_consecutive_detections = std::max(max_consecutive_detections, num_consecutive_detections);
    return;
}


void TrackedDetection2d::skip_one_detection()
{
    detection_t extrapolated_detection;

    extrapolated_detection.object_class = object_class;
    extrapolated_detection.bounding_box = compute_extrapolated_bounding_box();
    current_bounding_box = extrapolated_detection.bounding_box;

    const float &last_score = detections_in_time.back().score;
    // we use a negative score to indicate that we extrapolated the detection
    //extrapolated_detection.score = -abs(last_score); // FIXME not using negatives, because it makes trouble
    // should use a separate array of bools ??

    extrapolated_detection.score = last_score;

    detections_in_time.push_back(extrapolated_detection);

    num_extrapolated_detections += 1;
    num_consecutive_detections = 0;
    return;
}


int TrackedDetection2d::get_max_extrapolation_length() const
{
    // 15 frames ~= 1 second
    //const int num_detections_for_reliable_track = 15; // FIXME hardcoded parameter
    //const int num_detections_for_reliable_track = 8; // FIXME hardcoded parameter
    const int num_detections_for_reliable_track = 30; // 2 seconds // FIXME hardcoded parameter

    const int num_detections_in_time = std::max<int>(1, detections_in_time.size() - num_extrapolated_detections);
    const bool has_enough_detections = num_detections_in_time > num_detections_for_reliable_track;

    const float window_height = (current_bounding_box.max_corner().y() - current_bounding_box.min_corner().y());
    const int minimum_reliable_track_height = 128; // FIXME hardcoded parameter
    const bool is_high_enough = window_height > minimum_reliable_track_height;

    if(has_enough_detections and is_high_enough)
    {
        return max_extrapolation_length;
    }
    else
    {
        //return std::min(max_extrapolation_length, num_detections_in_time - 1);
        return 0; // we drop all non reliable tracks
    }
}

int TrackedDetection2d::get_extrapolation_length() const
{
    return num_extrapolated_detections;
}


size_t TrackedDetection2d::get_length() const
{
    return detections_in_time.size();
}


const TrackedDetection2d::detection_t &TrackedDetection2d::get_current_detection() const
{
    return detections_in_time.back();
}


const TrackedDetection2d::rectangle_t &TrackedDetection2d::get_current_bounding_box() const
{
    return current_bounding_box;
}


const TrackedDetection2d::detections_t &TrackedDetection2d::get_detections_in_time() const
{
    return detections_in_time;
}


const int TrackedDetection2d::get_id() const
{
    return track_id;
}


TrackedDetection2d::rectangle_t TrackedDetection2d::compute_extrapolated_bounding_box() const
{
    const bool estimate_2d_motion = true;
    if(not estimate_2d_motion)
    {
        return current_bounding_box;
    }

    rectangle_t extrapolated_bbox = current_bounding_box;

    const int max_num_deltas = 10; // FIXME hardcoded parameters

    const int num_detections = detections_in_time.size();
    const int num_deltas = std::min<int>(max_num_deltas, num_detections - 1);

    Eigen::VectorXf delta_x(num_deltas), delta_y(num_deltas), delta_height(num_deltas), weights(num_deltas);

    const rectangle_t &bbox = detections_in_time[num_detections - num_deltas - 1].bounding_box;
    float
            previous_center_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2.0,
            previous_center_y = (bbox.max_corner().y() + bbox.min_corner().y()) / 2.0,
            previous_height = bbox.max_corner().y() - bbox.min_corner().y();

    const int start_index = (num_detections - num_deltas);
    for(int i=start_index; i < num_detections; i+=1)
    {
        const rectangle_t &bbox = detections_in_time[i].bounding_box;
        const float
                center_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2.0,
                center_y = (bbox.max_corner().y() + bbox.min_corner().y()) / 2.0,
                height = bbox.max_corner().y() - bbox.min_corner().y();

        delta_x(i-start_index) = center_x - previous_center_x;
        delta_y(i-start_index) = center_y - previous_center_y;
        delta_height(i-start_index) = height - previous_height;

    } // end of "for each detection"


    const float sum_value = num_deltas*(num_deltas + 1)/0.5;
    for(int i=0; i < num_deltas; i+=1)
    {
        weights(i) = (i+1)/sum_value;
    }

    const float
            x_motion = delta_x.dot(weights),
            y_motion = delta_y.dot(weights),
            height_motion = delta_height.dot(weights);

    // obtain extrapolated bounding box
    {
        const rectangle_t &bbox = current_bounding_box;
        float
                center_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2.0,
                center_y = (bbox.max_corner().y() + bbox.min_corner().y()) / 2.0,
                height = bbox.max_corner().y() - bbox.min_corner().y();

        center_x += x_motion;
        center_y += y_motion;
        height += height_motion;
        height = std::max(height, 5.0f); // we avoid negative or micro-windows
        const float width = height*0.4f; // FIXME hardcoded parameter

        extrapolated_bbox.min_corner().x(center_x - width/2);
        extrapolated_bbox.min_corner().y(center_y - height/2);

        extrapolated_bbox.max_corner().x(center_x + width/2);
        extrapolated_bbox.max_corner().y(center_y + height/2);
    }

    return extrapolated_bbox;
}


void TrackedDetection2d::set_current_bounding_box_as_occluded()
{
    // we only reduce the score, after extrapolating (not after receiving a valid detection)
    if(num_extrapolated_detections > 0)
    {
        // we set the score to a very negative value,
        // so that it does not affect the low false positive area of the precision/recall curve
        detections_in_time.back().score = -1;

        num_consecutive_detections = 0; // if there was a match, it was probably wrong
    }
    return;
}




} // end namespace doppia
