#include "draw_the_detections.hpp"

#include "drawing/gil/colors.hpp"
#include "drawing/gil/line.hpp"

#include <boost/foreach.hpp>

namespace doppia {


void draw_the_detections(
        const AbstractObjectsDetector::detections_t &detections,
        const AbstractObjectsDetector::detections_t &ground_truth_detections,
        float &max_detection_score,
        const int additional_border,
        const boost::gil::rgb8_view_t &view)
{

    typedef AbstractObjectsDetector::detection_t detection_t;
    float min_score = 0; // we will saturate at negative scores
    //float min_score = std::numeric_limits<float>::max();

    BOOST_FOREACH(const detection_t &detection, detections)
    {
        //min_score = std::min(min_score, detection.score);
        max_detection_score = std::max(max_detection_score, detection.score);
    }

    //max_detection_score = 100; // FIXME hardcoded based on ./plot_detections_statistics.py over TUD brussels (octave 0 model)
    const float scaling = 255 / (max_detection_score - min_score);

    BOOST_FOREACH(const detection_t &detection, detections)
    {
        const boost::uint8_t normalized_score = static_cast<boost::uint8_t>(
                    std::min(std::max(0.0f, (detection.score - min_score)*scaling), 255.0f));
        //boost::gil::rgb8c_pixel_t color = rgb8_colors::white;
        boost::gil::rgb8c_pixel_t color(normalized_score, 0, 0);

        detection_t::rectangle_t box = detection.bounding_box;

        box.min_corner().x(box.min_corner().x() - additional_border);
        box.min_corner().y(box.min_corner().y() - additional_border);
        box.max_corner().x(box.max_corner().x() - additional_border);
        box.max_corner().y(box.max_corner().y() - additional_border);

        draw_rectangle(view, color, box, 4);
    }

    // draw ground truth
    for(size_t i=0; i < ground_truth_detections.size(); i+=1)
    {
        boost::gil::rgb8c_pixel_t color = rgb8_colors::white;
        const detection_t::rectangle_t &box = ground_truth_detections[i].bounding_box;
        draw_rectangle(view, color, box, 2);
    }

    return;
}


} // end of namespace doppia
