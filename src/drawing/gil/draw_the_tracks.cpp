#include "draw_the_tracks.hpp"

#include "video_input/MetricCamera.hpp"
#include "video_input/calibration/CameraCalibration.hpp"

#include "stereo_matching/ground_plane/GroundPlane.hpp"

#include "line.hpp"
#include "hsv_to_rgb.hpp"

#include <Eigen/Dense>

#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <boost/foreach.hpp>

#include <cstdio>


namespace doppia {

using namespace std;
using namespace boost;

namespace {
/// random_generator should be accessed only by one thread
boost::mt19937 random_generator;
} // end of anonymous namespace


gil::rgb8_pixel_t get_track_color(const int track_id,
                                  const float normalized_score,
                                  std::map<int, float> &track_id_to_hue)
{
    assert((normalized_score >= 0) and (normalized_score <= 1));


    //color = rgb8_colors::white;
    //color = gil::rgb8_pixel_t(normalized_score*255, 0, 0); // red box
    const float
            //value = 0.9,
            value = normalized_score,
            saturation = 0.8;


    if(track_id_to_hue.size() > 1000)
    {
        // to avoid memory leacks we reset the colors once in a while
        track_id_to_hue.clear();
    }


    if(track_id_to_hue.count(track_id) == 0)
    {
        boost::uniform_real<float> random_hue;
        // random generator should be accessed only by one thread
        track_id_to_hue[track_id] = random_hue(random_generator);
    }

    const float hue = track_id_to_hue[track_id];
    //printf("track_id_to_hue[%i] == %.3f\n", id, hue);
    //printf("track_id %i value == %.3f\n", id, value);
    return hsv_to_rgb(hue, saturation, value);
}


void draw_track(const DummyObjectsTracker::track_t &track,
                const int additional_border,
                const boost::gil::rgb8_pixel_t &color,
                const boost::gil::rgb8_view_t &view)
{

    // draw current bounding box --
    {
        DummyObjectsTracker::track_t::rectangle_t box = track.get_current_bounding_box();

        box.min_corner().x(box.min_corner().x() - additional_border);
        box.min_corner().y(box.min_corner().y() - additional_border);
        box.max_corner().x(box.max_corner().x() - additional_border);
        box.max_corner().y(box.max_corner().y() - additional_border);

        draw_rectangle(view, color, box, 4);
    }

    // draw tails
    {
        const AbstractObjectsTracker::detections_t &detections_in_time = track.get_detections_in_time();

        const AbstractObjectsTracker::detection_t::rectangle_t &bbox = detections_in_time.front().bounding_box;
        int
                previous_middle_low_point_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2,
                previous_middle_low_point_y = bbox.max_corner().y();

        BOOST_FOREACH(const AbstractObjectsTracker::detection_t &detection, detections_in_time)
        {
            const AbstractObjectsTracker::detection_t::rectangle_t &bbox = detection.bounding_box;

            const int
                    middle_low_point_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2,
                    middle_low_point_y = bbox.max_corner().y();

            draw_line(view, color, middle_low_point_x, middle_low_point_y,
                      previous_middle_low_point_x, previous_middle_low_point_y);

            previous_middle_low_point_x = middle_low_point_x;
            previous_middle_low_point_y = middle_low_point_y;
        } // end of "for each detection in time"

    }

    return;
}


void draw_the_tracks(
        const DummyObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const boost::gil::rgb8_view_t &view)
{

    typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::detection_t detection_t;
    typedef DummyObjectsTracker::detections_t detections_t;

    const float min_score = 0; // we will saturate at negative scores
    //float min_score = std::numeric_limits<float>::max();

    BOOST_FOREACH(const track_t &track, tracks)
    {
        //const float score = abs(track.get_current_detection().score);
        const float score = track.get_current_detection().score;
        //min_score = std::min(min_score, detection.score);
        max_detection_score = std::max(max_detection_score, score);
    }

    //printf("max_detection_score == %.3f\n", max_detection_score);

    const float scaling = 1.0 / (max_detection_score - min_score);

    BOOST_FOREACH(const track_t &track, tracks)
    {
        gil::rgb8_pixel_t color;

        // get track color
        {
            const int id = track.get_id();
            //const float score = abs(track.get_current_detection().score);
            const float score = track.get_current_detection().score;
            //printf("track_id %i score == %.3f\n", id, score);
            const float normalized_score = std::max(0.0f, (score - min_score)*scaling);

            color = get_track_color(id, normalized_score, track_id_to_hue);
        }

        draw_track(track, additional_border, color, view);

    } // end of "for each track"

    return;
}



} // end of namespace doppia
