#include "DummyObjectsTracker.hpp"

#include "helpers/get_option_value.hpp"
#include <boost/foreach.hpp>

#include <cmath>
#include <boost/tuple/tuple.hpp>

namespace doppia {

using namespace boost::program_options;
typedef TrackedDetection2d::rectangle_t rectangle_t;


options_description DummyObjectsTracker::get_args_options()
{

    options_description desc("DummyObjectsTracker options");

    desc.add_options()

            // 15 is what was used in Mitzel ICCV/COORP 2011
            ("objects_tracker.maximum_extrapolation_length", value<int>()->default_value(15),
             "number of frames without detection, before a detection is dropped")

            // FIXME should we instead use a score decay function ?

            ;

    return desc;
}


DummyObjectsTracker::DummyObjectsTracker(const boost::program_options::variables_map &options)
    :
      next_track_id(0),
      max_extrapolation_length(get_option_value<int>(options, "objects_tracker.maximum_extrapolation_length"))
{

    // nothing to do here
    return;
}


DummyObjectsTracker::~DummyObjectsTracker()
{
    // nothing to do here
    return;
}


void DummyObjectsTracker::set_detections(const AbstractObjectsTracker::detections_t &detections)
{
    // FIXME should use fast_pool_allocator
    // http://www.boost.org/doc/libs/1_49_0/libs/pool/doc/html/boost/fast_pool_allocator.html
    new_detections.clear();
    std::copy( detections.begin(), detections.end(), std::back_inserter( new_detections ) );
    return;
}


inline
float area(const rectangle_t &a)
{
    const float delta_x = a.max_corner().x() - a.min_corner().x();
    const float delta_y = a.max_corner().y() - a.min_corner().y();

    const float area = delta_x*delta_y;
    return area;
}


inline
float overlapping_area(const rectangle_t &a,
                       const rectangle_t &b)
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


/// PASCAL VOC criterion
float intersection_over_union(const rectangle_t &a, const rectangle_t &b)
{
    const float intersection_area = overlapping_area(a, b);
    const float area_a = area(a), area_b = area(b);
    const float union_area = area_a + area_b - intersection_area;

    if (union_area > 0)
    {
        return intersection_area / union_area;
    }
    else
    {
        return 0;
    }
}

/// update or create tracks as needed
/// v0 works, but takes matches in an arbritrary order, v1 should be used
void update_tracks_with_detections_v0(DummyObjectsTracker::tracks_t &tracks,
                                      int &next_track_id,
                                      DummyObjectsTracker::new_detections_t &new_detections,
                                      const int max_extrapolation_length)
{

    typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::new_detections_t new_detections_t;

    const float track_to_detection_match_threshold = 0.5;

    // match new_detections to the current track
    // extend tracks that do not match
    BOOST_FOREACH(track_t &track, tracks)
    {
        //const rectangle_t &expected_track_box = track.current_bounding_box;
        const rectangle_t expected_track_box =  track.compute_extrapolated_bounding_box();

        new_detections_t::iterator best_match_it = new_detections.end();
        float best_match_score = 0;

        for(new_detections_t::iterator detections_it = new_detections.begin();
            detections_it != new_detections.end();
            ++detections_it)
        {
            if(detections_it->object_class !=  track.object_class)
            {
                continue;
            }

            const rectangle_t &detection_box = detections_it->bounding_box;

            const float iou = intersection_over_union(expected_track_box, detection_box);
            if(iou >= track_to_detection_match_threshold)
            {
                if(iou > best_match_score)
                {
                    best_match_it = detections_it;
                    best_match_score = iou;
                }
            }

        } // end of "for each new detection"

        if(best_match_it != new_detections.end())
        {
            track.add_matched_detection(*best_match_it);
            new_detections.erase(best_match_it);
        }
        else
        {
            track.skip_one_detection();
        }

    } // end of "for each track"


    // create new tracks for the non-matched windows
    {
        for(new_detections_t::iterator detections_it = new_detections.begin();
            detections_it != new_detections.end();
            ++detections_it)
        {
            tracks.push_back(TrackedDetection2d(next_track_id, *detections_it, max_extrapolation_length));
            next_track_id += 1;
        } // end of "for each (remaining) new detection"

    }

    return;
}

namespace {

typedef boost::tuple<float, size_t, size_t> match_t;

bool compare_matches(const match_t &a, const match_t &b)
{
    return a.get<0>() > b.get<0>();
}

} // end of anonymous namespace

/// update or create tracks as needed
/// greedy matcher: considers the best match first, and the continue to lower scoring ones
void update_tracks_with_detections_v1(DummyObjectsTracker::tracks_t &tracks,
                                      int &next_track_id,
                                      DummyObjectsTracker::new_detections_t &new_detections,
                                      const int max_extrapolation_length)
{

    const float track_to_detection_match_threshold = 0.5;

    typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::tracks_t tracks_t;
    typedef DummyObjectsTracker::new_detections_t new_detections_t;
    typedef AbstractObjectsTracker::detection_t detection_t;

    //typedef boost::tuple<float, size_t, size_t> match_t;
    //typedef std::list<match_t> matches_t;
    typedef std::vector<match_t> matches_t;

    matches_t matches;

    // match new_detections to the current track
    // extend tracks that do not match
    size_t track_index = 0;
    for(tracks_t::const_iterator tracks_it=tracks.begin();
        tracks_it != tracks.end(); ++tracks_it, track_index+=1)
    {
        const track_t &track = *tracks_it;
        //const rectangle_t &expected_track_box = track.current_bounding_box;
        const rectangle_t expected_track_box =  track.compute_extrapolated_bounding_box();

        size_t detection_index = 0;
        for(new_detections_t::iterator detections_it = new_detections.begin();
            detections_it != new_detections.end();
            ++detections_it, detection_index+=1)
        {

            if(detections_it->object_class !=  track.object_class)
            {
                continue;
            }

            const rectangle_t &detection_box = detections_it->bounding_box;

            const float iou = intersection_over_union(expected_track_box, detection_box);
            if(iou >= track_to_detection_match_threshold)
            {
                matches.push_back(match_t(iou, track_index, detection_index));
            }
        } // end of "for each new detection"

    } // end of "for each track"

    std::sort(matches.begin(), matches.end(), compare_matches); // sorting a vector
    //matches.sort(compare_matches); // sorting a list

    // -1 means "not matched"
    std::vector<int>
            track_already_matched(tracks.size(), -1),
            detection_already_matched(new_detections.size(), -1);

    // find the matches ---
    for(matches_t::iterator matches_it=matches.begin();
        matches_it != matches.end(); ++matches_it)
    {

        match_t &accepted_match = *matches_it;
        const int
                the_track_index = accepted_match.get<1>(),
                the_detection_index = accepted_match.get<2>();

        if((track_already_matched[the_track_index] == -1)
           and (detection_already_matched[the_detection_index] == -1))
        {
            track_already_matched[the_track_index] = the_detection_index;
            detection_already_matched[the_detection_index] = the_track_index;
        }

    } // end of "for all matches"

    // update the tracks ---
    track_index = 0;
    for(tracks_t::iterator tracks_it=tracks.begin();
        tracks_it != tracks.end(); ++tracks_it, track_index+=1)
    {
        const int matched_detection_index = track_already_matched[track_index];
        if(matched_detection_index == -1)
        {
            // no match found
            tracks_it->skip_one_detection();
        }
        else
        {
            assert(matched_detection_index >= 0);

            // we retrieve the best matching detection
            new_detections_t::iterator detections_it = new_detections.begin();
            std::advance(detections_it, matched_detection_index);

            tracks_it->add_matched_detection(*detections_it);
        }

    } // end of "for each track"


    // create new tracks ---
    size_t detection_index = 0;
    for(new_detections_t::iterator detections_it = new_detections.begin();
        detections_it != new_detections.end();
        ++detections_it, detection_index+=1)
    {
        const int matched_track_index = detection_already_matched[detection_index];

        if(matched_track_index == -1)
        {
            tracks.push_back(TrackedDetection2d(next_track_id, *detections_it, max_extrapolation_length));
            next_track_id += 1;
        }
        else
        {
            // nothing to do, detection has already been added to matching track
        }
    } // end of "for each new detection"


    return;
}


/// Occluded tracks are set to a very low score
void handle_occlusions(DummyObjectsTracker::tracks_t &tracks)
{

    const float track_maximum_overlap_ratio = 0.6;

    typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::tracks_t tracks_t;
    typedef DummyObjectsTracker::new_detections_t new_detections_t;
    typedef AbstractObjectsTracker::detection_t::rectangle_t rectangle_t;

    for(tracks_t::iterator tracks_it=tracks.begin();
        tracks_it != tracks.end(); ++tracks_it)
    {
        const rectangle_t &box_a = tracks_it->get_current_bounding_box();
        const float area_box_a = area(box_a);

        for(tracks_t::iterator tracks_it2=tracks.begin();
            tracks_it2 != tracks.end(); ++tracks_it2)
        {
            if(tracks_it2 == tracks_it)
            {
                // we skip the same window
                continue;
            }
            else
            {
                const rectangle_t &box_b = tracks_it2->get_current_bounding_box();
                const float
                        intersection_area = overlapping_area(box_a, box_b),
                        overlap_ratio = intersection_area/area_box_a;

                const bool too_much_overlap = overlap_ratio > track_maximum_overlap_ratio;

                // we assume camera above the ground plane
                const bool box_a_is_behind_box_b = box_a.max_corner().y() < box_b.max_corner().y();

                if(too_much_overlap and box_a_is_behind_box_b)
                {
                    tracks_it->set_current_bounding_box_as_occluded();
                    break; // we move to the next track
                }
            }
        } // end of "for each track" (again)

    } // end of "for each track"


    return;
}


void DummyObjectsTracker::compute()
{

    //update_tracks_with_detections_v0(tracks, next_track_id, new_detections, max_extrapolation_length);
    update_tracks_with_detections_v1(tracks, next_track_id, new_detections, max_extrapolation_length);

    handle_occlusions(tracks);

    current_detections.clear();
    // check if one of the tracked objects is too old
    // set the current detections
    for(tracks_t::iterator tracks_it = tracks.begin();
        tracks_it != tracks.end(); )
    {
        const track_t &track = *tracks_it;

        // remove tracks that have extrapolated too long
        if(track.get_extrapolation_length() > track.get_max_extrapolation_length())
        {
            //printf("Removing track of length %zi\n", track.get_length());
            tracks_it = tracks.erase(tracks_it);
        }
        else if(track_is_outside_image(track))
        {
            tracks_it = tracks.erase(tracks_it);
        }
        else if(false and track.get_length() < 3)
        {
            // track is too short, we do not output that detection
            ++tracks_it;
        }
        else
        { // inside the image and not too long

            // copy the current detection
            current_detections.push_back(tracks_it->get_current_detection());

            // since we did not erase, we move forward
            ++tracks_it;
        }
    }

    return;
}


const AbstractObjectsTracker::detections_t &DummyObjectsTracker::get_current_detections() const
{
    return current_detections;
}


const DummyObjectsTracker::tracks_t &DummyObjectsTracker::get_tracks() const
{
    return tracks;
}


bool box_is_outside_image(const rectangle_t &bbox, const int image_width, const int image_height)
{

    bool outside_image = false;

    const bool use_width_center = true;
    if(use_width_center)
    {
        const float center_x = (bbox.min_corner().x() + bbox.max_corner().x()) / 2;
        outside_image |= center_x < 0;
        outside_image |= center_x < 30; // we check the boundary of left image rectification // FIXME hardcoded parameter
        outside_image |= center_x >= image_width;
    }
    else
    { // we use all four corners
        outside_image |= bbox.min_corner().x() >= image_width;
        outside_image |= bbox.min_corner().y() >= image_height;
        outside_image |= bbox.max_corner().x() < 0;
        outside_image |= bbox.max_corner().y() < 0;
    }

    return outside_image;
}


bool DummyObjectsTracker::track_is_outside_image(const DummyObjectsTracker::track_t &track) const
{
    const rectangle_t &bbox = track.get_current_bounding_box();
    return box_is_outside_image(bbox, image_width, image_height);
}


} // end of namespace doppia
