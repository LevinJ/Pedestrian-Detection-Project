#ifndef ObjectsDetectionGui_HPP
#define ObjectsDetectionGui_HPP


#include "objects_detection/AbstractObjectsDetector.hpp"

#include "applications/BaseSdlGui.hpp"
#include "applications/stixel_world/StixelWorldGui.hpp"

#include "applications/objects_detection/ObjectsDetectionApplication.hpp"

#include "objects_detection/Detection2d.hpp"
#include "helpers/data/DataSequence.hpp"

#include <boost/program_options.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <vector>
#include <map>

namespace doppia
{

class ObjectsDetectionApplication; // forward declaration

class ObjectsDetectionGui : public StixelWorldGui
{

    ObjectsDetectionApplication &application;

public:
    typedef ObjectsDetectionApplication::DetectionsDataSequence DetectionsDataSequence;
    typedef AbstractObjectsDetector::detection_t detection_t;
    typedef AbstractObjectsDetector::detections_t detections_t;

    static boost::program_options::options_description get_args_options();

    ObjectsDetectionGui(ObjectsDetectionApplication &application,
                        const boost::program_options::variables_map &options);
    ~ObjectsDetectionGui();

protected:

    bool process_inputs();

    int max_disparity;
    float max_detection_score;

    void resize_if_necessary();

    void draw_video_input();
    void draw_detections();
    void draw_tracks();
    void draw_stixel_world();

#if defined(USE_GPU)
    void draw_gpu_stixel_world();
#endif

    /// used by draw_tracks
    std::map<int, float> track_id_to_hue;

    scoped_ptr<DetectionsDataSequence> ground_truth_data_sequence_p;
    detections_t ground_truth_detections;
};


} // end of namespace doppia

#endif // ObjectsDetectionGui_HPP
