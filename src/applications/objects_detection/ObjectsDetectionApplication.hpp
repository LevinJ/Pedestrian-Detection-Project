#ifndef ObjectsDetectionApplication_HPP
#define ObjectsDetectionApplication_HPP


#include "applications/BaseApplication.hpp"

#include "video_input/AbstractVideoInput.hpp"


#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <string>

namespace doppia_protobuf {
class Detections;
}


namespace doppia
{

using namespace std;
using namespace  boost;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

 // forward declarations
class ObjectsDetectionGui;
class AbstractObjectsDetector;
class AbstractStixelWorldEstimator;
class AbstractObjectsTracker;
template<typename DataType> class DataSequence;
class ImagesFromDirectory;

class ObjectsDetectionApplication: public BaseApplication
{

protected:
    // processing modules
    shared_ptr<AbstractVideoInput> video_input_p;
    scoped_ptr<ImagesFromDirectory> directory_input_p;
    scoped_ptr<AbstractObjectsDetector> objects_detector_p;
    shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p;
    scoped_ptr<AbstractObjectsTracker> objects_tracker_p;

    friend class ObjectsDetectionGui;
    scoped_ptr<ObjectsDetectionGui> graphic_user_interface_p;

public:
    typedef DataSequence<doppia_protobuf::Detections> DetectionsDataSequence;

    static program_options::options_description get_options_description();

    std::string get_application_title();

    ObjectsDetectionApplication();
    ~ObjectsDetectionApplication();

protected:

    void get_all_options_descriptions(program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const program_options::variables_map &options);
    void setup_problem(const program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();

    void record_detections();

    scoped_ptr<DetectionsDataSequence> detections_data_sequence_p;
    bool should_save_detections, use_ground_plane_only, should_process_folder, silent_mode;

    int additional_border, stixels_computation_period;
};


} // end of namespace doppia

#endif // ObjectsDetectionApplication_HPP
