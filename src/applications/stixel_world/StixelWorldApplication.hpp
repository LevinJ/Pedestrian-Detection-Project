#ifndef StixelWorldApplication_HPP
#define StixelWorldApplication_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <fstream>

#include "applications/BaseApplication.hpp"

#include "video_input/AbstractVideoInput.hpp"
#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/motion/AbstractStixelMotionEstimator.hpp"

namespace doppia_protobuf {
class Stixels;
}

namespace doppia
{

using namespace std;
using namespace  boost;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

class StixelWorldGui; // forward declaration
template<typename DataType> class DataSequence;

class StixelWorldApplication : public BaseApplication
{

protected:
    // processing modules
    shared_ptr<AbstractVideoInput> video_input_p;
    shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p;
    shared_ptr<AbstractStixelMotionEstimator> stixel_motion_estimator_p;

public:

    static boost::program_options::options_description get_options_description();

    std::string get_application_title();

    StixelWorldApplication();
     ~StixelWorldApplication();

protected:

    void get_all_options_descriptions(program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();

    /// this method should be called sequentially
    void record_stixels();
    void add_ground_plane_corridor_data(doppia_protobuf::Stixels &stixels_data);
    void add_ground_plane_data(doppia_protobuf::Stixels &stixels_data);

    typedef DataSequence<doppia_protobuf::Stixels> StixelsDataSequence;
    scoped_ptr<StixelsDataSequence> stixels_data_sequence_p;
    bool should_save_stixels, should_save_ground_plane_corridor, silent_mode;

    friend class StixelWorldGui; // used for visualization
};


} // end of namespace doppia

#endif // StixelWorldApplication_HPP
