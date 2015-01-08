#ifndef GroundEstimationApplication_HPP
#define GroundEstimationApplication_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <fstream>

#include "applications/BaseApplication.hpp"

#include "video_input/AbstractVideoInput.hpp"
#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/cost_volume/AbstractDisparityCostVolumeEstimator.hpp"

#include "stereo_matching/ground_plane/GroundPlane.hpp"

namespace doppia_protobuf {
class Stixels;
}

namespace doppia
{

using namespace std;
using namespace  boost;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

// forward declarations
class GroundEstimationGui;
class GroundPlaneEstimator;
class FastGroundPlaneEstimator;
template<typename DataType> class DataSequence;

class GroundEstimationApplication : public BaseApplication
{
    // processing modules
    shared_ptr<AbstractVideoInput> video_input_p;
    shared_ptr<AbstractDisparityCostVolumeEstimator> cost_volume_estimator_p;
    scoped_ptr<GroundPlaneEstimator> ground_plane_estimator_p;
    scoped_ptr<FastGroundPlaneEstimator> fast_ground_plane_estimator_p;


    friend class GroundEstimationGui;

public:

    static boost::program_options::options_description get_options_description();

    std::string get_application_title();

    GroundEstimationApplication();
    virtual ~GroundEstimationApplication();

protected:

    void get_all_options_descriptions(program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();

    shared_ptr<DisparityCostVolume> pixels_matching_cost_volume_p;
    GroundPlane ground_plane_prior, current_ground_plane_estimate;

public:
    // GroundPlane is a Eigen structure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


} // end of namespace doppia

#endif // GroundEstimationApplication_HPP
