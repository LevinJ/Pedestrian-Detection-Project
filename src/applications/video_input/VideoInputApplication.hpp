#ifndef VideoInputApplication_HPP
#define VideoInputApplication_HPP

#include "applications/BaseApplication.hpp"

#include "video_input/AbstractVideoInput.hpp"
#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"


namespace doppia
{

using namespace std;
using namespace  boost;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

class VideoInputGui; // forward declaration

class VideoInputApplication : public BaseApplication
{

    // processing modules
    shared_ptr<AbstractVideoInput> video_input_p;

    friend class VideoInputGui;

public:

    static boost::program_options::options_description get_options_description(void);

    std::string get_application_title();

    VideoInputApplication();
    virtual ~VideoInputApplication();

protected:

    void get_all_options_descriptions(program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();

};


} // end of namespace doppia

#endif // VideoInputApplication_HPP
