#ifndef DOPPIA_ComputeFeatureChannelsApplication_HPP
#define DOPPIA_ComputeFeatureChannelsApplication_HPP

#include "applications/BaseApplication.hpp"
#include <boost/filesystem/path.hpp>

namespace doppia {

// forward declarations
class ImagesFromDirectory;
class AbstractChannelsComputer;

class ComputeFeatureChannelsApplication : public doppia::BaseApplication
{

    // processing modules
    boost::scoped_ptr<ImagesFromDirectory> directory_input_p;
    boost::scoped_ptr<AbstractChannelsComputer> channels_computer_p;

public:

    static boost::program_options::options_description get_options_description();

    std::string get_application_title();

    ComputeFeatureChannelsApplication();
    virtual ~ComputeFeatureChannelsApplication();

protected:

    void get_all_options_descriptions(boost::program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();


protected:

    bool silent_mode;
    size_t num_files_to_process;
    boost::filesystem::path output_path;

};

} // end of namespace doppia

#endif // DOPPIA_ComputeFeatureChannelsApplication_HPP
