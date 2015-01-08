#ifndef BASEAPPLICATION_HPP
#define BASEAPPLICATION_HPP

#include "AbstractApplication.hpp"
#include "AbstractGui.hpp"

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <fstream>

namespace doppia {


    using namespace std;
    using namespace  boost;

    //  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

class BaseApplication: public AbstractApplication
{
    // children classes will define they internal processing modules
    // shared_ptr<AbstractVideoInput> video_input_p;

    scoped_ptr<AbstractGui> graphic_user_interface_p;

public:

    static boost::program_options::options_description get_options_description(const std::string application_name);
    static void add_args_options(program_options::options_description &desc, const std::string application_name);

    /// human readable title used on the console and graphical user interface
    virtual std::string get_application_title() = 0;

    BaseApplication();
    virtual ~BaseApplication();

    int main(int argc, char *argv[]);

    /// helper method used by the user interfaces when recording screenshots
    const boost::filesystem::path &get_recording_path();

    /// helper method used by the user interfaces when recording screenshots
    /// this number is expected to change with that same frequency that update_gui is called
    virtual int get_current_frame_number() const = 0;

protected:

    virtual void get_all_options_descriptions(program_options::options_description &desc) = 0;


    /// @returns true if the arguments where correctly parsed
    bool parse_arguments(int argc, char *argv[], boost::program_options::variables_map &options);

    virtual void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);

    /// child classes will instanciate here the different processing modules
    virtual void setup_problem(const boost::program_options::variables_map &options) = 0;

    void init_gui(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    virtual AbstractGui* create_gui(const boost::program_options::variables_map &options) = 0;

    virtual void save_solution();

    /// core logic of the application
    /// this method should be redefined by all child classes
    /// @note this method needs to repeatedly call update_gui();
    virtual void main_loop() = 0;

    /// @returns false if the application should stop
    bool update_gui();

    boost::filesystem::path recordings_path;
    void create_recordings_path();

    boost::program_options::variables_map options;
    void record_program_options() const;


protected:
    std::ofstream log_file;

};

} // end of namespace doppia

#endif // BASEAPPLICATION_HPP
