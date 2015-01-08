#ifndef INTEGRALCHANNELSCOMPARISONTESTAPPLICATION_HPP
#define INTEGRALCHANNELSCOMPARISONTESTAPPLICATION_HPP

#include "applications/BaseApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

namespace doppia {

class IntegralChannelsForPedestrians;
class GpuIntegralChannelsForPedestrians;
class ImagesFromDirectory;

class IntegralChannelsComparisonTestApplication: public BaseApplication
{

protected:
    // processing modules
    scoped_ptr<ImagesFromDirectory> directory_input_p;
    boost::scoped_ptr<IntegralChannelsForPedestrians> cpu_integral_channels_computer_p;
    boost::scoped_ptr<GpuIntegralChannelsForPedestrians> gpu_integral_channels_computer_p;

public:
    IntegralChannelsComparisonTestApplication();
    ~IntegralChannelsComparisonTestApplication();

    /// human readable title used on the console and graphical user interface
    std::string get_application_title();

    /// helper method used by the user interfaces when recording screenshots
    /// this number is expected to change with that same frequency that update_gui is called
    int get_current_frame_number() const;

protected:

    void get_all_options_descriptions(program_options::options_description &desc);

    /// child classes will instanciate here the different processing modules
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    /// core logic of the application
    /// this method should be redefined by all child classes
    /// @note this method needs to repeatedly call update_gui();
    void main_loop();

    void process_frame(const AbstractVideoInput::input_image_view_t &input_view);

};

} // end of namespace doppia

#endif // INTEGRALCHANNELSCOMPARISONTESTAPPLICATION_HPP
