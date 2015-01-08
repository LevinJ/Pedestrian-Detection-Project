#ifndef BaseSdlGui_HPP
#define BaseSdlGui_HPP

#include "AbstractGui.hpp"

#include <boost/cstdint.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/function.hpp>

#include <map>
#include <utility> // for std::pair
#include <string>

// forward declaration
class SDL_Surface;

namespace doppia {

using namespace std;
using namespace  boost;

class BaseApplication; // forward declaration

class BaseSdlGui: public AbstractGui
{
public:
    static boost::program_options::options_description get_args_options();
    static void add_args_options(program_options::options_description &desc);

    /// @note child classes should call init_gui(w,h) inside the constructor
    BaseSdlGui(BaseApplication &application, const boost::program_options::variables_map &options);
    virtual ~BaseSdlGui();

    /// @returnss false if the application should stop
    bool update();

protected:

    void init_gui(const std::string &title, const int input_width, const int input_height);
    void resize_gui(const int input_width, const int input_height);

    bool save_all_screenshots;
    bool recorded_first_image;
    bool colorize_disparity;

    virtual void save_screenshot();

    /// @returns true if the application should stop
    virtual bool process_inputs();
    void print_inputs_instructions() const;

    /// SDL screen surface
    SDL_Surface *screen_p;
    boost::gil::rgb8_image_t screen_image;
    boost::gil::rgb8_view_t screen_image_view;

    /// little helper when changing frames one by one inside a pause
    bool should_stay_in_pause;
public:
    // give easy access to drawing (and hope no one will overwrite the variables)
    // FIXME add accessors
    boost::gil::rgb8_view_t screen_left_view, screen_right_view;

protected:
    typedef boost::function<void ()> drawing_function_t;

    typedef std::pair< drawing_function_t, std::string> view_t;
    typedef std::map<boost::uint8_t, view_t> views_map_t;
    views_map_t views_map;

    view_t current_view;

    void draw_empty_screen();

    /// copy the screen_image content to the actual application view
    void blit_to_screen();

protected:
    BaseApplication &base_application;
};

} // end of namespace doppia

#endif // BaseSdlGui_HPP
