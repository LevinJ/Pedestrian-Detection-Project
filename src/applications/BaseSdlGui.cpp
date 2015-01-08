#include "BaseSdlGui.hpp"

#include "BaseApplication.hpp"

#include "video_input/AbstractVideoInput.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"

#include "SDL/SDL.h"

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <boost/gil/extension/io/png_io.hpp>



namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "BaseSdlGui");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "BaseSdlGui");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "BaseSdlGui");
}

} // end of anonymous namespace


namespace doppia
{

program_options::options_description BaseSdlGui::get_args_options()
{
    program_options::options_description desc("BaseSdlGui options");

    desc.add_options()

            ("gui.save_all_screenshots",
             program_options::value<bool>()->default_value(false),
             "save a screenshot of the rendered window for every input frame")
            ;


    return desc;
}

void BaseSdlGui::add_args_options(program_options::options_description &desc)
{

    // add the options --
    typedef std::vector< boost::shared_ptr<boost::program_options::option_description> > base_options_t;

    // needs to be static to avoid crash when using options latter
    static program_options::options_description base_args_options = BaseSdlGui::get_args_options();

    const base_options_t &base_options = base_args_options.options();

    for(base_options_t::const_iterator it= base_options.begin(); it != base_options.end(); ++it)
    {
        desc.add(*it);
    }

    return;
}

BaseSdlGui::BaseSdlGui(BaseApplication &application, const program_options::variables_map &options)
    : AbstractGui(options), base_application(application)
{
    save_all_screenshots = get_option_value<bool>(options, "gui.save_all_screenshots");
    recorded_first_image = false;
    should_stay_in_pause = false;

    // child classes should call init_gui(w,h)

    // child classes should populate the views map
    views_map[SDLK_0] = view_t(boost::bind(&BaseSdlGui::draw_empty_screen, this), "draw_empty_screen");

    return;
}

void BaseSdlGui::init_gui(const std::string &title, const int input_width, const int input_height)
{

    // create the application window
    SDL_Init(SDL_INIT_VIDEO);
    resize_gui(input_width, input_height);

    SDL_WM_SetCaption(title.c_str(), base_application.get_application_title().c_str());

    print_inputs_instructions();

    // set the initial view mode -
    current_view = views_map[SDLK_0];

    return;
}

void BaseSdlGui::resize_gui(const int input_width, const int input_height)
{

    screen_p = SDL_SetVideoMode(input_width*2, input_height, 24, SDL_HWSURFACE);
    if(screen_p == NULL)
    {
        fprintf(stderr, "Couldn't set %ix%i video mode: %s\n",
                input_width*2, input_height,
                SDL_GetError());
        throw std::runtime_error("Could not set SDL_SetVideoMode");
    }

    screen_image.recreate(input_width*2, input_height);
    screen_image_view = boost::gil::view(screen_image);

    screen_left_view = boost::gil::subimage_view(screen_image_view, 0, 0, input_width, input_height);
    screen_right_view = boost::gil::subimage_view(screen_image_view, input_width, 0, input_width, input_height);
    return;
}


BaseSdlGui::~BaseSdlGui()
{
    // nothing to do here
    return;
}



bool BaseSdlGui::process_inputs()
{

    bool end_of_game = false;

    SDL_Event event;

    while ( SDL_PollEvent(&event) )
    {
        switch(event.type)
        {
        case SDL_VIDEORESIZE:
            // we do not want the user the play around with the window size
            throw std::runtime_error("BaseSdlGui::process_inputs does not support window resizing");
            break;

        case SDL_QUIT:
            end_of_game = true;
            break;
        }
    }

    Uint8 *keys = SDL_GetKeyState(NULL);

    if(keys[SDLK_ESCAPE] or keys[SDLK_q])
    {
        end_of_game = true;
    }


    // check for a change in the view mode --
    for(views_map_t::const_iterator views_it=views_map.begin();
        views_it != views_map.end();
        ++views_it)
    {

        const boost::uint8_t &view_key = views_it->first;

        if(keys[view_key])
        {
            const view_t &new_view = views_it->second;
            const std::string &new_view_name = new_view.second;

            // boost::function operator== cannot be defined (?!)
            // http://www.boost.org/doc/libs/1_44_0/doc/html/function/faq.html#id1284482
            // to work around the issue we use the string description and pray that the strings are unique

            //if(drawing_function != view_drawing_function)
            if(current_view.second != new_view_name)
            {
                printf("Switching to view %i: %s\n",
                       (view_key - SDLK_0), new_view_name.c_str());
                current_view = new_view;
            }

        }

    } // end of "for each view in views_map"


    if( keys[SDLK_s])
    {
        save_screenshot();
    }

    bool application_is_in_pause = should_stay_in_pause;
    should_stay_in_pause = false;

    if( (keys[SDLK_p] or keys[SDLK_SPACE]) and (application_is_in_pause == false))
    {
        application_is_in_pause = true;
        printf("Entering into a pause\n");
    }

    while(application_is_in_pause)
    {

        SDL_WaitEvent(&event);

        switch(event.type)
        {
        case SDL_QUIT:
            end_of_game = true;
            break;
        }

        Uint8 *keys = SDL_GetKeyState(NULL);

        if(event.type == SDL_QUIT or
                keys[SDLK_p] or keys[SDLK_SPACE] or
                keys[SDLK_q] or keys[SDLK_ESCAPE])
        {
            application_is_in_pause = false;
            printf("Exiting the pause\n");
        }

        if(keys[SDLK_q] or keys[SDLK_ESCAPE])
        {
            end_of_game = true;
        }

        if( keys[SDLK_s])
        {
            save_screenshot();
        }

        if( keys[SDLK_RIGHT] or keys[SDLK_DOWN] or  keys[SDLK_PAGEDOWN])
        {
            printf("Moving one frame forwards\n");
            // during next cycle will be in pause again
            // this allow to move one frame at a time
            should_stay_in_pause = true;
            break;
        }

    } // end of "while application is in pause"

    return end_of_game;
}


void  BaseSdlGui::print_inputs_instructions() const
{

    // for the color tricks see
    // http://linuxgazette.net/issue65/padala.html

    printf("User interface keyboards inputs:\n");
    printf("\t\033[1;32mpause\033[0m\t\t the application with \tSPACE or P\n");
    printf("\t\033[1;32mquit\033[0m\t\t the application with \tESC, Q or closing the window\n");
    printf("\t\033[1;32mscreenshots\033[0m\t are saved with \tS\n");
    printf("\t\033[1;32mview_mode\033[0m\t is changed with \t1, 2, 3, etc.\n");

    return;
}


/// @returns true if the application should stop
bool BaseSdlGui::update()
{
    // show stereo and disparity map side to side

    const bool end_of_game = process_inputs();

    if(end_of_game == false)
    {
        // call the current drawing function
        const drawing_function_t &drawing_function = current_view.first;
        drawing_function();
        // draw function is responsable of drawing on the gil images
        // this is then moved into the screen via blit_to_screen()
        blit_to_screen();

        if(save_all_screenshots)
        {
            save_screenshot();
        }
    }

    return end_of_game;
}


void BaseSdlGui::draw_empty_screen()
{

    boost::gil::rgb8_pixel_t magenta(255,0,255), cyan(0,255,255);

    boost::gil::fill_pixels(screen_left_view, magenta);
    boost::gil::fill_pixels(screen_right_view, cyan);

    return;
}



void BaseSdlGui::blit_to_screen()
{

    // write output_image into SDL screen
    const boost::gil::rgb8_view_t &view = this->screen_image_view;

    const int depth = 24;
    //const int pitch = (view.row_begin(1) - view.row_begin(0)) / sizeof(bost::gil::rgb8c_pixel_t);
    const int pitch = view.width()*3;


    // SDL interprets each pixel as a 32-bit number, so our masks must depend
    // on the endianness (byte order) of the machine
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    //const Uint32  r_mask = 0xff000000, g_mask = 0x00ff0000, b_mask = 0x0000ff00, a_mask = 0x000000ff;
    const Uint32 r_mask = 0x00ff0000, g_mask = 0x0000ff00, b_mask = 0x000000ff, a_mask = 0x00000000;
#else
    //const Uint32 r_mask = 0x000000ff, g_mask = 0x0000ff00, b_mask = 0x00ff0000, a_mask = 0xff000000;
    const Uint32 r_mask = 0x0000ff, g_mask = 0x0000ff00, b_mask = 0x00ff0000, a_mask = 0x00000000;
#endif

    SDL_Surface *surface_p =
            SDL_CreateRGBSurfaceFrom( boost::gil::interleaved_view_get_raw_data(view),
                                     view.width(), view.height(),
                                     depth, pitch,
                                     r_mask, g_mask, b_mask, a_mask);
    if (SDL_MUSTLOCK(screen_p))
    {
        if (SDL_LockSurface(screen_p) < 0)
        {
            log_error() << "Couldn't lock SDL screen: " << SDL_GetError() << endl;
            throw std::runtime_error("Failed to lock SDL screen");
        }
    }

    // copy the full image to the top left corner of the screen
    SDL_BlitSurface(surface_p, NULL, screen_p, NULL);
    //SDL_FillRect(screen_p, NULL, 1000); // fill with blue

    if (SDL_MUSTLOCK(screen_p))
    {
        SDL_UnlockSurface(screen_p);
    }

    SDL_Flip(screen_p);

    SDL_FreeSurface(surface_p);

    return;
}

void  BaseSdlGui::save_screenshot()
{
    using namespace boost::filesystem;
    const boost::filesystem::path &recording_path = base_application.get_recording_path();

    if(save_all_screenshots == true and recorded_first_image == false)
    {
        recorded_first_image = true;
        printf("Will save all the screenshots in the recordings directory\n");
    }

    // retrieve screenshot name --
    const int frame_number = base_application.get_current_frame_number();
    const  boost::filesystem::path screenshot_filename =
            recording_path / boost::str(boost::format("screenshot_frame_%i.png") % frame_number);

    // record the screenshot --
    boost::gil::png_write_view(screenshot_filename.string(), screen_image_view);

    if(save_all_screenshots == false)
    {
        printf("Recorded image %s\n", screenshot_filename.string().c_str());
    }

    return;
}


} // end of namespace doppia


