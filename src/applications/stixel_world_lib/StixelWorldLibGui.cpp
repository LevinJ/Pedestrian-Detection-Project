#include "StixelWorldLibGui.hpp"

#include "video_input/AbstractVideoInput.hpp"
#include "video_input/MetricStereoCamera.hpp"

#include "stereo_matching/stixels/AbstractStixelWorldEstimator.hpp"
#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/FastStixelsEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/FastStixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/ImagePlaneStixelsEstimator.hpp"

#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_stixel_world.hpp"
#include "drawing/gil/draw_the_ground_corridor.hpp"

#include <SDL/SDL.h>

#include <boost/format.hpp>

#include <boost/bind.hpp>
#include <boost/array.hpp>

#include <iostream> // for std::cerr

namespace stixel_world {

using namespace doppia;
using boost::array;

StixelWorldLibGui::StixelWorldLibGui(const int input_width, const int input_height,
                                               boost::shared_ptr<doppia::MetricStereoCamera> stereo_calibration,
                                                boost::shared_ptr<doppia::AbstractStixelWorldEstimator> stixel_world_estimator)
    : stereo_camera_p(stereo_calibration),
      stixel_world_estimator_p(stixel_world_estimator)
{
    init_gui("StixelWorldLibGui", input_width, input_height);

    // populate the views map --
    views_map[SDLK_0] = view_t(boost::bind(&StixelWorldLibGui::draw_empty_screen, this), "draw_empty_screen");

    views_map[SDLK_1] = view_t(boost::bind(&StixelWorldLibGui::draw_video_input, this), "draw_video_input");
    views_map[SDLK_2] = view_t(boost::bind(&StixelWorldLibGui::draw_stixel_world, this), "draw_stixel_world");
    views_map[SDLK_3] = view_t(boost::bind(&StixelWorldLibGui::draw_ground_plane_estimation, this), "draw_ground_plane_estimation");
    views_map[SDLK_4] = view_t(boost::bind(&StixelWorldLibGui::draw_stixels_estimation, this), "draw_stixels_estimation");
    //views_map[SDLK_7] = view_t(boost::bind(&StixelWorldLibGui::draw_stixels_height_estimation, this), "draw_stixels_height_estimation");
    //views_map[SDLK_8] = view_t(boost::bind(&StixelWorldLibGui::draw_disparity_map, this), "draw_disparity_map");
    //views_map[SDLK_9] = view_t(boost::bind(&StixelWorldLibGui::draw_gpu_stixel_world, this), "draw_gpu_stixel_world");

    // draw the first image --
    draw_video_input();

    // set the initial view mode --
    current_view = views_map[SDLK_1]; // video input

    if(stixel_world_estimator_p)
    {
        current_view = views_map[SDLK_4]; // draw stixels world
    }

    return;
}


StixelWorldLibGui::~StixelWorldLibGui()
{
    // nothing to do here
    return;
}


void StixelWorldLibGui::set_left_input(boost::gil::rgb8c_view_t &view)
{
    input_left_view = view;
    return;
}


void StixelWorldLibGui::set_right_input(boost::gil::rgb8c_view_t &view)
{
    input_right_view = view;
    return;
}


void StixelWorldLibGui::init_gui(const std::string &title, const int input_width, const int input_height)
{

    // create the application window
    SDL_Init(SDL_INIT_VIDEO);
    resize_gui(input_width, input_height);

    SDL_WM_SetCaption(title.c_str(), title.c_str());

    print_inputs_instructions();

    // set the initial view mode -
    current_view = views_map[SDLK_0];

    return;
}

void StixelWorldLibGui::resize_gui(const int input_width, const int input_height)
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


bool StixelWorldLibGui::process_inputs()
{
    bool end_of_game = false;

    SDL_Event event;

    while ( SDL_PollEvent(&event) )
    {
        switch(event.type)
        {
        case SDL_VIDEORESIZE:
            // we do not want the user the play around with the window size
            throw std::runtime_error("StixelWorldLibGui::process_inputs does not support window resizing");
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


    return end_of_game;
}


void  StixelWorldLibGui::print_inputs_instructions() const
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
bool StixelWorldLibGui::update()
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
    }

    return end_of_game;
}


void StixelWorldLibGui::draw_empty_screen()
{

    boost::gil::rgb8_pixel_t magenta(255,0,255), cyan(0,255,255);

    boost::gil::fill_pixels(screen_left_view, magenta);
    boost::gil::fill_pixels(screen_right_view, cyan);

    return;
}



void StixelWorldLibGui::blit_to_screen()
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
            std::cerr << "Couldn't lock SDL screen: " << SDL_GetError() << std::endl;
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


void StixelWorldLibGui::draw_video_input()
{
    copy_and_convert_pixels(input_left_view, screen_left_view);
    copy_and_convert_pixels(input_right_view, screen_right_view);
    return;
}


void StixelWorldLibGui::draw_stixel_world()
{
    // draw the ground plane and the stixels

    StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p.get());

    if(the_stixel_world_estimator_p != NULL)
    {

        StixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< StixelsEstimatorWithHeightEstimation *>(the_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p)
        {
            doppia::draw_stixel_world(the_stixel_world_estimator_p->get_stixels(),
                                      the_stixels_estimator_p->get_depth_map(),
                                      input_left_view, screen_left_view, screen_right_view);

        }
        else
        {
            doppia::draw_stixel_world(the_stixel_world_estimator_p->get_stixels(),
                                      input_left_view, input_right_view,
                                      screen_left_view, screen_right_view);
        }

    }
    else if(the_fast_stixel_world_estimator_p != NULL)
    {
        FastStixelsEstimatorWithHeightEstimation *the_stixels_estimator_p =
                dynamic_cast< FastStixelsEstimatorWithHeightEstimation *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());
        if(the_stixels_estimator_p != NULL)
        {
            doppia::draw_stixel_world(the_fast_stixel_world_estimator_p->get_stixels(),
                                      the_stixels_estimator_p->get_disparity_likelihood_map(),
                                      input_left_view, screen_left_view, screen_right_view);
        }
        else
        {
            doppia::draw_stixel_world(the_fast_stixel_world_estimator_p->get_stixels(),
                                      input_left_view, input_right_view,
                                      screen_left_view, screen_right_view);
        }
    }
    else
    {
        // we simply freeze the left screen
        copy_and_convert_pixels(input_right_view, screen_right_view);
    }


    return;
} // end of StixelWorldLibGui::draw_stixel_world()


void StixelWorldLibGui::draw_ground_plane_estimation()
{
    // draw the ground plane and the stixels

    if(stixel_world_estimator_p)
    {

        AbstractStixelWorldEstimator &stixel_world_estimator = *(stixel_world_estimator_p);
        BaseGroundPlaneEstimator *base_ground_plane_estimator_p = NULL;

        StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(&stixel_world_estimator);
        FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(&stixel_world_estimator);

        if(the_stixel_world_estimator_p != NULL)
        {
            base_ground_plane_estimator_p = the_stixel_world_estimator_p->ground_plane_estimator_p.get();
        }
        else if(the_fast_stixel_world_estimator_p != NULL)
        {
            base_ground_plane_estimator_p = the_fast_stixel_world_estimator_p->ground_plane_estimator_p.get();
        }

        // Left screen --
        {
            // copy left screen image ---
            copy_and_convert_pixels(input_left_view, screen_left_view);

            // add the ground bottom and top corridor ---
            const GroundPlane &ground_plane = stixel_world_estimator.get_ground_plane();
            const MetricCamera &camera = stereo_camera_p->get_left_camera();
            draw_the_ground_corridor(screen_left_view, camera, ground_plane);

            if(false)
            {
                // draw the stixels ---
                draw_the_stixels(screen_left_view,
                                 stixel_world_estimator.get_stixels());
            }

            // add our prior on the ground area ---
            const bool draw_prior_on_ground_area = true;
            if(draw_prior_on_ground_area and base_ground_plane_estimator_p != NULL)
            {
                const std::vector<int> &ground_object_boundary_prior =
                        base_ground_plane_estimator_p->get_ground_area_prior();
                const std::vector<int> &boundary = ground_object_boundary_prior;
                for(std::size_t u=0; u < boundary.size(); u+=1)
                {
                    const int &disparity = boundary[u];
                    screen_left_view(u, disparity) = rgb8_colors::cyan;
                }
            }

        } // end of left screen -

        // Right screen --
        {
            // will draw on the right screen
            if(the_stixel_world_estimator_p != NULL)
            {
                draw_ground_plane_estimator(*(the_stixel_world_estimator_p->ground_plane_estimator_p),
                                            input_right_view, stereo_camera_p->get_calibration(), screen_right_view);
            }
            else  if(the_fast_stixel_world_estimator_p != NULL)
            {
                draw_ground_plane_estimator(*(the_fast_stixel_world_estimator_p->ground_plane_estimator_p),
                                            input_right_view, stereo_camera_p->get_calibration(), screen_right_view);
            }
        }

    } // end of "the_stixel_world_estimator_p != NULL"
    else
    {
        // we simply freeze the screen
    }

    return;
} // end of StixelWorldLibGui::draw_ground_plane_estimation


void StixelWorldLibGui::draw_stixels_estimation()
{
    // draw the ground plane and the stixels

    StixelWorldEstimator *the_stixel_world_estimator_p = dynamic_cast< StixelWorldEstimator *>(stixel_world_estimator_p.get());
    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p.get());

    if(the_stixel_world_estimator_p != NULL)
    {
        doppia::draw_stixels_estimation(*(the_stixel_world_estimator_p->stixels_estimator_p),
                                        input_left_view, screen_left_view, screen_right_view);
    }
    else if(the_fast_stixel_world_estimator_p != NULL)
    {
        const FastStixelsEstimator *fast_estimator_p = \
                dynamic_cast<FastStixelsEstimator *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());

        const ImagePlaneStixelsEstimator *uv_estimator_p = \
                dynamic_cast<ImagePlaneStixelsEstimator *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());

        if(fast_estimator_p)
        {
            doppia::draw_stixels_estimation(*fast_estimator_p,
                                            input_left_view, screen_left_view, screen_right_view);
        }
        else if(uv_estimator_p)
        {
            doppia::draw_stixels_estimation(*uv_estimator_p,
                                            input_left_view, screen_left_view, screen_right_view);
        }
        else
        {
            // simply freeze the screen
        }
    }
    else
    { // the_stixel_world_estimator_p == NULL
        // simply freeze the screen
    }

    return;
} // end of StixelWorldLibGui::draw_stixels_estimation


} // end of namespace init_stixel_world
