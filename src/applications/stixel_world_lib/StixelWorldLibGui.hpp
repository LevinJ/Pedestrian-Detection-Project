#ifndef StixelWorldLibGui_HPP
#define StixelWorldLibGui_HPP

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <map>

// forward declarations
struct SDL_Surface;

namespace doppia {
// forward declarations
class MetricStereoCamera;
class AbstractStixelWorldEstimator;
}

namespace stixel_world {




class StixelWorldLibGui
{
public:
    StixelWorldLibGui(const int input_width, const int input_height,
                           boost::shared_ptr<doppia::MetricStereoCamera> stereo_camera_p,
                           boost::shared_ptr<doppia::AbstractStixelWorldEstimator> stixel_world_estimator_p);
    ~StixelWorldLibGui();

    void set_left_input(boost::gil::rgb8c_view_t &view);
    void set_right_input(boost::gil::rgb8c_view_t &view);

    /// Returns false if the application should stop
    bool update();

protected:

    void init_gui(const std::string &title, const int input_width, const int input_height);
    void resize_gui(const int input_width, const int input_height);

    /// @returns true if the application should stop
    virtual bool process_inputs();
    void print_inputs_instructions() const;

    /// input elements
    boost::gil::rgb8c_view_t input_left_view, input_right_view;
    boost::shared_ptr<doppia::MetricStereoCamera> stereo_camera_p;
    boost::shared_ptr<doppia::AbstractStixelWorldEstimator> stixel_world_estimator_p;

    /// SDL screen surface
    SDL_Surface *screen_p;
    boost::gil::rgb8_image_t screen_image;
    boost::gil::rgb8_view_t screen_image_view;
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

    void draw_video_input();
    void draw_stixel_world();
    void draw_ground_plane_estimation();
    void draw_stixels_estimation();

};

} // end of namespace init_stixel_world

#endif // StixelWorldLibGui_HPP
