#ifndef OBJECTS_DETECTION_OBJECTSDETECTIONLIBGUI_HPP
#define OBJECTS_DETECTION_OBJECTSDETECTIONLIBGUI_HPP

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#if not defined(OBJECTS_DETECTION_WITH_UI_LIB)
namespace objects_detection {

class FakeObjectsDetectionLibGui
{
public:
    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    void set_monocular_input(input_image_const_view_t &);
    void set_left_input(input_image_const_view_t &);
    void set_right_input(input_image_const_view_t &);

    void update();
};

// we cannot use a typedef since it would be incompatible with forward declarations,
// we use a child class instead.
//typedef FakeObjectsDetectionLibGui ObjectsDetectionLibGui;
class ObjectsDetectionLibGui: public FakeObjectsDetectionLibGui
{
public:
};

} // namespace objects_detection


#else // OBJECTS_DETECTION_WITH_UI_LIB is defined

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <map>


// forward declarations
struct SDL_Surface;

namespace doppia {
// forward declarations
class MetricStereoCamera;
class AbstractObjectsDetector;
class AbstractStixelWorldEstimator;
}

namespace objects_detection {




class ObjectsDetectionLibGui
{
public:

    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    ObjectsDetectionLibGui(const int input_width, const int input_height,
                           boost::shared_ptr<doppia::MetricStereoCamera> stereo_camera_p,
                           boost::shared_ptr<doppia::AbstractObjectsDetector> objects_detector_p,
                           boost::shared_ptr<doppia::AbstractStixelWorldEstimator> stixel_world_estimator_p);
    ~ObjectsDetectionLibGui();


    void set_monocular_input(input_image_const_view_t &view);

    void set_left_input(input_image_const_view_t &view);
    void set_right_input(input_image_const_view_t &view);

    /// @returnss false if the application should stop
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
    boost::shared_ptr<doppia::AbstractObjectsDetector> objects_detector_p;
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

    float max_detection_score;

    void draw_video_input();
    void draw_detections();

#if not defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    void draw_stixel_world();
    void draw_ground_plane_estimation();
    void draw_stixels_estimation();
#endif

};

} // end of namespace objects_detection

#endif // end of "if OBJECTS_DETECTION_WITH_UI_LIB is not defined"

#endif // OBJECTS_DETECTION_OBJECTSDETECTIONLIBGUI_HPP
