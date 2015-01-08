#ifndef StixelWorldGui_HPP
#define StixelWorldGui_HPP

#include "applications/BaseSdlGui.hpp"

#include "video_input/AbstractVideoInput.hpp"

#include <boost/program_options.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/function.hpp>

#include "stereo_matching/stixels/Stixel.hpp" // for stixels_t
#include <Eigen/Core>

// forward declarations
class SDL_Surface;

namespace doppia
{

// forward declarations
class AbstractVideoInput;
class AbstractStixelWorldEstimator;
class AbstractStixelMotionEstimator;
class GroundPlaneEstimator;
class FastGroundPlaneEstimator;
class StixelsEstimator;

class StixelWorldGui: public BaseSdlGui
{

protected:
    // processing modules
    shared_ptr<AbstractVideoInput> video_input_p;
    shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p;
    shared_ptr<AbstractStixelMotionEstimator> stixel_motion_estimator_p;

public:
    static boost::program_options::options_description get_args_options(const bool add_base_gui_options = true);

    StixelWorldGui(
            BaseApplication &application_,
            shared_ptr<AbstractVideoInput> video_input_p_,
            shared_ptr<AbstractStixelWorldEstimator> stixel_world_estimator_p_,
            shared_ptr<AbstractStixelMotionEstimator> stixel_motion_estimator_p_,
            const boost::program_options::variables_map &options,
            const bool init_gui = true);

    ~StixelWorldGui();

protected:

    bool process_inputs();

    bool colorize_disparity;

    int max_disparity;
//    int max_possible_motion_in_pixels;

    void draw_video_input();
    void draw_disparity_map();
    void draw_ground_plane_estimation();
    void draw_stixels_estimation();
    void draw_stixels_height_estimation();
    void draw_stixel_world();

    void draw_stixel_motion_tracks();
    void draw_stixel_motion_matrix();
    void draw_stixel_motion(); // in accordance with plot_stixels_motion() in python evaluation code

    ///  Convert the disparity map values into a false color RGB image
    void colorize_disparity_map(const boost::gil::rgb8_planar_view_t &view);

    void save_screenshot();
};



void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self);

void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 AbstractVideoInput &video_input,
                                 BaseSdlGui &self);

void draw_stixels_height_estimation(const stixels_t &the_stixels,
                                    const Eigen::MatrixXf &height_cost,
                                    const std::vector<int> &u_v_ground_obstacle_boundary,
                                    AbstractVideoInput &video_input,
                                    BaseSdlGui &self);

void draw_stixels_estimation(const StixelsEstimator &estimator,
                             AbstractVideoInput &video_input,
                             BaseSdlGui &self);

void draw_stixel_motion_tracks(const AbstractStixelMotionEstimator& motion_estimator,
                               AbstractVideoInput& video_input,
                               BaseSdlGui& self );

void draw_stixel_motion_matrix(const AbstractStixelMotionEstimator& motion_estimator,
                               AbstractVideoInput& video_input,
                               BaseSdlGui& self );

void draw_stixel_motion(const AbstractStixelMotionEstimator& motion_estimator,
                        AbstractVideoInput& video_input,
                        BaseSdlGui& self );

void draw_stixel_world(const stixels_t &the_stixels,
                       AbstractVideoInput &video_input,
                       BaseSdlGui &self);


void draw_stixel_world(const stixels_t &the_stixels, const Eigen::MatrixXf &depth_map,
                       AbstractVideoInput &video_input,
                       BaseSdlGui &self);


} // end of namespace doppia

#endif // StixelWorldGui_HPP
