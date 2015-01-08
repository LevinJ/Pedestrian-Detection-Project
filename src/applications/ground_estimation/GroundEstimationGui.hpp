#ifndef GroundEstimationGui_HPP
#define GroundEstimationGui_HPP

#include "applications/BaseSdlGui.hpp"

#include <boost/program_options.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/function.hpp>


// forward declaration
class SDL_Surface;

namespace doppia
{

using namespace boost;

// forward declarations
class GroundEstimationApplication;
class GroundPlaneEstimator;
class FastGroundPlaneEstimator;

class GroundEstimationGui: public BaseSdlGui
{

    GroundEstimationApplication &application;

public:
    static boost::program_options::options_description get_args_options();

    GroundEstimationGui(GroundEstimationApplication &application, const boost::program_options::variables_map &options);
    ~GroundEstimationGui();

private:

    int max_disparity;

    void draw_video_input();
    void draw_ground_plane_estimation();
};

} // end of namespace doppia

#endif // GroundEstimationGui_HPP
