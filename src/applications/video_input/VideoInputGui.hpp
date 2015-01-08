#ifndef VideoInputGui_HPP
#define VideoInputGui_HPP

#include "applications/BaseSdlGui.hpp"

namespace doppia
{

class VideoInputApplication; // forward declaration


class VideoInputGui: public BaseSdlGui
{

    VideoInputApplication &application;


public:
    static boost::program_options::options_description get_args_options();

    VideoInputGui(VideoInputApplication &application, const boost::program_options::variables_map &options);
    ~VideoInputGui();

protected:

    void draw_video_input();

};

} // end of namespace doppia

#endif // VideoInputGui_HPP
