#ifndef VIDEOFROMFILE_HPP
#define VIDEOFROMFILE_HPP


#include "AbstractVideoInput.hpp"

#include "preprocessing/AbstractPreprocessor.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <boost/thread.hpp>


namespace doppia
{

using boost::shared_ptr;

///
/// Loads images from a video stream stored as a set of images.
/// Supports a preprocessor object for things like unbayering, rectification, etc ...
///
/// Based on Andreas Ess code
///
class VideoFromFiles : public AbstractVideoInput
{
public:


    static boost::program_options::options_description get_args_options();

    VideoFromFiles(const boost::program_options::variables_map &options,
                   const shared_ptr<StereoCameraCalibration> &stereo_calibration_p);
    ~VideoFromFiles();

    bool next_frame();

    bool previous_frame();

    const input_image_view_t &get_left_image();
    const input_image_view_t &get_right_image();

    int get_number_of_frames();

    bool set_frame(const int frame_number);

    /// VideoFromFiles will take ownership over the preprocessor object.
    /// When VideoFromFiles object is destroyed, AbstractPreprocessor will probably be destroyed too..
    void set_preprocessor(const shared_ptr<AbstractPreprocessor> &preprocessor);
    const shared_ptr<AbstractPreprocessor> &get_preprocessor() const;

    const StereoCameraCalibration& get_stereo_calibration() const;

protected:

    bool read_frame_from_disk(const int frame_number,
                              input_image_t &left_image, input_image_t &right_image,
                              input_image_view_t &left_view, input_image_view_t &right_view);


    void read_future_frame_thead();

    /// Base for directory and filename
    std::string left_filename_mask, right_filename_mask;
    boost::format left_filename_format, right_filename_format;

    int start_frame, end_frame, total_number_of_frames;

    shared_ptr<StereoCameraCalibration> stereo_calibration_p;

    /// Optional preprocessor object
    shared_ptr<AbstractPreprocessor> preprocessor_p;

    input_image_t left_image, right_image;
    input_image_view_t left_image_view, right_image_view;

    boost::barrier read_future_frame_start_barrier, read_future_frame_ended_barrier;
    boost::thread image_reading_thread;

    /// shared_future instead of unique_future to work around lack of move support in pre C++0x compilers
    boost::shared_future<bool> read_next_image_future;
    bool found_future_image;
    int future_image_frame_number;
    input_image_t future_left_image, future_right_image;
    input_image_view_t future_left_image_view, future_right_image_view;

};

} // end of namespace doppia

#endif
