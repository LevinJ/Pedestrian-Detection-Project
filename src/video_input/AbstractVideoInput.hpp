#ifndef ABSTRACTVIDEOINPUT_HPP
#define ABSTRACTVIDEOINPUT_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>


namespace doppia
{

// forward declarations
class StereoCameraCalibration;
class MetricStereoCamera;


class AbstractVideoInput
{

public:
    typedef boost::gil::rgb8_image_t input_image_t;

    typedef boost::gil::rgb8c_view_t input_image_view_t;

    typedef input_image_view_t::point_t dimensions_t;

    static boost::program_options::options_description get_args_options();

    AbstractVideoInput(const boost::program_options::variables_map &options);
    virtual ~AbstractVideoInput() = 0;

    virtual const input_image_view_t &get_left_image() = 0;
    virtual const input_image_view_t &get_right_image() = 0;

    /// Advance in stream, @returns true if successful
    virtual bool next_frame() = 0;

    /// Go back in stream
    virtual bool previous_frame() = 0;

    /// @returns number of frames in this stream
    virtual int get_number_of_frames() = 0;


    /// Set current absolute frame. Calling set_frame will update the left and right images.
    /// @returns true if update succeeded.
    virtual bool set_frame(const int frame_number) = 0;

    /// Get current frame number
    int get_current_frame_number() const;

    /// The VideoInput object is the sole holder of the stereo camera calibration.
    /// This data will be used mainly internally, but can be used by other sucessive algorithms.
    virtual const StereoCameraCalibration& get_stereo_calibration() const = 0;

    const MetricStereoCamera& get_metric_camera() const;

    float camera_height; // in meters
    float camera_pitch, camera_roll; // in radians

protected:

    int current_frame_number;

    /// defined as mutable to allow lazy initialization when
    /// calling the "mainly const" get_metric_camera method
    mutable boost::scoped_ptr<MetricStereoCamera> metric_camera_p;

};

} // end of namespace doppia

#endif // ABSTRACTVIDEOINPUT_HPP
