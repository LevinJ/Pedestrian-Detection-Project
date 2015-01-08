#ifndef ABSTRACTPREPROCESSOR_HPP
#define ABSTRACTPREPROCESSOR_HPP

#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "video_input/AbstractVideoInput.hpp"

#include <boost/program_options.hpp>

namespace doppia
{

/**
 * Image preprocessing class:
 *  - Unbayering
 *  - Undistortion
 *  - Rectification
 *  - Smoothing
 *
 * Based on code from Andreas Ess
 **/
class AbstractPreprocessor
{
public:

    typedef AbstractVideoInput::input_image_view_t input_image_view_t;
    typedef AbstractVideoInput::input_image_t::view_t output_image_view_t;
    typedef AbstractVideoInput::input_image_view_t::point_t dimensions_t;

    static boost::program_options::options_description get_args_options();

    AbstractPreprocessor(const dimensions_t &dimensions,
                         const StereoCameraCalibration &stereo_calibration,
                         const boost::program_options::variables_map &options);

    virtual ~AbstractPreprocessor();

    void run(const input_image_view_t& input,
             const output_image_view_t &output);


    ///  input and ouput images maybe the same image
    /// @param camera_index indicates which camera is being given as input. The undistortion and rectification steps are camera dependent.
    virtual void run(const input_image_view_t& input, const int camera_index,
                     const output_image_view_t &output) = 0;


    /// @returns the stereo calibration corresponding to the post-processed images
    virtual const StereoCameraCalibration& get_post_processing_calibration() const = 0;


protected:

    const dimensions_t input_dimensions;
    const StereoCameraCalibration &stereo_calibration;
    bool do_unbayering, do_undistortion, do_rectification, do_smoothing;

};

} // end of namespace doppia

#endif // ABSTRACTPREPROCESSOR_HPP
