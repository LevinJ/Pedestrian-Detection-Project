
#include "AbstractPreprocessor.hpp"

#include <boost/program_options.hpp>

#include "helpers/get_option_value.hpp"

namespace doppia
{
    using namespace boost;

    program_options::options_description AbstractPreprocessor::get_args_options()
    {
         program_options::options_description desc("AbstractPreprocessor options");


            desc.add_options()

                    ("preprocess.unbayer",
                     program_options::value<bool>()->default_value(false),
                     "unbayer the input images")

                    ("preprocess.undistort",
                     program_options::value<bool>()->default_value(true),
                     "undistort the input images")

                    ("preprocess.rectify",
                     program_options::value<bool>()->default_value(false),
                     "rectify the input images")

                    ("preprocess.smooth",
                     program_options::value<bool>()->default_value(false),
                     "smooth the input images")

                    ;


        return desc;
    }


AbstractPreprocessor::AbstractPreprocessor(const dimensions_t &dimensions,
                                           const StereoCameraCalibration &_stereo_calibration,
                                           const program_options::variables_map &options)
    :input_dimensions(dimensions), stereo_calibration(_stereo_calibration)
{

    do_unbayering = get_option_value<bool>(options, "preprocess.unbayer");
    do_undistortion = get_option_value<bool>(options, "preprocess.undistort");
    do_rectification = get_option_value<bool>(options, "preprocess.rectify");
    do_smoothing = get_option_value<bool>(options, "preprocess.smooth");


    return;
}



AbstractPreprocessor::~AbstractPreprocessor()
{
    // nothing to do here
    return;
}

void AbstractPreprocessor::run(const input_image_view_t& input,
         const output_image_view_t &output)
{
    run(input, 0, output);
    return;
}


} // end of namespace doppia


