#include "VideoInputFactory.hpp"

#include "VideoFromFiles.hpp"
#include "preprocessing/CpuPreprocessor.hpp"
#include "calibration/StereoCameraCalibration.hpp"

#include "helpers/get_option_value.hpp"

namespace doppia
{

using namespace std;
using namespace boost::program_options;

options_description
VideoInputFactory::get_args_options()
{
    options_description desc("VideoInputFactory options");

    desc.add_options()

            ("video_input.source", value<string>()->default_value("directory"),
             "video input source: directory, movie or camera")

            ("video_input.calibration_filename", value<string>(),
             "filename protocol buffer text description of the stereo rig calibration. See calibration.proto for mor details")

            ;

    desc.add(AbstractVideoInput::get_args_options());
    desc.add(VideoFromFiles::get_args_options());
    desc.add(AbstractPreprocessor::get_args_options());
    desc.add(CpuPreprocessor::get_args_options());

    //        desc.add(get_section_options("video_input", "AbstractVideoInput options", AbstractVideoInput::get_args_options()));
    //        desc.add(get_section_options("video_input", "VideoFromFiles options", VideoFromFiles::get_args_options()));
    //        desc.add(get_section_options("video_input", "AbstractPreprocessor options", AbstractPreprocessor::get_args_options()));
    //   desc.add(get_section_options("video_input", "CpuPreprocessor options", CpuPreprocessor::get_args_options()));

    return desc;
}


shared_ptr<AbstractPreprocessor> new_preprocessor_instance(const variables_map &options, AbstractVideoInput &video_input)
{
    shared_ptr<AbstractPreprocessor> preprocess_p;
    preprocess_p.reset(new CpuPreprocessor(video_input.get_left_image().dimensions(),
                                           video_input.get_stereo_calibration(),
                                           options));
    return preprocess_p;
}

AbstractVideoInput*
VideoInputFactory::new_instance(const variables_map &options)
{

    // create the stereo matcher instance
    const string source = get_option_value<std::string>(options, "video_input.source");
    const string calibration_filename = get_option_value<std::string>(options, "video_input.calibration_filename");

    // the calibration object is temporary, used only to precompute data inside the CpuPreprocessor
    const shared_ptr<StereoCameraCalibration> stereo_calibration_p(new StereoCameraCalibration(calibration_filename));

    AbstractVideoInput* video_source_p = NULL;
    if (source.compare("directory") == 0)
    {
        VideoFromFiles * const video_from_files_p = new VideoFromFiles(options, stereo_calibration_p);
        video_source_p = video_from_files_p;

        video_from_files_p->set_preprocessor(new_preprocessor_instance(options, *video_source_p));
    }
    else if (source.compare("movie") == 0)
    {
        throw std::runtime_error("movie video input is not yet implemented");
    }
    else if (source.compare("camera") == 0)
    {
        throw std::runtime_error("camera video input is not yet implemented");
    }
    else
    {
        throw std::runtime_error("Unknown 'video_input.source' value");
    }


    return video_source_p;
}


} // end of namespace doppia
