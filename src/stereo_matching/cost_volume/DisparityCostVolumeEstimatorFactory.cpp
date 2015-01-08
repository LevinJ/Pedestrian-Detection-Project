#include "DisparityCostVolumeEstimatorFactory.hpp"

#include "AbstractDisparityCostVolumeEstimator.hpp"

#include "DisparityCostVolumeEstimator.hpp"
#include "FastDisparityCostVolumeEstimator.hpp"
#include "DisparityCostVolumeFromDepthMap.hpp"

#include "stereo_matching/SimpleBlockMatcher.hpp"
#include "stereo_matching/ConstantSpaceBeliefPropagation.hpp"
#include "stereo_matching/OpenCvStereo.hpp"


#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <stdexcept>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "DisparityCostVolumeEstimatorFactory");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "DisparityCostVolumeEstimatorFactory");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "DisparityCostVolumeEstimatorFactory");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "DisparityCostVolumeEstimatorFactory");
}

} // end of anonymous namespace

namespace doppia {

using namespace boost::program_options;

options_description
DisparityCostVolumeEstimatorFactory::get_args_options()
{

    options_description desc("DisparityCostVolumeEstimatorFactory options");

    desc.add_options()
            ("cost_volume.method", value<string>()->default_value("fast_pixelwise"),
             "stereo matching methods: pixelwise, fast_pixelwise, " \
             "csbp or simple_sad" )
            ;

    //desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(AbstractStereoBlockMatcher::get_args_options());
    desc.add(SimpleBlockMatcher::get_args_options());
    desc.add(ConstantSpaceBeliefPropagation::get_args_options());
    desc.add(OpenCvStereo::get_args_options());
    desc.add(DisparityCostVolumeEstimator::get_args_options());

    return desc;
}


AbstractDisparityCostVolumeEstimator*
DisparityCostVolumeEstimatorFactory::new_instance(const variables_map &options)
{

    const string pixels_matching_method = get_option_value<std::string>(options, "pixels_matching");

    const bool supported_by_fast_estimator =
            (pixels_matching_method.compare("sad") == 0) or
            (pixels_matching_method.compare("ssd") == 0);

    const string method = get_option_value<std::string>(options, "cost_volume.method");

    AbstractDisparityCostVolumeEstimator* cost_volume_estimator_p = NULL;
    boost::shared_ptr<AbstractStereoMatcher> stereo_matcher_p;

    if (method.empty() or (method.compare("fast_pixelwise") == 0))
    {

        const bool use_fast_disparity_cost = true;
        //const bool use_fast_disparity_cost = false;
        if(use_fast_disparity_cost and supported_by_fast_estimator)
        {
            cost_volume_estimator_p = new FastDisparityCostVolumeEstimator(options);
        }
        else
        {
            log_warning() << "Requested FastDisparityCostVolumeEstimator but using DisparityCostVolumeEstimator instead" << std::endl;
            cost_volume_estimator_p = new DisparityCostVolumeEstimator(options);
        }
    }
    else if (method.compare("pixelwise") == 0)
    {
        cost_volume_estimator_p = new DisparityCostVolumeEstimator(options);
    }
    else if (method.compare("simple_sad") == 0 or
             method.compare("simple_ssd") == 0 )
    {
        log_info() << "Disparity cost volume will be estimated from the depth map provided by " \
                      "block matching using " << method << std::endl;

        stereo_matcher_p.reset(new SimpleBlockMatcher(options));
        cost_volume_estimator_p = new DisparityCostVolumeFromDepthMap(options, stereo_matcher_p);
    }
    else if (method.compare("opencv_sad") == 0 or
             method.compare("opencv_bm") == 0 or
             method.compare("opencv_gc") == 0 or
             method.compare("opencv_csbp") == 0 )
    {
        log_info() << "Disparity cost volume will be estimated from the depth map provided by " \
                      "OpenCv using " << method << std::endl;

        stereo_matcher_p.reset(new OpenCvStereo(options));
        cost_volume_estimator_p = new DisparityCostVolumeFromDepthMap(options, stereo_matcher_p);
    }
    else if (method.compare("csbp") == 0  )
    {
        log_info() << "Disparity cost volume will be estimated from the depth map provided by " << method << std::endl;
        stereo_matcher_p.reset(new ConstantSpaceBeliefPropagation(options));
        cost_volume_estimator_p = new DisparityCostVolumeFromDepthMap(options, stereo_matcher_p);
    }
    else
    {
        log_error() << "DisparityCostVolumeEstimatorFactory received cost_volume.method value == " << method << std::endl;
        throw std::invalid_argument("Unknown 'cost_volume.method' value");
    }

    return cost_volume_estimator_p;
}



} // end of namespace doppia
