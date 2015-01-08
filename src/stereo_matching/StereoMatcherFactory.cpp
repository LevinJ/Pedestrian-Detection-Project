
#include "AbstractStereoMatcher.hpp"

#include "StereoMatcherFactory.hpp"
#include "CensusStereoMatcher.hpp"
#include "DummyGpu.hpp"
#include "SimpleBlockMatcher.hpp"
#include "HierarchicalBeliefPropagation.hpp"
#include "ConstantSpaceBeliefPropagation.hpp"
#include "RecordedDisparities.hpp"
#include "SimpleTreesOptimizationStereo.hpp"
#include "SimpleTreesGpuStereo.hpp"

#include "cost_volume/DisparityCostVolumeEstimator.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/get_section_options.hpp"

namespace doppia {

using namespace boost;

using namespace program_options;


options_description
StereoMatcherFactory::get_args_options()
{

    options_description desc("StereoMatcherFactory options");


    desc.add_options()

            ("stereo.method", value<string>()->default_value("census"),
             "matching methods: none, census, simple_sad, simple_ssd, simple_lcdm, simple_census, " \
             "hbp, csbp, " \
             "gpu_sad, gpu_ssd, gpu_census, " \
             "simple_trees, " \
             "or recorded");

    desc.add(AbstractStereoMatcher::get_args_options());
    desc.add(AbstractStereoBlockMatcher::get_args_options());
    desc.add(CensusStereoMatcher::get_args_options());
    desc.add(DummyGpu::get_args_options());
    desc.add(SimpleBlockMatcher::get_args_options());
    desc.add(HierarchicalBeliefPropagation::get_args_options());
    desc.add(ConstantSpaceBeliefPropagation::get_args_options());
    desc.add(RecordedDisparities::get_args_options());
    desc.add(SimpleTreesOptimizationStereo::get_args_options());
    desc.add(SimpleTreesGpuStereo::get_args_options());

    desc.add(DisparityCostVolumeEstimator::get_args_options());

    //  desc.add(get_section_options("stereo", "AbstractStereoMatcher options", AbstractStereoMatcher::get_args_options()));
    // desc.add(get_section_options("stereo", "AbstractStereoBlockMatcher options", AbstractStereoBlockMatcher::get_args_options()));
    // desc.add(get_section_options("stereo", "CensusStereoMatcher options", CensusStereoMatcher::get_args_options()));



    /*
    desc.add(DummyCensus::get_args_options());
    desc.add(DiffuseMatcher::get_args_options());
    desc.add(SumOfAbsoluteDifference::get_args_options());
    desc.add(OpenCvStereo::get_args_options());
    desc.add(NasaVisionWorkBenchStereo::get_args_options());
    desc.add(NasaVisionWorkBenchStereoRaw::get_args_options());
    desc.add(GeodesicStereoMatcher::get_args_options());
    desc.add(DummyGpu::get_args_options());
*/

    return desc;
}


AbstractStereoMatcher*
StereoMatcherFactory::new_instance(const variables_map &options, boost::shared_ptr<const AbstractVideoInput> video_input_p)
{

    // create the stereo matcher instance
    const string method = get_option_value<std::string>(options, "stereo.method");


    AbstractStereoMatcher* stereo_matcher_p = NULL;
    if (method.empty() or (method.compare("census") == 0))
    {
        stereo_matcher_p = new CensusStereoMatcher(options);
    }
    /*
    else if ( method.compare("dummy_census") == 0)
    {
        stereo_matcher_p = new DummyCensus(options);
    }
    else if ( method.compare("geodesic") == 0)
    {
        stereo_matcher_p = new GeodesicStereoMatcher(options);
    }
    else if (method.compare("diffuse") == 0)
    {
        stereo_matcher_p = new DiffuseMatcher(options);
    }
    else if (method.compare("sad") == 0)
    {
        stereo_matcher_p = new SumOfAbsoluteDifference(options);
    }*/
    else if (method.compare("simple_sad") == 0 or
             method.compare("simple_ssd") == 0 or
             method.compare("simple_lcdm") == 0 or
             method.compare("simple_census") == 0 )
    {
        stereo_matcher_p = new SimpleBlockMatcher(options);
    }
    else if (method.compare("gpu_sad") == 0 or method.compare("gpu_ssd") == 0 or method.compare("gpu_census") == 0 )
    {
        stereo_matcher_p = new DummyGpu(options);
    }
    else if ((method.compare("gpu_simple_trees") == 0) or (method.compare("gpu_trees") == 0))
    {
        stereo_matcher_p = new SimpleTreesGpuStereo(options);
    }
    /*
    else if (method.compare("opencv_sad") == 0 or
             method.compare("opencv_bp") == 0 or
             method.compare("opencv_csbp") == 0 )
    {
        stereo_matcher_p = new OpenCvStereo(options);
    }
    else if (method.compare("nasa") == 0)
    {
        stereo_matcher_p = new NasaVisionWorkBenchStereo(options);
    }
    else if (method.compare("nasa_raw") == 0)
    {
        stereo_matcher_p = new NasaVisionWorkBenchStereoRaw(options);
    }
    else if (method.compare("sgm") == 0)
    {
        throw std::runtime_error("sgm method not implemented in this executable");
    }*/
    else if (method.compare("hbp") == 0)
    {
        stereo_matcher_p = new HierarchicalBeliefPropagation(options);
    }
    else if (method.compare("csbp") == 0)
    {
        stereo_matcher_p = new ConstantSpaceBeliefPropagation(options);
    }
    else if (method.compare("recorded") == 0)
    {
        stereo_matcher_p = new RecordedDisparities(options, video_input_p);
    }
    else if ((method.compare("simple_trees") == 0) or (method.compare("trees") == 0))
    {
        stereo_matcher_p = new SimpleTreesOptimizationStereo(options);
    }
    else if (method.compare("none") == 0)
    {
        stereo_matcher_p = NULL;
    }
    else
    {
        printf("StereoMatcherFactory received stereo.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'stereo.method' value");
    }


    return stereo_matcher_p;

} // end of StereoMatcherFactory::new_instance




} // end of namespace doppia


