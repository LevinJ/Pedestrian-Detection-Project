#include "IntegralChannelsComputerFactory.hpp"

#include "IntegralChannelsForPedestrians.hpp"
#if defined(USE_GPU)
#include "GpuIntegralChannelsForPedestrians.hpp"
#include "helpers/gpu/select_gpu_device.hpp"
#endif

#include "AbstractChannelsComputer.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/predicate.hpp>


namespace doppia {

MODULE_LOG_MACRO("IntegralChannelsComputerFactory")

using namespace std;
using namespace boost;
using namespace boost::program_options;

options_description IntegralChannelsComputerFactory::get_options_description()
{

    options_description desc("IntegralChannelsComputerFactory options");

    desc.add_options()

        #if defined(USE_GPU)
            ("gpu.device_id",
             program_options::value<int>()->default_value(0),
             "select which GPU use for detection.")
        #endif

            ;

    //desc.add(AbstractIntegralChannelsComputer::get_options_description());

    //desc.add(ChannelsComputerFactory::get_options_description());
    //desc.add(IntegralChannelsForPedestrians::get_options_description());

#if defined(USE_GPU)
    //desc.add(GpuIntegralChannelsForPedestrians::get_options_description());
#endif

    return desc;
}


AbstractIntegralChannelsComputer *IntegralChannelsComputerFactory::new_instance(const variables_map &options,
                                                                                   const std::string &method)
{
#if defined(USE_GPU)
    const int gpu_device_id = get_option_value<int>(options, "gpu.device_id");
    set_gpu_device(gpu_device_id);
#endif

    std::string channels_method = method;
    if(channels_method.compare("") == 0)
    {
        channels_method = get_option_value<string>(options, "channels.method");
    }
    log.info() << "channels.method == " << channels_method << std::endl;

    bool use_presmoothing = not boost::algorithm::ends_with(channels_method, "_nonsmooth");

if (boost::algorithm::starts_with(channels_method, "hog6_luv")) // and hog6_luv_nonsmooth
    {
#if defined(USE_GPU)
        log.info() << "Will use specialized GPU enabled GpuIntegralChannelsForPedestrians" << std::endl;
        return new GpuIntegralChannelsForPedestrians(options, use_presmoothing);
#else
        log.info() << "Will use specialized CPU only IntegralChannelsForPedestrians" << std::endl;
        return new IntegralChannelsForPedestrians(options, use_presmoothing);
#endif
    }
    else
    {
        log.error() << "Unknown channels.method = " << channels_method << std::endl;
        throw std::runtime_error("IntegralChannelsComputerFactory(): Unrecognised set of channels");
    }

    return NULL;
}


} // end of namespace doppia
