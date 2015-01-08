#include "select_gpu_device.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include "helpers/Log.hpp"
#include "cuda_safe_call.hpp"

namespace
{

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "set_gpu_device");
}

} // end of anonymous namespace


namespace doppia {

void set_gpu_device(const int device_id)
{

    int number_of_devices;
    cuda_safe_call( cudaGetDeviceCount(&number_of_devices) );

    if(device_id >= number_of_devices)
    {
        log_error() << "Requested device id: " << device_id
                    << " but only " << number_of_devices << " exist" << std::endl;
        throw std::runtime_error("Requested a device_id that does not exist");
    }

    cuda_safe_call( cudaSetDevice(device_id) );

    int current_device_id;
    cuda_safe_call( cudaGetDevice(&current_device_id) );

    if(current_device_id != device_id)
    {
        throw std::runtime_error("cudaSetDevice failed to set the device");
    }

    return;
}

} // end of namespace doppia
