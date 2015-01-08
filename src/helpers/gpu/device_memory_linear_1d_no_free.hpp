#ifndef DOPPIA_DEVICE_MEMORY_LINEAR_1D_NO_FREE_HPP
#define DOPPIA_DEVICE_MEMORY_LINEAR_1D_NO_FREE_HPP

#include "cudatemplates/devicememorylinear.hpp"

namespace doppia {

/// This is a dangerous class that will not free the allocated GPU memory
/// Should only be used for global variables.
/// For global variables CUDA will free the global data before the object destructor is called.
/// When calling the destructor over already free data, an "unload of CUDA runtime failed" error is generated.
/// Using DeviceMemoryLinear1DNoFree avoid having this double allocation. Use with extreme care.
template <class Type>
class DeviceMemoryLinear1DNoFree: public Cuda::DeviceMemoryLinear1D<Type>
{
public:

    // Default constructor
    inline DeviceMemoryLinear1DNoFree()
    {

    }

    inline ~DeviceMemoryLinear1DNoFree()
    {
        // by setting the buffer to NULL, this disable any deallocation
        this->buffer = NULL;
        return;
    }

};

} // end of namespace doppia

#endif // DOPPIA_DEVICE_MEMORY_LINEAR_1D_NO_FREE_HPP
