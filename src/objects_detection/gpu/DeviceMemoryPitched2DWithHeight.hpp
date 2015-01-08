#ifndef DOPPIA_DEVICEMEMORYPITCHED2DWITHHEIGHT_HPP
#define DOPPIA_DEVICEMEMORYPITCHED2DWITHHEIGHT_HPP

#include <cudatemplates/devicememorypitched.hpp>

namespace doppia {

/// Special object created to solve the issue of handling 2d or 3d templates with the same code
/// When using 2d templates we need the height information _everywhere_.
/// This class allows to propagate the data, without having to change the vast majority of classes.
template <class Type>
class DeviceMemoryPitched2DWithHeight: public Cuda::DeviceMemoryPitched2D<Type>
{

public:
    size_t height;

    // default constructor
    inline DeviceMemoryPitched2DWithHeight()
    {
        height = 0;
        return;
    }

    struct KernelConstDataWithHeight: public Cuda::DeviceMemoryPitched2D<Type>::KernelConstData
    {
    public:
        const size_t height;

        KernelConstDataWithHeight(const DeviceMemoryPitched2DWithHeight &mem)
            : Cuda::DeviceMemoryPitched2D<Type>::KernelConstData(mem),
              height(mem.height)
        {
            // nothing to do here
            return;
        }
    };

    typedef KernelConstDataWithHeight KernelConstData;
};

} // end of namespace doppia

#endif // DOPPIA_DEVICEMEMORYPITCHED2DWITHHEIGHT_HPP
