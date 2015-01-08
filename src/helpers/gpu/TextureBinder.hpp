#ifndef CV_GPU_TEXTUREBINDER_HPP
#define CV_GPU_TEXTUREBINDER_HPP

#include <cstdlib> // required to get devmem2d to compile

#include <opencv2/core/version.hpp>
#if CV_MINOR_VERSION <= 3
#include <opencv2/gpu/devmem2d.hpp> // opencv 2.3
#else
#include <opencv2/core/devmem2d.hpp> // opencv 2.4
#endif

#include <cuda_runtime_api.h>

#include "cuda_safe_call.hpp"

namespace cv {
namespace gpu {

//using doppia::cuda_safe_call; // it is an evil macro

/// This class is a copy paste from the OpenCv source code
/// (put here since not exposed on OpenCv's public API)
class TextureBinder
{
public:

    TextureBinder() : texture_reference_p(NULL)
    {
        // nothing to do here
        return;
    }
    template <typename T> TextureBinder(const textureReference* tex, const DevMem2D_<T>& img) : texture_reference_p(NULL)
    {
        bind(tex, img);
        return;
    }

    template <typename T> TextureBinder(const char* tex_name, const DevMem2D_<T>& img) : texture_reference_p(NULL)
    {
        bind(tex_name, img);
        return;
    }

    ~TextureBinder()
    {
        unbind();
        return;
    }

    template <typename T> void bind(const textureReference* tex_p, const DevMem2D_<T>& img)
    {
        unbind();

        //cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
        cudaChannelFormatDesc desc;
        desc = cudaCreateChannelDesc<T>();
        cuda_safe_call( cudaBindTexture2D(0, tex_p, img.ptr(), &desc, img.cols, img.rows, img.step) );

        texture_reference_p = tex_p;
        return;
    }


    template <typename T> void bind(const char* tex_name, const DevMem2D_<T>& img)
    {
        const textureReference* tex;
        cuda_safe_call( cudaGetTextureReference(&tex, tex_name) );
        bind(tex, img);
        return;
    }


    void unbind()
    {
        if (texture_reference_p)
        {
            cudaUnbindTexture(texture_reference_p);
            texture_reference_p = 0;
        }
        return;
    }

private:
    const textureReference* texture_reference_p;
};


} // end of namespace gpu
} // end of namespace cv

#endif // CV_GPU_TEXTUREBINDER_HPP
