#ifndef CUDA_SAFE_CALL_HPP
#define CUDA_SAFE_CALL_HPP

/// inspired by OpenCv's 2.3 modules/gpu/src/cuda/safe_call.hpp

#include <cuda_runtime_api.h>
//#include <cufft.h>
//#include <nppdefs.h>

#if defined(__GNUC__)
#define cuda_safe_call(expr)  doppia::___cuda_safe_call(expr, __FILE__, __LINE__, __func__)
//#define cufft_safe_call(expr)  doppia::___cufft_safe_call(expr, __FILE__, __LINE__, __func__)
//#define npp_safe_call(expr)  doppia::___npp_safe_call(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) or defined(__MSVC__) */
#define cuda_safe_call(expr)  doppia::___cuda_safe_call(expr, __FILE__, __LINE__)
//#define cufft_safe_call(expr)  doppia::___cufft_safe_call(expr, __FILE__, __LINE__)
//#define npp_safe_call(expr)  doppia::___npp_safe_call(expr, __FILE__, __LINE__)
#endif

namespace doppia
{

void cuda_error(const char *error_string, const char *file, const int line, const char *func = "");
//void nppError(int err, const char *file, const int line, const char *func = "");
//void cufftError(int err, const char *file, const int line, const char *func = "");

static inline void ___cuda_safe_call(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
    {
        cuda_error(cudaGetErrorString(err), file, line, func);
    }

    return;
}

/*
static inline void ___cufft_safe_call(cufftResult_t err, const char *file, const int line, const char *func = "")
{
    if (CUFFT_SUCCESS != err)
    {
        doppia::cufft_error(err, file, line, func);
    }
    return;
}

static inline void ___npp_safe_call(int err, const char *file, const int line, const char *func = "")
{
    if (err < 0)
    {
        doppia::npp_error(err, file, line, func);
    }
    return;
}
*/

} // end of namespace doppia

#endif // CUDA_SAFE_CALL_HPP
