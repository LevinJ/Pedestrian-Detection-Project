/// Cuda code that provides speed critical functions related to GpuIntegralChannelsForPedestrians.cpp
/// part of the code here originally based on OpenCv 2.3 cv::gpu::hog::compute_gradients_8UC1

#include "integral_channels.cu.hpp"

#include "helpers/gpu/cuda_safe_call.hpp"
#include "helpers/gpu/TextureBinder.hpp"
#include "helpers/gpu/device_memory_linear_1d_no_free.hpp"

#include "cudatemplates/copy.hpp"
#include "cudatemplates/devicememorylinear.hpp"
#include "cudatemplates/devicememoryreference.hpp"
#include "cudatemplates/hostmemoryheap.hpp"
#include "cudatemplates/stream.hpp"


#include <algorithm>
#include <cmath>
//#include "math_functions.h"
#include <cfloat>

namespace {


template<class T> static inline void upload_constant(const char* name, const T& value)
{
    cuda_safe_call( cudaMemcpyToSymbol(name, &value, sizeof(T)) );
    return;
}

template<class T> static inline void upload_constant(const char* name, const T& value, cudaStream_t stream)
{
    cuda_safe_call( cudaMemcpyToSymbolAsync(name, &value, sizeof(T), 0, cudaMemcpyHostToDevice, stream) );
    return;
}


/// small helper function that computes "static_cast<int>(ceil(static_cast<float>(total)/grain))", but faster
static inline int div_up(const int total, const int grain)
{
    return (total + grain - 1) / grain;
}

/// Helper method to handle OpenCv's GPU data
template <typename DstType, typename SrcDevMem2dType>
Cuda::DeviceMemoryReference2D<DstType> devmem2d_to_device_reference_2d(
        //const cv::gpu::DevMem2D_<SrcType> &src // this does not work with cv::gpu::DevMem2D
        SrcDevMem2dType &src)
{
    Cuda::Layout<DstType, 2> layout(Cuda::Size<2>(src.cols, src.rows));
    layout.setPitch(src.step); // step is in bytes
    return Cuda::DeviceMemoryReference2D<DstType>(layout, reinterpret_cast<DstType *>(src.data));
}


const int num_threads = 256;
//const int num_threads = 320; // we optimize for images of width 640 pixel

dim3 get_block_dimensions(const int width, const int height)
{
    // FIXME for some strange reason block 16x16 provides 'bogus' outputs
    //dim3 block_dimensions(16, 16);
    dim3 block_dimensions(num_threads, 1); // required when using smart hog kernel

    return block_dimensions;
}


} // end of anonymous namespace

namespace doppia {

namespace integral_channels {

using namespace boost;
using namespace cv::gpu;
using boost::uint8_t;

/// How many angle bins in the HOG channels ?
/// (will also define where to store the LUV channels)
const int num_angles_bin = 6;

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
/// GPU version of AngleBinComputer
/// @see AngleBinComputer
typedef DeviceMemoryLinear1DNoFree<float2> angle_bins_vectors_t;
angle_bins_vectors_t angle_bins_vectors;
__constant__ float2 const_angle_bins_vectors[num_angles_bin];

void lazy_init_angle_bins_vectors()
{
    if(angle_bins_vectors.getBuffer() != NULL)
    {
        // already initialized
        return;
    }
    else
    { // first call
        angle_bins_vectors.alloc(num_angles_bin);
        Cuda::HostMemoryHeap1D<float2> cpu_angle_bins_vectors(static_cast<size_t>(num_angles_bin));
        const float angle_quantum = M_PI/num_angles_bin;

        float theta = 0; // theta is in the center of the angle bin
        for(int bin_index = 0; bin_index < num_angles_bin; bin_index += 1, theta+= angle_quantum)
        {
            cpu_angle_bins_vectors[bin_index].x = std::cos(theta);
            cpu_angle_bins_vectors[bin_index].y = std::sin(theta);
        }

        Cuda::Symbol<float2, 1> const_angle_bins_vectors_symbol(Cuda::Size<1>(num_angles_bin), const_angle_bins_vectors);
        Cuda::copy(angle_bins_vectors, cpu_angle_bins_vectors);
        Cuda::copy(const_angle_bins_vectors_symbol, cpu_angle_bins_vectors);
    }

    return;
}


/// calling convention is the same of atan2
__device__ __forceinline__
int fast_angle_bin(const float &y, const float &x)
{
    // no need to explicitly handle the case y and x == 0,
    // this correspond to zero gradient areas, thus whatever the bin, the histograms will not be modified

    // (all benchmarks for num_angle_bins == 6)
    // using atan2 code runs at ~650 [Hz]
    // using tex1Dfetch code runs at ~645 [Hz] (code removed, see revision 1468:3542402a30e8 )
    // using constant memory code runs at ~651 [Hz]
    // using global memory code runs at ?? (not tested) [Hz]

    int index = 0;
    const float2 &bin_vector_zero = const_angle_bins_vectors[0];
    //const float2 bin_vector_zero = const_angle_bins_vectors[0];
    //const float2 &bin_vector_zero = angle_bins_vectors[0];
    float max_dot_product = fabs(x*bin_vector_zero.x + y*bin_vector_zero.y);

    // let us hope this gets unrolled
#pragma unroll
    for(int i=1; i < num_angles_bin; i+=1)
    {
        const float2 &bin_vector_i = const_angle_bins_vectors[i];
        //const float2 bin_vector_i = const_angle_bins_vectors[i];
        //const float2 &bin_vector_i = angle_bins_vectors[i];
        const float dot_product = fabs(x*bin_vector_i.x + y*bin_vector_i.y);
        if(dot_product > max_dot_product)
        {
            max_dot_product = dot_product;
            index = i;
        }
    } // end of "for each bin"
    return index;
}


__device__ __forceinline__
int atan2_angle_bin(const float &y, const float &x)
{

    const float angle_quantum = M_PI/num_angles_bin;
    const float angle_scaling = 1/angle_quantum; // from angle to angle_channel_index

    float angle = atan2(y, x) + (angle_quantum/2);
    if(angle < 0)
    {
        angle += M_PI; // reflect negative angles
    }

    //assert(angle >= 0);
    const int angle_channel_index = static_cast<int>(angle*angle_scaling) % num_angles_bin;

    return angle_channel_index;
}



// end of Gpu version of AngleBinComputer
// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

typedef Cuda::DeviceMemoryPitched2D<uint8_t>::Texture input_gray_texture_t;
input_gray_texture_t input_gray_texture;


typedef Cuda::DeviceMemoryPitched2D<uchar4>::Texture input_rgb_texture_t;
input_rgb_texture_t input_rgb_texture;


/// shared piece of between
/// code between compute_hog_channels_kernel_implementation and
/// compute_rgb_hog_channels_kernel_implementation
template <int num_threads, int num_angles_bin>
__device__ __forceinline__
void set_hog_channels_kernel(
        const int x, const int y,
        gpu_channels_t::KernelData &feature_channels,
        const float dx, const float dy)
{

    //const float max_magnitude = sqrt(2)*(255*2); // works with GaussianFilter and pre_smoothing_filter
    const float max_magnitude = sqrtf(2)*255; // used with a - b
    const float magnitude_scaling = 255.0f/max_magnitude;
    // the compiler is smart enough to define the value of magnitude_ and angle_scaling at compile time

    const float magnitude = sqrtf((dx * dx) + (dy * dy)) * magnitude_scaling;
    const uint8_t magnitude_u8 = static_cast<uint8_t>(magnitude);

    int angle_channel_index;

    // using atan2 the GPU integral channel computation runs at ~650 [Hz]
    // using fast_angle_bin it runs at ~645 [Hz] (for now..)
    // (3.74 [Hz] objects detection with fast_angle_bin, versus 3.77 [Hz])
    const bool use_atan2 = false;
    if(use_atan2)
    {
        angle_channel_index = atan2_angle_bin(dy, dx);
    }
    else
    {
        angle_channel_index = fast_angle_bin(dy, dx);
    }

    const size_t
            &channel_stride = feature_channels.stride[1],
            &row_stride = feature_channels.stride[0];

    // set the magnitude value on the magnitude channel and on the angle specific channel
    const int
            pixel_offset = x + (y * row_stride),
            i = pixel_offset + (num_angles_bin*channel_stride),
            j = pixel_offset + (angle_channel_index*channel_stride);
    feature_channels.data[i] = magnitude_u8;
    feature_channels.data[j] = magnitude_u8;

    return;
}




/// Using simple code via texture access is faster than using
/// smart code via global memory (e.g. like OpenCv's HOG code)
/// @see revision 1461:391e877b96c0
/// input gray image is read via input_gray_texture
/// we assume that x,y are inside the image
template <int num_threads, int num_angles_bin>
__device__ __forceinline__
void compute_hog_channels_kernel_implementation(
        const int &x, const int &y,
        gpu_channels_t::KernelData &feature_channels)
{

    const float
            dx_a = tex2D(input_gray_texture, x + 1, y),
            dx_b = tex2D(input_gray_texture, x - 1, y),
            dx = dx_a - dx_b,
            dy_a = tex2D(input_gray_texture, x, y + 1),
            dy_b = tex2D(input_gray_texture, x, y - 1),
            dy = dy_a - dy_b;

    set_hog_channels_kernel<num_threads, num_angles_bin>(x, y, feature_channels, dx, dy);
    return;
}

/// This is the RGB version of compute_hog_channels_kernel_implementation
/// @see doppia::compute_color_derivatives(...)
template <int num_threads, int num_angles_bin>
__device__ __forceinline__
void compute_rgb_hog_channels_kernel_implementation(
        const int &x, const int &y,
        gpu_channels_t::KernelData &feature_channels)
{

    const uchar4
            x_a = tex2D(input_rgb_texture, x + 1, y),
            x_b = tex2D(input_rgb_texture, x - 1, y),
            y_a = tex2D(input_rgb_texture, x, y + 1),
            y_b = tex2D(input_rgb_texture, x, y - 1);

    const float
            dx_r = static_cast<float>(x_a.x) - x_b.x,
            dx_g = static_cast<float>(x_a.y) - x_b.y,
            dx_b = static_cast<float>(x_a.z) - x_b.z,
            dx_rg = fmaxf(dx_r, dx_g),
            dx_rgb = fmaxf(dx_rg, dx_b),
            dy_r = static_cast<float>(y_a.x) - y_b.x,
            dy_g = static_cast<float>(y_a.y) - y_b.y,
            dy_b = static_cast<float>(y_a.z) - y_b.z,
            dy_rg = fmaxf(dy_r, dy_g),
            dy_rgb = fmaxf(dy_rg, dy_b);

    set_hog_channels_kernel<num_threads, num_angles_bin>(x, y, feature_channels, dx_rgb, dy_rgb);
    return;
}



/// inspired by opencv's compute_gradients_8UC1_kernel
template <int num_threads, int num_angles_bin>
__global__
//__device__
void compute_hog_channels_kernel(
        const int image_width, const int image_height,
        //const PtrElemStep input_gray_image,
        gpu_channels_t::KernelData feature_channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < image_width) and (y < image_height))
    {
        compute_hog_channels_kernel_implementation<num_threads, num_angles_bin>
        //compute_rgb_hog_channels_kernel_implementation<num_threads, num_angles_bin>
                (x, y, feature_channels);
    } // end of "if x,y inside the image"


    return;
}



void compute_hog_channels(const cv::gpu::DevMem2D& input_gray_image, gpu_channels_t &feature_channels)
{

    assert(feature_channels.getLayout().size[2] >= static_cast<size_t>(num_angles_bin));

    lazy_init_angle_bins_vectors();

    // bind the input image
    const Cuda::DeviceMemoryReference2D<uint8_t>
            input_gray_reference = devmem2d_to_device_reference_2d<uint8_t>(input_gray_image);
    input_gray_texture.filterMode = cudaFilterModePoint; // normal access to the values
    input_gray_reference.bindTexture(input_gray_texture);

    const int width = input_gray_image.cols, height = input_gray_image.rows;
    const dim3 block_dimensions = get_block_dimensions(width, height);
    const dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));

    compute_hog_channels_kernel
            <num_threads, num_angles_bin>
            <<<grid_dimensions, block_dimensions>>>
                                                  (width, height, feature_channels);

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    input_gray_reference.unbindTexture(input_gray_texture);

    return;
}


/// cube root approximation using bit hack for 32-bit float
/// provides a very crude approximation
/// let us hope this hack holds on GPU
__device__ __forceinline__
float cbrt_5_f32(float f)
{
    unsigned int* p = reinterpret_cast<unsigned int *>(&f);
    *p = *p/3 + 709921077;
    return f;
}

/// iterative cube root approximation using Halley's method (float)
__device__ __forceinline__
float cbrta_halley_f32(const float a, const float R)
{
    const float a3 = a*a*a;
    const float b = a * (a3 + R + R) / (a3 + a3 + R);
    return b;
}

/// Code based on
/// http://metamerist.com/cbrt/cbrt.htm
/// cube root approximation using 2 iterations of Halley's method (float)
/// this is expected to be ~2.5x times faster than std::pow(x, 3)
__device__ __forceinline__
float fast_cube_root(const float d)
{
    float a = cbrt_5_f32(d);
    a = cbrta_halley_f32(a, d);
    return cbrta_halley_f32(a, d);
}

//const size_t cube_root_table_size = 512;
const size_t cube_root_table_size = 2048;

//typedef Cuda::DeviceMemoryLinear1D<float> cube_root_table_t;
typedef DeviceMemoryLinear1DNoFree<float>  cube_root_table_t;
cube_root_table_t cube_root_table;
bool cube_root_table_is_initialized = false;


typedef cube_root_table_t::Texture cube_root_table_texture_t;
cube_root_table_texture_t cube_root_table_texture;
__constant__ float const_cube_root_table[cube_root_table_size];


/// this code is based on CPU's fast_rgb_to_luv
/// cube_root_table is accessed via cube_root_table_texture
/// (seems slightly faster than global and constant memory)
/// (@see revision 1461:391e877b96c0 for old code)
__device__ __forceinline__
uchar3 rgb_to_luv(const uchar4 &rgba_u8)
{
    uchar3 luv_u8;

    const float
            r = rgba_u8.x / 255.0f,
            g = rgba_u8.y / 255.0f,
            b = rgba_u8.z / 255.0f,
            x = 0.412453f*r + 0.35758f*g + 0.180423f*b,
            y = 0.212671f*r + 0.71516f*g + 0.072169f*b,
            z = 0.019334f*r + 0.119193f*g + 0.950227f*b;

    const float
            x_n = 0.312713f, y_n = 0.329016f,
            uv_n_divisor = -2.f*x_n + 12.f*y_n + 3.f,
            u_n = 4.f*x_n / uv_n_divisor,
            v_n = 9.f*y_n / uv_n_divisor;

    const float
            uv_divisor = fmax((x + 15.f*y + 3.f*z), FLT_EPSILON),
            u = 4.f*x / uv_divisor,
            v = 9.f*y / uv_divisor;

    // on GPU (Europa laptop),
    // fast math runs at ~240 [Hz],
    // fast cube root at ~280 [Hz],
    // constant memory at ~320 [Hz],
    // global memory at ~330 [Hz],
    // texture fetch at ~330 [Hz] (but seems slightly faster than global memory on average).

    //const float y_cube_root = powf(y, 1.0f/3.0f); // canonical implementation.
    //const float y_cube_root = __powf(y, 1.0f/3.0f); // canonical fast math implementation.
    //const float y_cube_root = fast_cube_root(y); // ~90 [Hz] on test_objects_detection
    const int max_i = cube_root_table_size - 1;
    //const int i = min(static_cast<int>(y*max_i), max_i); // safe
    //const int i = static_cast<int>(y*max_i); // less safe, but should be ok
    const int i = __float2int_rn(y*max_i); // less safe, but should be ok

    //const float y_cube_root = cube_root_table.data[i]; // global memory
    //const float y_cube_root = const_cube_root_table[i]; // constant memory
    // tex1Dfetch should be used to access linear memory (not text1D)
    // http://forums.nvidia.com/index.php?showtopic=164023
    // texture access seem to be slightly faster than memory access
    const float y_cube_root = tex1Dfetch(cube_root_table_texture, i);

    const float
            l_value = fmax(0.f, ((116.f * y_cube_root) - 16.f)),
            u_value = 13.f * l_value * (u - u_n),
            v_value = 13.f * l_value * (v - v_n);

    // L in [0, 100], U in [-134, 220], V in [-140, 122]
    const float
            scaled_l = l_value * (255.f / 100.f),
            scaled_u = (u_value + 134.f) * (255.f / (220.f + 134.f )),
            scaled_v = (v_value + 140.f) * (255.f / (122.f + 140.f ));

    luv_u8.x = static_cast<boost::uint8_t>(scaled_l);
    luv_u8.y = static_cast<boost::uint8_t>(scaled_u);
    luv_u8.z = static_cast<boost::uint8_t>(scaled_v);

    return luv_u8;
}

/// the input_rgb_image is accessed via input_rgb_texture
__device__ __forceinline__
void compute_pixel_luv_channels(
        const int &x, const int &y,
        gpu_channels_t::KernelData &feature_channels)
{
    // we expect RGBA input

    // u8 indicates 8 bits unsigned representation
    const uchar4 rgba_u8 = tex2D(input_rgb_texture, x, y);

    // compute
    //const uchar3 luv_u8 = {rgba_u8.x, rgba_u8.y, rgba_u8.z}; // just for testing
    const uchar3 luv_u8 = rgb_to_luv(rgba_u8);

    const size_t
            &channel_stride = feature_channels.stride[1],
            &row_stride = feature_channels.stride[0];

    // set the magnitude value on the magnitude channel and on the angle specific channel
    const int
            pixel_offset = x + (blockIdx.y * row_stride),
            l_index = pixel_offset + ((num_angles_bin + 1)*channel_stride),
            u_index = pixel_offset + ((num_angles_bin + 2)*channel_stride),
            v_index = pixel_offset + ((num_angles_bin + 3)*channel_stride);

    feature_channels.data[l_index] = luv_u8.x;
    feature_channels.data[u_index] = luv_u8.y;
    feature_channels.data[v_index] = luv_u8.z;

    return;
}

/// the input_rgb_image is accessed via input_rgb_texture
template <int num_threads, int num_angles_bin>
__global__
void compute_luv_channels_kernel(
        const int image_width, const int image_height,
        gpu_channels_t::KernelData feature_channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < image_width) and (y < image_height))
    {
        compute_pixel_luv_channels(x,y, feature_channels);
    } // end of "if x,y inside the image"

    return;
}

void lazy_init_cube_root_table()
{

    if(cube_root_table_is_initialized)
    {
        // already initialized
        return;
    }
    else
    { // first call

        cube_root_table.alloc(cube_root_table_size);

        Cuda::HostMemoryHeap1D<float> lookup_table(cube_root_table_size);
        const int max_i = cube_root_table_size - 1;
        for(int i=0; i < static_cast<int>(cube_root_table_size); i+=1)
        {
            const float x = static_cast<float>(i) / max_i;
            lookup_table[i] = pow(x, 1.0f / 3.0f);
        }

        Cuda::Symbol<float, 1> const_cube_root_table_symbol(Cuda::Size<1>(cube_root_table_size), const_cube_root_table);
        Cuda::copy(cube_root_table, lookup_table);
        Cuda::copy(const_cube_root_table_symbol, lookup_table);

        cube_root_table_is_initialized = true;
    }

    return;
}


void compute_luv_channels(const cv::gpu::DevMem2D& input_rgb_image, gpu_channels_t &feature_channels)
{
    assert(feature_channels.getLayout().size[2] >= static_cast<size_t>(num_angles_bin + 3));
    assert(input_rgb_image.step >= static_cast<size_t>(4*input_rgb_image.cols)); // check that data is in RGBA format

    lazy_init_cube_root_table();
    cube_root_table.bindTexture(cube_root_table_texture);

    const Cuda::DeviceMemoryReference2D<uchar4>
            input_rgb_reference = devmem2d_to_device_reference_2d<uchar4>(input_rgb_image);
    input_rgb_texture.filterMode = cudaFilterModePoint; // normal access to the values
    input_rgb_reference.bindTexture(input_rgb_texture);

    const int width = input_rgb_image.cols, height = input_rgb_image.rows;
    const dim3 block_dimensions = get_block_dimensions(width, height);
    const dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));

    compute_luv_channels_kernel
            <num_threads, num_angles_bin>
            <<<grid_dimensions, block_dimensions>>>
                                                  (width, height,feature_channels);

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    cube_root_table.unbindTexture(cube_root_table_texture);
    input_rgb_reference.unbindTexture(input_rgb_texture);

    return;
}




/// the input gray and rgb images are accessed via
/// input_gray_texture and input_rgb_texture respectivelly
/// they both represent the same image, in different color codes
/// the cube_root_table is also accessed via a texture
template <int num_threads, int num_angles_bin>
__global__
void compute_hog_and_luv_channels_kernel(
        const int image_width, const int image_height,
        gpu_channels_t::KernelData feature_channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < image_width) and (y < image_height))
    {
        compute_hog_channels_kernel_implementation<num_threads, num_angles_bin>
        //compute_rgb_hog_channels_kernel_implementation<num_threads, num_angles_bin>
                (x, y, feature_channels);

        compute_pixel_luv_channels(x, y, feature_channels);
    } // end of "if x,y inside the image"

    return;
}


void compute_hog_and_luv_channels(
        const cv::gpu::DevMem2D& input_gray_image,
        const cv::gpu::DevMem2D& input_rgb_image,
        gpu_channels_t &feature_channels)
{

    assert(feature_channels.getLayout().size[2] >= static_cast<size_t>(num_angles_bin + 3));
    assert(input_rgb_image.step >= static_cast<size_t>(4*input_rgb_image.cols)); // check that data is in RGBA format
    assert(input_gray_image.cols == input_rgb_image.cols);
    assert(input_gray_image.rows == input_rgb_image.rows);

    lazy_init_angle_bins_vectors();

    lazy_init_cube_root_table();
    cube_root_table.bindTexture(cube_root_table_texture);

    const Cuda::DeviceMemoryReference2D<uint8_t>
            input_gray_reference = devmem2d_to_device_reference_2d<uint8_t>(input_gray_image);
    input_gray_texture.filterMode = cudaFilterModePoint; // normal access to the values
    input_gray_reference.bindTexture(input_gray_texture);

    const Cuda::DeviceMemoryReference2D<uchar4>
            input_rgb_reference = devmem2d_to_device_reference_2d<uchar4>(input_rgb_image);
    input_rgb_texture.filterMode = cudaFilterModePoint; // normal access to the values
    input_rgb_reference.bindTexture(input_rgb_texture);


    const int width = input_rgb_image.cols, height = input_rgb_image.rows;
    const dim3 block_dimensions = get_block_dimensions(width, height);
    const dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));


    // After code clean up and full data access via textures,
    // running calling a kernel that does both operations is faster than calling one after another
    // compute_hog_and_luv_channels_kernel results in 2.33 Hz, versus 2.30 Hz for the two separate kernels
    const bool two_side_by_side = false;
    if(two_side_by_side)
    {
        static Cuda::Stream hog_stream, luv_stream;

        compute_hog_channels_kernel
                <num_threads, num_angles_bin>
                <<<grid_dimensions, block_dimensions, 0, hog_stream>>>
                                                                     (width, height,
                                                                      feature_channels);
        compute_luv_channels_kernel
                <num_threads, num_angles_bin>
                <<<grid_dimensions, block_dimensions, 0, luv_stream>>>
                                                                     (width, height,
                                                                      feature_channels);
    }
    else
    {
        compute_hog_and_luv_channels_kernel
                <num_threads, num_angles_bin>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       feature_channels);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() ); // make sure all GPU computations are finished

    cube_root_table.unbindTexture(cube_root_table_texture);
    input_gray_reference.unbindTexture(input_gray_texture);
    input_rgb_reference.unbindTexture(input_rgb_texture);

    return;
}


} // end of namespace integral_channels

} // end of namespace doppia
