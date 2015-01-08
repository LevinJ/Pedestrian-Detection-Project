/// Cuda code that provides speed critical functions related to GpuIntegralChannelsForPedestrians.cpp
/// part of the code here originally based on OpenCv 2.3 cv::gpu::hog::compute_gradients_8UC1

#include "shrinking.cu.hpp"

#include "helpers/gpu/cuda_safe_call.hpp"

#include "cudatemplates/copy.hpp"

namespace {

/// small helper function that computes "static_cast<int>(ceil(static_cast<float>(total)/grain))", but faster
static inline int div_up(const int total, const int grain)
{
    return (total + grain - 1) / grain;
}


dim3 get_2d_block_dimensions(const int /*width*/, const int /*height*/)
{
    // square block seems slightly faster than line block
    //dim3 block_dimensions(16, 16);
    dim3 block_dimensions(20, 20);

    //const int num_threads = 256;
    //const int num_threads = 320; // we optimize for images of width 640 pixel
    //dim3 block_dimensions(num_threads, 1);

    return block_dimensions;
}


} // end of anonymous namespace

namespace doppia {

namespace integral_channels {

using namespace boost;
using boost::uint8_t;


// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

typedef gpu_channel_ref_t::Texture gpu_channel_ref_texture_t;
gpu_channel_ref_texture_t input_channel_texture;


/// the input_channel is accessed via input_channel_texture
/// @param x,y are in the shrunk_channel coordinates
/// @param shrunk_channel should be 1/shrinking_factor the size of the input_channel
template <int shrinking_factor>
__device__ __forceinline__
void compute_shrinked_pixel(
        const int x, const int y,
        const gpu_channel_ref_t::KernelConstData &input_channel,
        const gpu_channel_ref_t::KernelData &shrunk_channel)
{
    int sum = 0;
    const int input_x = x*shrinking_factor, input_y = y*shrinking_factor;

    const int
            max_input_x = input_channel.size[0] -1,
            max_input_y = input_channel.size[1] -1;

    // adding #pragma unroll here makes no difference
    for(int row=0; row < shrinking_factor; row+=1)
    {
        const int t_y = min(input_y + row, max_input_y);

        for(int col=0; col < shrinking_factor; col+=1)
        {
            const int t_x = min(input_x + col, max_input_x);
            sum += tex2D(input_channel_texture, t_x, t_y);
        } // end of "for each row"
    } // end of "for each row"


    sum /= (shrinking_factor*shrinking_factor); // rescale back to [0, 255]

    const size_t &row_stride = shrunk_channel.stride[0];
    const int shrunk_pixel_index = x + y * row_stride;
    shrunk_channel.data[shrunk_pixel_index] = static_cast<gpu_channel_ref_t::Type>(sum);
    return;
}


/// the input_channel is accessed via input_channel_texture
template <int shrinking_factor>
__global__
void shrink_channel_kernel(
        const gpu_channel_ref_t::KernelConstData input_channel,
        const gpu_channel_ref_t::KernelData shrunk_channel)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;


    if ((x < shrunk_channel.size[0]) and (y < shrunk_channel.size[1]))
    {
        compute_shrinked_pixel<shrinking_factor>(x, y, input_channel, shrunk_channel);
    } // end of "if x,y inside the image"

    return;
}


void shrink_channel(const gpu_channel_ref_t &input_channel, gpu_channel_ref_t &shrunk_channel, const int shrinking_factor)
{

    const int
            input_width = input_channel.size[0],
            input_height = input_channel.size[1],
            shrunk_width = shrunk_channel.size[0],
            shrunk_height = shrunk_channel.size[1];

    int expected_shrunk_width = input_width, expected_shrunk_height = input_height;
    if(shrinking_factor == 4)
    {
        expected_shrunk_width = (((input_width + 1)/2) + 1)/2;
        expected_shrunk_height = (((input_height + 1)/2) + 1)/2;
    }
    else if(shrinking_factor == 2)
    {
        expected_shrunk_width = (input_width + 1)/2;
        expected_shrunk_height = (input_height + 1)/2;
    }
    else if(shrinking_factor == 1)
    {
        expected_shrunk_width = input_width;
        expected_shrunk_height = input_height;
    }
    else
    {
        printf("shrinking_factor == %i\n", shrinking_factor);
        throw std::invalid_argument("shrink_channel called with an unsupported shrinking_factor value");
    }

    if((expected_shrunk_width != shrunk_width) or  (expected_shrunk_height != shrunk_height))
    {
        printf("Expected shrunk width, height == (%i, %i)\n", expected_shrunk_width, expected_shrunk_height);
        printf("Actual shrunk width, height == (%i, %i)\n", shrunk_width, shrunk_height);
        throw std::invalid_argument("shrunk_channel size does not match expected size");
    }

    const dim3 block_dimensions = get_2d_block_dimensions(shrunk_width, shrunk_height);
    const dim3 grid_dimensions(div_up(shrunk_width, block_dimensions.x), div_up(shrunk_height, block_dimensions.y));

    input_channel.bindTexture(input_channel_texture);

    if(shrinking_factor == 4)
    {
        shrink_channel_kernel<4>
                <<<grid_dimensions, block_dimensions>>>
                                                      (input_channel, shrunk_channel);
    }
    else if (shrinking_factor == 2)
    {
        shrink_channel_kernel<2>
                <<<grid_dimensions, block_dimensions>>>
                                                      (input_channel, shrunk_channel);
    }
    else if(shrinking_factor == 1)
    {
        Cuda::copy(shrunk_channel, input_channel);
    }
    else
    {
        printf("shrinking_factor == %i\n", shrinking_factor);
        throw std::invalid_argument("shrink_channel called with an unsupported shrinking_factor value");
    }


    input_channel.unbindTexture(input_channel_texture);

    return;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

typedef Cuda::DeviceMemory<gpu_channels_t::Type, 1>::Texture gpu_channels_1d_texture_t;
gpu_channels_1d_texture_t input_channels_texture;


/// helper method to map the device memory to the specific texture reference
/// @see bind_integral_channels_texture inside integral_channels_detector.cu
void bind_input_channels_texture(const gpu_channels_t &input_channels)
{
    //input_channels_texture.filterMode = cudaFilterModeLinear; // linear interpolation of the values
    input_channels_texture.filterMode = cudaFilterModePoint; // normal access to the values

    // cuda does not support binding 3d memory data.
    // We hack this and bind the 3d data, as if it was 1d data,
    // and then have ad-hoc texture access in the kernel
    const cudaChannelFormatDesc texture_channel_description = \
            cudaCreateChannelDesc<gpu_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
            or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc failed");
    }
    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // FIXME add this strategy into cudatemplates
    // ( Note: if you are having an exception being raised on this line,
    // you most probably are using images too big for you GPU )
    CUDA_CHECK(cudaBindTexture(0, input_channels_texture, input_channels.getBuffer(),
                               texture_channel_description, input_channels.getBytes()));

    cuda_safe_call( cudaGetLastError() );
    return;
}


/// input_channels is accessed via input_channels_texture,
/// or via the global memory
/// @param x,y are in the shrunk_channel coordinates
/// @param shrunk_channel should be 1/shrinking_factor the size of the input_channel
template <int shrinking_factor>
__device__ __forceinline__
void compute_shrinked_pixel(
        const int x, const int y, const int channel_index,
        const gpu_channels_t::KernelConstData &input_channels,
        const gpu_channels_t::KernelData &shrunk_channels)
{
    int sum = 0;
    const int input_x = x*shrinking_factor, input_y = y*shrinking_factor;

    const int
            max_input_x = input_channels.size[0] -1,
            max_input_y = input_channels.size[1] -1;

    const size_t
            &input_channel_stride = input_channels.stride[1],
            &input_row_stride = input_channels.stride[0],
            &shrunk_channel_stride = shrunk_channels.stride[1],
            &shrunk_row_stride = shrunk_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const int
            input_channel_offset = channel_index*input_channel_stride,
            shrunk_channel_offset = channel_index*shrunk_channel_stride;

    // adding #pragma unroll here makes no difference
    for(int row=0; row < shrinking_factor; row+=1)
    {
        const int
                t_y = min(input_y + row, max_input_y),
                input_row_offset = t_y*input_row_stride + input_channel_offset;

        for(int col=0; col < shrinking_factor; col+=1)
        {
            const int
                    t_x = min(input_x + col, max_input_x),
                    input_pixel_index = t_x + input_row_offset;

            // tex1Dfetch should be used to access linear memory (not text1D)
            sum += tex1Dfetch(input_channels_texture, input_pixel_index);
            //sum += input_channels.data[input_pixel_index];
            
        } // end of "for each row"
    } // end of "for each row"

    sum /= (shrinking_factor*shrinking_factor); // rescale back to [0, 255]

    const int shrunk_pixel_index = x + y * shrunk_row_stride + shrunk_channel_offset;
    shrunk_channels.data[shrunk_pixel_index] = static_cast<gpu_channels_t::Type>(sum);
    return;
}


/// input_channels is accessed via input_channels_texture,
/// or via the global memory
template <int shrinking_factor>
__global__
void shrink_channels_kernel(
        const gpu_channels_t::KernelConstData input_channels,
        const gpu_channels_t::KernelData shrunk_channels)
{
    const int
            x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            //channel_index = blockIdx.z * blockDim.z + threadIdx.z;
            channel_index = blockIdx.z; // we expect blockDim.z and threadIdx.z == 1

    if ((x < shrunk_channels.size[0]) and (y < shrunk_channels.size[1]))
    {
        compute_shrinked_pixel<shrinking_factor>(x, y, channel_index,
                                                 input_channels, shrunk_channels);
    } // end of "if x,y inside the image"

    return;
}


void shrink_channels(const gpu_channels_t &input_channels, gpu_channels_t &shrunk_channels, const int shrinking_factor)
{

    const int
            input_width = input_channels.size[0],
            input_height = input_channels.size[1],
            shrunk_width = shrunk_channels.size[0],
            shrunk_height = shrunk_channels.size[1];

    int expected_shrunk_width = input_width, expected_shrunk_height = input_height;
    if(shrinking_factor == 4)
    {
        expected_shrunk_width = (((input_width + 1)/2) + 1)/2;
        expected_shrunk_height = (((input_height + 1)/2) + 1)/2;
    }
    else if(shrinking_factor == 2)
    {
        expected_shrunk_width = (input_width + 1)/2;
        expected_shrunk_height = (input_height + 1)/2;
    }
    else if(shrinking_factor == 1)
    {
        expected_shrunk_width = input_width;
        expected_shrunk_height = input_height;
    }
    else
    {
        printf("shrinking_factor == %i\n", shrinking_factor);
        throw std::invalid_argument("shrink_channel called with an unsupported shrinking_factor value");
    }

    if((expected_shrunk_width != shrunk_width) or  (expected_shrunk_height != shrunk_height))
    {
        printf("Expected shrunk width, height == (%i, %i)\n", expected_shrunk_width, expected_shrunk_height);
        printf("Actual shrunk width, height == (%i, %i)\n", shrunk_width, shrunk_height);
        throw std::invalid_argument("shrunk_channel size does not match expected size");
    }

    const int
            num_input_channels = input_channels.size[2],
            num_shrunk_channels = shrunk_channels.size[2];

    if(num_input_channels != num_shrunk_channels)
    {
        throw std::invalid_argument("the number of channels is different between input_channels and shrunk_channels");
    }

    dim3 block_dimensions = get_2d_block_dimensions(shrunk_width, shrunk_height);
    block_dimensions.z = 1;
    const dim3 grid_dimensions(
                div_up(shrunk_width, block_dimensions.x),
                div_up(shrunk_height, block_dimensions.y),
                num_input_channels);

    bind_input_channels_texture(input_channels);

    if(shrinking_factor == 4)
    {
        shrink_channels_kernel<4>
                <<<grid_dimensions, block_dimensions>>>
                                                      (input_channels, shrunk_channels);
    }
    else if (shrinking_factor == 2)
    {
        shrink_channels_kernel<2>
                <<<grid_dimensions, block_dimensions>>>
                                                      (input_channels, shrunk_channels);
    }
    else if(shrinking_factor == 1)
    {
        Cuda::copy(shrunk_channels, input_channels);
    }
    else
    {
        printf("shrinking_factor == %i\n", shrinking_factor);
        throw std::invalid_argument("shrink_channel called with an unsupported shrinking_factor value");
    }

    cuda_safe_call( cudaUnbindTexture(input_channels_texture) );
    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    return;
}


// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


} // end of namespace integral_channels

} // end of namespace doppia
