
#include <opencv2/core/version.hpp>
#if CV_MINOR_VERSION <= 3
#include <opencv2/gpu/devmem2d.hpp> // opencv 2.3
#else
#include <opencv2/core/devmem2d.hpp> // opencv 2.4
#endif


#include <cudatemplates/devicememorypitched.hpp>
#include <boost/cstdint.hpp>

namespace doppia {
namespace integral_channels {

//typedef Cuda::DeviceMemoryPitched3D<boost::uint16_t> gpu_channels_t;
typedef Cuda::DeviceMemoryPitched3D<boost::uint8_t> gpu_channels_t;

void setup_constants(const int num_angle_bins);

void compute_gradients_8UC1(int nbins, int height, int width, const cv::gpu::DevMem2D& img, 
                            float angle_scale,
                            cv::gpu::DevMem2Df &grad, cv::gpu::DevMem2D &qangle);


void compute_hog_channels(const cv::gpu::DevMem2D& input_image, gpu_channels_t &feature_channels);

void compute_luv_channels(const cv::gpu::DevMem2D& input_image, gpu_channels_t &feature_channels);

void compute_hog_and_luv_channels(const cv::gpu::DevMem2D& input_gray_image,
                                  const cv::gpu::DevMem2D& input_rgb_image,
                                  gpu_channels_t &feature_channels);

} // end of namespace integral_channels
} // end of namespace doppia


