#ifndef INTEGRALCHANNELSFORPEDESTRIANS_HPP
#define INTEGRALCHANNELSFORPEDESTRIANS_HPP

#include "AbstractIntegralChannelsComputer.hpp"

#include "AngleBinComputer.hpp"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/cstdint.hpp>
#include <boost/multi_array.hpp>

#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#if not defined(OBJECTS_DETECTION_LIB)
#include <Eigen/Core>
#endif

#define NUM_TEXTON_CHANNELS 8

namespace doppia {

/// Compute the integral channels for pedestrians detection
/// as described in "Integral Channel Features", P. Dollar et al. BMVC 2009
/// we also provide the feature vector at multiple scales using
/// "The Fastest Pedestrian Detector in the West", P. Dollar et al. BMVC 2010
/// see http://vision.ucsd.edu/~pdollar/research/research.html
class IntegralChannelsForPedestrians: public AbstractIntegralChannelsComputer
{
public:

    typedef boost::geometry::model::d2::point_xy<boost::int16_t> point_t;
    typedef boost::geometry::model::box<point_t> rectangle_t;

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8c_view_t input_image_view_t;

    typedef cv::FilterEngine filter_t;
    typedef cv::Ptr<filter_t> filter_shared_pointer_t;

    // FIXME hardcoded parameter
    typedef AngleBinComputer<6> angle_bin_computer_t;

public:

    static boost::program_options::options_description get_options_description();

    IntegralChannelsForPedestrians(const boost::program_options::variables_map &options,
                                   const bool use_presmoothing_ = true);
    IntegralChannelsForPedestrians(const size_t num_hog_angle_bins = 6,
                                   const bool use_presmoothing_ = true);

    ~IntegralChannelsForPedestrians();

    /// dummy operator= to use IntegralChannelsForPedestrians inside a std::vector<>
    IntegralChannelsForPedestrians & operator=(const IntegralChannelsForPedestrians &other);

    size_t get_num_channels() const;

protected:
    void set_input_image(const input_image_view_t &input_image);

public:
    void compute();

    /// @deprecated
    /// Only used in the (also deprecated) IntegralChannelsLinearSvmSlidingWindow
    int get_feature_vector_length() const;

    /// how much we shrink the channel images ?
    static int get_shrinking_factor();

    typedef boost::gil::rgb8c_view_t::point_t input_size_t;
    const input_size_t get_input_size() const;

#if not defined(OBJECTS_DETECTION_LIB)
    typedef Eigen::VectorXf feature_vector_t;

    void get_feature_vector(const rectangle_t &r, feature_vector_t &feature_vector);

    /// get the concatenated raw values of the channels on this rectangle
    void get_channels_values(const rectangle_t &r, feature_vector_t &feature_vector);
#endif

    /// helper function, just for debugging
    void save_channels_to_file();

protected:

    const angle_bin_computer_t angle_bin_computer;
    const int num_hog_angle_bins;
    const bool use_presmoothing;

    /// how much the shrink the channel images ?
    const int shrinking_factor;

    input_image_t input_image;
    input_image_view_t input_image_view;
    input_image_view_t::point_t input_size, channel_size;

public:

    /// channels as computed from the input image, before shrinking
    //typedef boost::multi_array<boost::uint8_t, 3> input_channels_t;
    typedef AbstractChannelsComputer::input_channels_t input_channels_t;
    typedef input_channels_t::reference input_channel_t;

    // uint8_t is enough when resizing factor is 1
    // when using resizing factor 4, uint16_t allows to avoid additional quantization
    // (in practice, we always use resizing factor 4)
    //typedef boost::multi_array<boost::uint8_t, 3> channels_t;
    typedef boost::multi_array<boost::uint16_t, 3> channels_t;
    typedef channels_t::reference channel_t;

    // uint32 will support images up to size 4x4x2000x2000 (x255)
    typedef boost::multi_array<boost::uint32_t, 3> integral_channels_t;
    typedef integral_channels_t::reference integral_channel_t;
    typedef integral_channels_t::const_reference const_integral_channel_t;

    typedef boost::multi_array<boost::uint32_t, 3>::array_view<3>::type integral_channels_view_t;
    typedef boost::multi_array<boost::uint32_t, 3>::const_array_view<3>::type integral_channels_const_view_t;

protected:

    /// helper temporary matrices, used to avoid multiple allocations
    cv::Mat smoothed_input_mat, luv_mat;

    /// the channels as computed from the input image, before shrinking
    input_channels_t input_channels;

    /// the channels used for classification, after shrinking
    channels_t channels;

    /// the integral images of the shrunk channels
    integral_channels_t integral_channels;

    filter_shared_pointer_t pre_smoothing_filter_p;

    /// original baseline implementation (using opencv)
    void compute_v0();
    void compute_hog_channels_v0();
    void compute_luv_channels_v0();
    void resize_channels_v0();
    void resize_channel_v0(const input_channel_t input_channel, channel_t channel);
    void integrate_channels_v0();

    /// faster version (following P. Dollar approximation and better parallelization)
    void compute_v1();
    void compute_hog_channels_v1();
    void resize_channels_v1();
    void resize_channel_v1(const input_channel_t input_channel, channel_t channel);

public:
    /// helper function for low level operations
    const channels_t &get_channels() const;
    const integral_channels_t &get_integral_channels() const;

    const input_channels_t &get_input_channels_uint8();
    const channels_t &get_input_channels_uint16();
    const integral_channels_t &get_integral_channels();

protected:
    friend class DetectorsComparisonTestApplication;

};

/// helper method shared with GpuIntegralChannelsForPedestrians
std::vector<float> get_binomial_kernel_1d(const int binomial_filter_radius);

#if not defined(OBJECTS_DETECTION_LIB)
/// helper method to debuc the integral images content
void get_channel_matrix(const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
                        const size_t channel_index,
                        Eigen::MatrixXf &channel_matrix);
#endif

/// helper method to visualize the integral images content
void save_integral_channels_to_file(const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
                                    const std::string file_path);

} // end of namespace doppia

#endif // INTEGRALCHANNELSFORPEDESTRIANS_HPP
