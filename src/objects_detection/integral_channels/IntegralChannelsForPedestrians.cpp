#include "IntegralChannelsForPedestrians.hpp"

#include <boost/gil/algorithm.hpp>

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <boost/geometry/arithmetic/arithmetic.hpp>
//#include <boost/geometry/algorithms/equals.hpp>

#include "image_processing/integrate.hpp"
#include "image_processing/fast_rgb_to_luv.hpp"

#include "drawing/gil/draw_matrix.hpp"

#include "helpers/ModuleLog.hpp"
#include "helpers/fill_multi_array.hpp"
#include "helpers/get_option_value.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <stdexcept>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "IntegralChannelsForPedestrians");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "IntegralChannelsForPedestrians");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "IntegralChannelsForPedestrians");
}

} // end of anonymous namespace

namespace std
{
ostream& operator<<(ostream &s, const doppia::IntegralChannelsForPedestrians::rectangle_t &r)
{
    s << "( "
      << r.min_corner().x() << ", " << r.min_corner().y() << ", "
      << r.max_corner().x() << ", " << r.max_corner().y() << ")";

    return s;
}

} // end of namespace std


namespace doppia {

using logging::log;
using namespace std;
using namespace boost;

typedef IntegralChannelsForPedestrians::point_t point_t;
typedef IntegralChannelsForPedestrians::rectangle_t rectangle_t;
typedef IntegralChannelsForPedestrians::filter_shared_pointer_t filter_shared_pointer_t;

/// coefficients values come from
/// www.doc.ic.ac.uk/~wl/papers/bf95.pdf
/// www.cse.yorku.ca/~kosta/CompVis_Notes/binomial_filters.pdf
/// http://en.wikipedia.org/wiki/Pascal%27s_triangle
std::vector<float> get_binomial_kernel_1d(const int binomial_filter_radius)
{

    const int radius = binomial_filter_radius;

    std::vector<float> coefficients;

    if(radius == 0)
    {
        coefficients.push_back(1.0f);
    }
    else if(radius == 1)
    {
        // 1 2 1
        const int sum = 4;
        coefficients.push_back(1.0f/sum);
        coefficients.push_back(2.0f/sum);
        coefficients.push_back(1.0f/sum);
    }
    else if(radius == 2)
    {
        // 1 4 6 4 1
        const int sum = 16;
        coefficients.push_back(1.0f/sum);
        coefficients.push_back(4.0f/sum);
        coefficients.push_back(6.0f/sum);
        coefficients.push_back(4.0f/sum);
        coefficients.push_back(1.0f/sum);
    }
    else
    {
        throw std::runtime_error("get_binomial_kernel_1d only accepts radius values 0, 1 and 2");
    }

    assert(coefficients.empty() == false);
    return coefficients;
}


filter_shared_pointer_t
create_pre_smoothing_filter()
{
    const int binomial_filter_degree = 1;
    //const int binomial_filter_degree = 2;

    const bool copy_data = true;
    const cv::Mat binomial_kernel_1d = cv::Mat(get_binomial_kernel_1d(binomial_filter_degree), copy_data);
    filter_shared_pointer_t pre_smoothing_filter_p =
            cv::createSeparableLinearFilter(CV_8UC3, CV_8UC3, binomial_kernel_1d, binomial_kernel_1d);

    return pre_smoothing_filter_p;
}


boost::program_options::options_description IntegralChannelsForPedestrians::get_options_description()
{
    using namespace boost::program_options;
    options_description desc("IntegralChannelsForPedestrians options");

    desc.add_options()

            ("channels.num_hog_angle_bins", value<int>()->default_value(6),
             "Number of angle bins used when computing the HOG channels.\n"
             "Currently supported values are 6 or 18."
             )

            ;

    return desc;
}


IntegralChannelsForPedestrians::IntegralChannelsForPedestrians(const program_options::variables_map &options,
                                                               const bool use_presmoothing_)
    : num_hog_angle_bins(get_option_value<int>(options, "channels.num_hog_angle_bins")),
      use_presmoothing(use_presmoothing_),
      shrinking_factor(get_shrinking_factor())
{
    if (num_hog_angle_bins != angle_bin_computer.get_num_bins())
    {
        log_error() << "Received channels.num_hog_angle_bins == " << num_hog_angle_bins
                    << " but only value " <<  angle_bin_computer.get_num_bins() << " is currently supported"
                       << std::endl;
        throw std::invalid_argument("Requested a value channels.num_hog_angle_bins "
                                    "not currently supported by IntegralChannelsForPedestrians");
    }

    pre_smoothing_filter_p = create_pre_smoothing_filter();
    return;
}


IntegralChannelsForPedestrians::IntegralChannelsForPedestrians(const size_t num_hog_angle_bins_, const bool use_presmoothing_)
    : num_hog_angle_bins(num_hog_angle_bins_),
      use_presmoothing(use_presmoothing_),
      shrinking_factor(get_shrinking_factor())
{
    if (num_hog_angle_bins != angle_bin_computer.get_num_bins())
    {
        log_error() << "Received channels.num_hog_angle_bins == " << num_hog_angle_bins
                    << " but only value " <<  angle_bin_computer.get_num_bins() << " is currently supported"
                       << std::endl;
        throw std::invalid_argument("Requested a value channels.num_hog_angle_bins "
                                    "not currently supported by IntegralChannelsForPedestrians");
    }

    pre_smoothing_filter_p = create_pre_smoothing_filter();
    return;
}


IntegralChannelsForPedestrians::~IntegralChannelsForPedestrians()
{
    // nothing to do here
    return;
}


IntegralChannelsForPedestrians & IntegralChannelsForPedestrians::operator=(const IntegralChannelsForPedestrians &/*other*/)
{
    // dummy implementation, we do nothing here
    return *this;
}


size_t IntegralChannelsForPedestrians::get_num_channels() const
{
    // HOG bins + 1 gradient magnitude + LU
    return num_hog_angle_bins + 4;
}


void IntegralChannelsForPedestrians::set_input_image(const boost::gil::rgb8c_view_t &the_input_view)
{
    // 6 gradients orientations, 1 gradient intensity, 3 LUV color channels
    const int num_channels = 10;

    input_size = the_input_view.dimensions();

    //channel_size = input_image.dimensions() / shrinking_factor;
    // +shrinking_factor/2 to round-up
    if(shrinking_factor == 4)
    {
        channel_size.x = (( (input_size.x+1) / 2) + 1) / 2;
        channel_size.y = (( (input_size.y+1) / 2) + 1) / 2;
    }
    else if(shrinking_factor == 2)
    {
        channel_size.x = (input_size.x+1) / 2;
        channel_size.y = (input_size.y+1) / 2;
    }
    else
    {
        channel_size = input_size;
    }

    static int num_calls = 0;
    if(num_calls < 50)
    { // we only log the first N calls

        log_debug() << "Input image dimensions ("
                    << the_input_view.width() << ", " << the_input_view.height() << ")"
                    << std::endl;

        log_debug() << "Channel size (" << channel_size.x << ", " << channel_size.y << ")" << std::endl;

        num_calls += 1;
    }

    if(channel_size.x == 0 or channel_size.y == 0)
    {
        log_error() << "Input image dimensions ("
                    << the_input_view.width() << ", " << the_input_view.height() << ")"
                    << std::endl;

        throw std::runtime_error("Input image for IntegralChannelsForPedestrians::set_image was too small");
    }

    // allocate the channel images
    input_channels.resize(boost::extents[num_channels][input_size.y][input_size.x]);
    channels.resize(boost::extents[num_channels][channel_size.y][channel_size.x]);
    integral_channels.resize(boost::extents[num_channels][channel_size.y+1][channel_size.x+1]);
    // since  hog_input_channels[angle_index][y][x] does not set the value for all hog channels,
    // we need to set them all to zero
    fill(input_channels, 0);
    // all other channels will be completelly overwritten, so no need to fill them in

    // copy the input image
    input_image.recreate(input_size);
    input_image_view = boost::gil::const_view(input_image);
    gil::copy_pixels(the_input_view, boost::gil::view(input_image));
    return;
}


void IntegralChannelsForPedestrians::compute()
{
    //compute_v0();
    compute_v1();

    return;
}


void compute_derivative(cv::InputArray _src, cv::OutputArray _dst, int ddepth, const int dx, const int dy)
{
    cv::Mat src = _src.getMat();
    if (ddepth < 0)
    {
        ddepth = src.depth();
    }
    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    cv::Mat dst = _dst.getMat();

    //const int kernel_type = std::max(CV_32F, std::max(ddepth, src.depth()));
    //const int kernel_type = cv::DataType<float>::type;
    //const int kernel_type = cv::DataType<boost::int8_t>::type;

    static cv::Ptr<cv::FilterEngine> dx_filter, dy_filter;
    if(dx_filter.empty())
    {
        const cv::Mat dx_kernel = (cv::Mat_<boost::int8_t>(1, 3) << -1, 0, 1);
        dx_filter = cv::createLinearFilter(src.type(), dst.type(), dx_kernel);
    }

    if(dy_filter.empty())
    {
        const cv::Mat dy_kernel = (cv::Mat_<boost::int8_t>(3, 1) << -1, 0, 1);
        dy_filter = cv::createLinearFilter(src.type(), dst.type(), dy_kernel);
    }

    if(false)
    {/*
        cv::Mat kx, ky;
        const int ksize =3;
        cv::getDerivKernels( kx, ky, 1, 0, ksize, true, cv::DataType<float>());
        dx_filter = cv::createLinearFilter(src.type(), dst.type(), kx);

        dy_filter = cv::createLinearFilter(src.type(), dst.type(), dy_kernel);

        dx_filter = kx;
*/
    }

    if((dx == 1) and (dy == 0))
    {
        dx_filter->apply(src, dst);
    }
    else if((dx == 0) and (dy == 1))
    {
        dy_filter->apply(src, dst);
    }
    else
    {
        throw std::runtime_error("compute_derivative received an non-supported dx, dy pair");
    }

    return;
}


/// Imitating Pedro Felzenszwalb HOG code, we take the maximum of each color channel
void compute_color_derivatives(cv::InputArray _src, cv::OutputArray _dst_dx, cv::OutputArray _dst_dy)
{
    const cv::Mat src = _src.getMat();
    if(src.type() != CV_8UC3)
    {
        throw std::invalid_argument("compute_color_derivative only accepts rgb8 input images");
    }

    _dst_dx.create( src.size(), CV_MAKETYPE(CV_16S, src.channels()) );
    _dst_dy.create( src.size(), CV_MAKETYPE(CV_16S, src.channels()) );
    cv::Mat
            df_dx = _dst_dx.getMat(),
            df_dy = _dst_dy.getMat();

    // FIXME _dst_dx.create borders will be initialized to zero automagically ?
    for(int row=1; row < (src.rows - 1); row +=1)
    {
        for(int col=1; col < (src.cols - 1); col +=1)
        {
            const cv::Vec3b
                    xa = src.at<cv::Vec3b>(row, col + 1), xb = src.at<cv::Vec3b>(row, col - 1),
                    ya = src.at<cv::Vec3b>(row + 1, col), yb = src.at<cv::Vec3b>(row - 1, col);

            const boost::int16_t
                    dx_r = xa[0] - xb[0],
                    dx_g = xa[1] - xb[1],
                    dx_b = xa[2] - xb[2],
                    dx_rg = std::max(dx_r, dx_g),
                    dx_rgb = std::max(dx_rg, dx_b),
                    dy_r = ya[0] - yb[0],
                    dy_g = ya[1] - yb[1],
                    dy_b = ya[2] - yb[2],
                    dy_rg = std::max(dy_r, dy_g),
                    dy_rgb = std::max(dy_rg, dy_b);

            df_dx.at<boost::int16_t>(row, col) = dx_rgb;
            df_dy.at<boost::int16_t>(row, col) = dy_rgb;
        } // end of "for each column"
    } // end of "for each row"

    return;
}


inline
void compute_hog_channels(const cv::Mat &df_dx, const cv::Mat &df_dy,
                          const IntegralChannelsForPedestrians::input_image_view_t::point_t &input_size,
                          const IntegralChannelsForPedestrians::angle_bin_computer_t &angle_bin_computer,
                          const float magnitude_scaling,
                          IntegralChannelsForPedestrians::input_channels_t &input_channels)
{

    const int num_angle_bins = angle_bin_computer.get_num_bins();
    const float angle_quantum = M_PI/num_angle_bins;

    bool max_magnitude_too_large = false;

    const cv::Mat_<int16_t>
            &const_df_dx = df_dx,
            &const_df_dy = df_dy;

#pragma omp parallel for
    for(int y=0; y < input_size.y; y+=1)
    {
        for(int x = 0; x < input_size.x; x += 1)
        {
            const float
                    dx = const_df_dx(y,x),
                    dy = const_df_dy(y,x);

            float magnitude = sqrt(dx*dx+dy*dy) * magnitude_scaling;

            if(false and (x == 152) and (y == 91))
            {
                printf("Magnitude at (152, 91) == %.3f\n", magnitude);
            }

            assert(magnitude < 256);
            if(true and magnitude >= 256)
            {
                printf("magnitude == %.3f <? 256\n", magnitude);
                //max_magnitude_too_large = true;
                //throw std::runtime_error("Error on magnitude_scaling inside IntegralChannelsForPedestrians::compute");
                magnitude = 255;
            }

            const uint8_t magnitude_u8 = static_cast<uint8_t>(magnitude);

            const bool do_soft_binning = false; // FIXME this should be a program option
            if(do_soft_binning == false)
            {
                // no soft binning

                // Using atan2 runs at 58 Hz,
                // while AngleBinComputer runs at 62 Hz
                const bool use_atan2 = false;
                if(use_atan2)
                {
                    float angle = atan2(dy, dx)  + (angle_quantum/2);
                    if(angle < 0)
                    {
                        angle += M_PI; // reflect negative angles
                    }
                    assert(angle >= 0);
                    const int angle_index = static_cast<int>(angle / angle_quantum) % num_angle_bins;
                    input_channels[angle_index][y][x] = magnitude_u8;
                }
                else
                {
                    const int angle_index = angle_bin_computer(dy, dx);
                    input_channels[angle_index][y][x] = magnitude_u8;
                }
            }
            else
            { // soft binning enabled

                // FIXME implement softbinning using atan2

                for(int angle_bin_index = 0; angle_bin_index < num_angle_bins; angle_bin_index += 1)
                {
                    const float soft_value = angle_bin_computer.soft_binning(dy, dx, angle_bin_index);
                    // 0 <= soft_value <= abs(dx + dy)
                    const float soft_value_scaling = 1/2.0f;
                    // FIXME this scaling is brittle, depends on the dx/dy computing method
                    input_channels[angle_bin_index][y][x] = static_cast<uint8_t>(soft_value*soft_value_scaling);
                }
            }

            input_channels[num_angle_bins][y][x] = magnitude_u8;

        } // end of "for each column"
    } // end of "for each row"

    if(max_magnitude_too_large)
    {
        throw std::runtime_error("Error on magnitude_scaling inside IntegralChannelsForPedestrians::compute");
    }

    return;
}


void IntegralChannelsForPedestrians::compute_hog_channels_v0()
{
    // 6 gradient orientations channels, 1 gradient magnitude channel
    cv::Mat gray_input_mat, df_dx, df_dy;

    cv::cvtColor(smoothed_input_mat, gray_input_mat, CV_RGB2GRAY);
    cv::Sobel(gray_input_mat, df_dx, CV_16S, 1, 0);
    cv::Sobel(gray_input_mat, df_dy, CV_16S, 0, 1);

    if(false)
    {
        cv::imwrite("sobel_x.png", df_dx);
        cv::imwrite("sobel_y.png", df_dy);

        throw std::runtime_error("Stopped everything for debugging purposes");
    }

    if(df_dx.type() != CV_16SC1)
    {
        log_error() << "df_dx.type() == " << df_dx.type() << std::endl;
        log_error() << "df_dx.depth() == " << df_dx.depth() << std::endl;
        log_error() << "df_dx.channels() == " << df_dx.channels() << std::endl;
        throw std::runtime_error("cv::Sobel returned matrices of unexpected type");
    }

    //const float max_magnitude = 255*4; // works with input_mat.copyTo(smoothed_input_mat)
    //const float max_magnitude = sqrt(2)*(255*2); // works with GaussianFilter and pre_smoothing_filter
    const float max_magnitude = sqrt(2)*(255*2); // kinda works with cv::Sobel, but not really
    const float magnitude_scaling = 255.0/max_magnitude;

    const bool print_df_min_max = false;
    if(print_df_min_max)
    {
        double min_df_dx, max_df_dx, min_df_dy, max_df_dy;
        cv::minMaxLoc(df_dx, &min_df_dx, &max_df_dx);
        cv::minMaxLoc(df_dy, &min_df_dy, &max_df_dy);

        log_info() << "max(abs(df_dx)) == " << std::max(std::abs(min_df_dx), std::abs(max_df_dx)) << std::endl;
        log_info() << "max(abs(df_dy)) == " << std::max(std::abs(min_df_dy), std::abs(max_df_dy)) << std::endl;
    }

    compute_hog_channels(df_dx, df_dy, input_size, angle_bin_computer, magnitude_scaling, input_channels);
    return;
}


void IntegralChannelsForPedestrians::compute_hog_channels_v1()
{
    // 6 gradient orientations channels, 1 gradient magnitude channel
    cv::Mat gray_input_mat, df_dx, df_dy;

    const bool use_gray_derivatives = true;
    if(use_gray_derivatives)
    {
        cv::cvtColor(smoothed_input_mat, gray_input_mat, CV_RGB2GRAY);
        compute_derivative(gray_input_mat, df_dx, CV_16S, 1, 0);
        compute_derivative(gray_input_mat, df_dy, CV_16S, 0, 1);
    }
    else
    {
        compute_color_derivatives(smoothed_input_mat, df_dx, df_dy);
    }

    if(false)
    {
        cv::imwrite("df_dx.png", df_dx);
        cv::imwrite("df_dy.png", df_dy);

        throw std::runtime_error("Stopped everything for debugging purposes");
    }


    if(df_dx.type() != CV_16SC1)
    {
        log_error() << "df_dx.type() == " << df_dx.type() << std::endl;
        log_error() << "df_dx.depth() == " << df_dx.depth() << std::endl;
        log_error() << "df_dx.channels() == " << df_dx.channels() << std::endl;
        throw std::runtime_error("compute_derivative returned matrices of unexpected type");
    }

    const float max_magnitude = sqrt(2)*255; // works compute_derivative
    const float magnitude_scaling = 255.0/max_magnitude;

    const bool print_df_min_max = false;
    if(print_df_min_max)
    {
        double min_df_dx, max_df_dx, min_df_dy, max_df_dy;
        cv::minMaxLoc(df_dx, &min_df_dx, &max_df_dx);
        cv::minMaxLoc(df_dy, &min_df_dy, &max_df_dy);

        log_info() << "max(abs(df_dx)) == " << std::max(std::abs(min_df_dx), std::abs(max_df_dx)) << std::endl;
        log_info() << "max(abs(df_dy)) == " << std::max(std::abs(min_df_dy), std::abs(max_df_dy)) << std::endl;
    }

    compute_hog_channels(df_dx, df_dy, input_size, angle_bin_computer, magnitude_scaling, input_channels);
    return;
}


void IntegralChannelsForPedestrians::compute_luv_channels_v0()
{

    const ptrdiff_t input_channel_rowsize_in_bytes = input_size.x*sizeof(input_channels_t::element);

    const gil::rgb8_planar_view_t color_input_channels =
            gil::planar_rgb_view(
                input_size.x, input_size.y,
                reinterpret_cast<boost::uint8_t*>(input_channels[7].origin()),
                reinterpret_cast<boost::uint8_t*>(input_channels[8].origin()),
                reinterpret_cast<boost::uint8_t*>(input_channels[9].origin()),
                input_channel_rowsize_in_bytes);

    const bool use_opencv = false;
    if(use_opencv == false)
    {
        assert(smoothed_input_mat.type() == CV_8UC3);
        const gil::rgb8c_view_t rgb_view =
                gil::interleaved_view(smoothed_input_mat.cols, smoothed_input_mat.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(smoothed_input_mat.data),
                                      static_cast<size_t>(smoothed_input_mat.step));

        const gil::dev3n8_planar_view_t luv_view = color_input_channels;
        fast_rgb_to_luv(rgb_view, luv_view);
    }
    else
    {
        luv_mat.create(smoothed_input_mat.rows, smoothed_input_mat.cols,
                       smoothed_input_mat.type());

        cv::cvtColor(smoothed_input_mat, luv_mat, CV_RGB2Luv);

        const gil::rgb8c_view_t luv_mat_view =
                gil::interleaved_view(luv_mat.cols, luv_mat.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(luv_mat.data),
                                      static_cast<size_t>(luv_mat.step));

        gil::copy_pixels(luv_mat_view, color_input_channels);
    }

    return;
}


void IntegralChannelsForPedestrians::resize_channel_v0(const input_channel_t input_channel, channel_t channel)
{

    // we must shift the data to compensate for the complementary loss during pyrDown
    int uint16_scaling_factor = 1;
    if(shrinking_factor == 2)
    {
        uint16_scaling_factor = 4; // 2**2
    }
    else if(shrinking_factor == 4)
    {
        uint16_scaling_factor = 16; // 2**4
    }


    gil::gray8c_view_t input_channel_view =
            gil::interleaved_view(input_size.x, input_size.y,
                                  reinterpret_cast<gil::gray8c_pixel_t *>(input_channel.origin()),
                                  input_size.x*sizeof(input_channel_t::element));
    gil::opencv::ipl_image_wrapper input_channel_ipl = gil::opencv::create_ipl_image(input_channel_view);
    cv::Mat input_channel_mat(input_channel_ipl.get());

    if(boost::is_same<channels_t::element, boost::uint8_t>::value)
    {
        // input_channel_mat is of type 8 bytes, one_half and one_fourth will be of the same type
    }
    else if(boost::is_same<channels_t::element, boost::uint16_t>::value)
    {
        cv::Mat input_channel_uint8_mat(input_channel_ipl.get());
        // convert input_channel_ipl into uint16_t and store inside input_channel_mat
        // we must shift the data to compensate for the complementary loss during pyrDown
        input_channel_uint8_mat.convertTo(input_channel_mat, cv::DataType<boost::uint16_t>::type, uint16_scaling_factor);
        // one_half and one_fourth will be of uint16_t too
    }
    else
    {
        throw std::runtime_error("IntegralChannelsForPedestrians::compute() was compiled with an unsupport type for channels_t::element");
    }

    cv::Mat one_half, one_fourth;

    if(shrinking_factor == 4)
    {
        cv::pyrDown(input_channel_mat, one_half);
        cv::pyrDown(one_half, one_fourth);
    }
    else if (shrinking_factor == 2)
    {
        cv::pyrDown(input_channel_mat, one_half);
        one_fourth = one_half;
    }
    else
    {
        one_fourth = input_channel_mat;
    }

    // copy one_fourth to this->channel
    {
        const ptrdiff_t channel_rowsize_in_bytes = channel_size.x*sizeof(channels_t::element);

        if(boost::is_same<channels_t::element, boost::uint8_t>::value)
        {
            if(one_fourth.type() != cv::DataType<boost::uint8_t>::type )
            {
                throw std::runtime_error("Something went wrong with the opencv data types inside IntegralChannelsForPedestrians::compute()");
            }
            const gil::gray8c_view_t one_fourth_view =
                    gil::interleaved_view(one_fourth.cols, one_fourth.rows,
                                          reinterpret_cast<gil::gray8c_pixel_t*>(one_fourth.data),
                                          static_cast<size_t>(one_fourth.step));

            const gil::gray8_view_t channel_view =
                    gil::interleaved_view(channel_size.x, channel_size.y,
                                          reinterpret_cast<gil::gray8_pixel_t *>(channel.origin()),
                                          channel_rowsize_in_bytes);

            gil::copy_pixels(one_fourth_view, channel_view);
        }
        else if(boost::is_same<channels_t::element, boost::uint16_t>::value)
        {
            if(one_fourth.type() != cv::DataType<boost::uint16_t>::type )
            {
                throw std::runtime_error("Something went wrong with the opencv data types inside IntegralChannelsForPedestrians::compute()");
            }

            const gil::gray16c_view_t one_fourth_view =
                    gil::interleaved_view(one_fourth.cols, one_fourth.rows,
                                          reinterpret_cast<gil::gray16c_pixel_t*>(one_fourth.data),
                                          static_cast<size_t>(one_fourth.step));

            const gil::gray16_view_t channel_view =
                    gil::interleaved_view(channel_size.x, channel_size.y,
                                          reinterpret_cast<gil::gray16_pixel_t *>(channel.origin()),
                                          channel_rowsize_in_bytes);

            gil::copy_pixels(one_fourth_view, channel_view);
        }
        else
        {
            throw std::runtime_error("IntegralChannelsForPedestrians::compute() was compiled with an unsupport type for channels_t::element");
        }
    }

    return;
}


void set_test_integral_image(IntegralChannelsForPedestrians::channels_t &channels)
{
    // dummy test integral image, used for debugging only

    for(size_t channel_index = 0; channel_index<channels.size(); channel_index += 1)
    {
        IntegralChannelsForPedestrians::channels_t::reference channel = channels[channel_index];
        for(size_t row=0; row < channel.shape()[0]; row+=1)
        {
            for(size_t col=0; col < channel.shape()[1]; col+=1)
            {
                const float row_scale = 100.0f/(channel.shape()[0]);
                const float col_scale = 10.0f/(channel.shape()[1]);
                channel[row][col] = static_cast<boost::uint8_t>(min(255.0f, row_scale*row + col_scale*col + channel_index));
            } // end of "for each col"
        } // end of "for each row"
    } // end of "for each channel"

    return;
}


void IntegralChannelsForPedestrians::resize_channels_v0()
{

#pragma omp parallel for
    // we compute all channels in parallel
    for(size_t c = 0; c < input_channels.shape()[0]; c += 1)
    { // for all orientation channels and the gradient magnitude

        const input_channel_t input_channel = input_channels[c];
        channel_t channel = channels[c];

        resize_channel_v0(input_channel, channel);

    } // end of "for each channel"


    const bool should_set_test_integral_image = false;
    if(should_set_test_integral_image)
    {
        set_test_integral_image(channels);
    }

    return;
}



void IntegralChannelsForPedestrians::resize_channel_v1(const input_channel_t input_channel, channel_t channel)
{

    gil::gray8c_view_t input_channel_view =
            gil::interleaved_view(input_size.x, input_size.y,
                                  reinterpret_cast<gil::gray8c_pixel_t *>(input_channel.origin()),
                                  input_size.x*sizeof(input_channel_t::element));
    gil::opencv::ipl_image_wrapper input_channel_ipl = gil::opencv::create_ipl_image(input_channel_view);

    // this function is called from multiple threads, so it should not share matrices
    cv::Mat input_channel_uint8_mat(input_channel_ipl.get()), input_channel_mat, resized_channel_mat;

    // here we actually resize the channel --
    {
        // convert input_channel_ipl into uint16_t and store inside input_channel_mat
        input_channel_uint8_mat.convertTo(input_channel_mat, cv::DataType<boost::uint16_t>::type);


        // FIXME does INTER_AREA average the pixels in the area ?
        cv::resize(input_channel_mat, resized_channel_mat,
                   cv::Size(channel_size.x, channel_size.y), 0, 0,
                   cv::INTER_AREA);
    }

    // copy the result to the channel
    {
        const ptrdiff_t channel_rowsize_in_bytes = channel_size.x*sizeof(channels_t::element);

        if(resized_channel_mat.type() != cv::DataType<boost::uint16_t>::type )
        {
            throw std::runtime_error("Something went wrong with the opencv data types inside IntegralChannelsForPedestrians::compute()");
        }

        const gil::gray16c_view_t resized_input_channel_view =
                gil::interleaved_view(resized_channel_mat.cols, resized_channel_mat.rows,
                                      reinterpret_cast<gil::gray16c_pixel_t*>(resized_channel_mat.data),
                                      static_cast<size_t>(resized_channel_mat.step));

        const gil::gray16_view_t channel_view =
                gil::interleaved_view(channel_size.x, channel_size.y,
                                      reinterpret_cast<gil::gray16_pixel_t *>(channel.origin()),
                                      channel_rowsize_in_bytes);

        gil::copy_pixels(resized_input_channel_view, channel_view);
    }

    return;
}


void IntegralChannelsForPedestrians::resize_channels_v1()
{

#pragma omp parallel for
    // we compute all channels in parallel
    for(size_t c = 0; c < input_channels.shape()[0]; c += 1)
    { // for all orientation channels and the gradient magnitude

        const input_channel_t input_channel = input_channels[c];
        channel_t channel = channels[c];

        resize_channel_v1(input_channel, channel);

    } // end of "for each channel"


    const bool should_set_test_integral_image = false;
    if(should_set_test_integral_image)
    {
        set_test_integral_image(channels);
    }

    return;
}


void IntegralChannelsForPedestrians::integrate_channels_v0()
{

    // compute and store the channel integrals
#pragma omp parallel for
    for(size_t channel_index = 0; channel_index<channels.size(); channel_index += 1)
    {
        // for some strange reason explicitly defining this reference is needed to get the code compiling
        integral_channels_t::reference integral_channel = integral_channels[channel_index];
        integrate(channels[channel_index], integral_channel);
    }

    return;
}


void IntegralChannelsForPedestrians::compute_v0()
{
    // in OpenCv 2.2 pyrDown, cvtColor and integral/integrate are all non-parallel operations
    // when possible, we run each channel task in parallel

    // smooth the input image
    {
        const gil::opencv::ipl_image_wrapper input_ipl = gil::opencv::create_ipl_image(input_image_view);
        const cv::Mat input_mat(input_ipl.get());
        smoothed_input_mat.create(input_mat.size(), input_mat.type());

        // smoothing the input
        pre_smoothing_filter_p->apply(input_mat, smoothed_input_mat);
    }

    compute_hog_channels_v0();
    compute_luv_channels_v0();
    resize_channels_v0();
    integrate_channels_v0();

    return;
}


void IntegralChannelsForPedestrians::compute_v1()
{

    if(boost::is_same<channels_t::element, boost::uint16_t>::value == false)
    {
        throw std::logic_error("IntegralChannelsForPedestrians::compute_v1 only supports uint16_t channels_t");
    }

    // in OpenCv 2.2 pyrDown, cvtColor and integral/integrate are all non-parallel operations
    // when possible, we run each channel task in parallel

    const gil::opencv::ipl_image_wrapper input_ipl = gil::opencv::create_ipl_image(input_image_view);
    const cv::Mat input_mat(input_ipl.get());

    if(use_presmoothing)
    {
        smoothed_input_mat.create(input_mat.size(), input_mat.type());

        // smoothing the input
        pre_smoothing_filter_p->apply(input_mat, smoothed_input_mat);
    }
    else
    {
        input_mat.copyTo(smoothed_input_mat);
    }

    compute_hog_channels_v1();
    compute_luv_channels_v0();
    resize_channels_v1();
    integrate_channels_v0();

    return;
}


void IntegralChannelsForPedestrians::save_channels_to_file()
{

    const string filename = "integral_channels_for_pedestrians.png";
    save_integral_channels_to_file(integral_channels, filename);
    log_info() << "Created image " << filename << std::endl;

    return;
}

#if defined(OBJECTS_DETECTION_LIB)

void save_integral_channels_to_file(const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
                                    const string file_path)
{
    throw std::runtime_error("save_integral_channels_to_file not supported in this compiled version");
    return;
}

#else

void get_channel_matrix(const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
                        const size_t channel_index,
                        Eigen::MatrixXf &channel_matrix)
{
    const size_t num_channels = integral_channels.shape()[0];

    if(channel_index >= num_channels)
    {
        throw std::invalid_argument("channel_index >= num_channels in get_channel_matrix(...)");
    }

    const size_t channel_size_x = integral_channels.shape()[2] - 1, channel_size_y = integral_channels.shape()[1] - 1;

    IntegralChannelsForPedestrians::integral_channels_t::const_reference
            integral_channel = integral_channels[channel_index];

    // reconstruct "non integral image" from integral image
    channel_matrix = Eigen::MatrixXf::Zero(channel_size_y, channel_size_x);

    for(size_t y=0; y < channel_size_y; y+=1)
    {
        for(size_t x = 0; x < channel_size_x; x += 1)
        {
            const uint32_t
                    a = integral_channel[y][x],
                    b = integral_channel[y+0][x+1],
                    c = integral_channel[y+1][x+1],
                    d = integral_channel[y+1][x+0];
            channel_matrix(y,x) = a +c -b -d;
        } // end of "for each column"
    } // end of "for each row"

    return;
}


void save_integral_channels_to_file(const IntegralChannelsForPedestrians::integral_channels_t &integral_channels,
                                    const string file_path)
{
    const size_t num_channels = integral_channels.shape()[0];
    const size_t channel_size_x = integral_channels.shape()[2] - 1, channel_size_y = integral_channels.shape()[1] - 1;

    gil::rgb8_image_t channels_image(channel_size_x*num_channels, channel_size_y);
    gil::rgb8_view_t channels_image_view = gil::view(channels_image);
    Eigen::MatrixXf channel_matrix;

    for(size_t i = 0; i < num_channels; i += 1 )
    {
        gil::rgb8_view_t channel_view =
                gil::subimage_view(channels_image_view,
                                   channel_size_x*i, 0, channel_size_x, channel_size_y);

        // reconstruct "non integral image" from integral image
        get_channel_matrix(integral_channels, i, channel_matrix);

        // copy matrix to overall image
        draw_matrix(channel_matrix, channel_view);
    } // end of "for each channel image"

    gil::png_write_view(file_path, gil::const_view(channels_image));
    return;
}

#endif // OBJECTS_DETECTION_LIB is defined or not


int IntegralChannelsForPedestrians::get_feature_vector_length() const
{
    // FIXME hardcoded INRIAPerson window size
    return 64 * 128 * 10 / (shrinking_factor * shrinking_factor);
}


int IntegralChannelsForPedestrians::get_shrinking_factor()
{
#if defined(SHRINKING_FACTOR_4)
    return 4;
#elif defined(SHRINKING_FACTOR_2)
    return 2;
#elif defined(SHRINKING_FACTOR_1)
    return 1;
#else
    return 4; // default: 4 is the value that we use for most of our experiments
#endif
}


const IntegralChannelsForPedestrians::input_size_t IntegralChannelsForPedestrians::get_input_size() const
{
    return input_image_view.dimensions();
}


/// simple version of geometry::equals
/// avoids the need for boost 1.46
inline bool equals(const point_t &p1, const point_t &p2)
{
    return (p1.x() == p2.x()) and (p1.y() == p2.y());
}

/// simple version of geometry::equals
/// avoids the need for boost 1.46
bool equals(const rectangle_t &r1, const rectangle_t &r2)
{
    bool is_equal = true;
    is_equal &= equals(r1.max_corner(), r2.max_corner());
    is_equal &= equals(r1.min_corner(), r2.min_corner());
    return is_equal;
}

#if not defined(OBJECTS_DETECTION_LIB)
void IntegralChannelsForPedestrians::get_feature_vector(const rectangle_t &r, feature_vector_t &feature_vector)
{
    const point_t canonical_size(64, 128);
    //point_t rectangle_size = r.max_corner() - r.min_corner();
    point_t rectangle_size = r.max_corner();
    geometry::subtract_point(rectangle_size, r.min_corner());

    //if(geometry::equals(rectangle_size, canonical_size) == false)
    if(equals(rectangle_size, canonical_size) == false)
    {
        throw std::runtime_error("Current version of IntegralChannelsForPedestrians::get_feature_vector only supports rectangles of size 64x128");
    }

    feature_vector.setZero(get_feature_vector_length());

    // feature vector is composed
    throw std::runtime_error("IntegralChannelsForPedestrians::get_feature_vector not yet implemented");

    return;
}
#endif


bool rectangle_is_smaller_or_equal(const rectangle_t &smaller, const rectangle_t &bigger)
{
    bool is_smaller_or_equal = true;
    is_smaller_or_equal &=     (smaller.min_corner().x() >= bigger.min_corner().x())
                               and (smaller.min_corner().y() >= bigger.min_corner().y());

    is_smaller_or_equal &=     (smaller.max_corner().x() <= bigger.max_corner().x())
                               and (smaller.max_corner().y() <= bigger.max_corner().y());

    return is_smaller_or_equal;
}


#if not defined(OBJECTS_DETECTION_LIB)
void IntegralChannelsForPedestrians::get_channels_values(const rectangle_t &input_rectangle, feature_vector_t &feature_vector)
{

    const rectangle_t channel_size_rectangle( point_t(0,0), point_t(channel_size.x, channel_size.y));
    //point_t rectangle_size = r.max_corner() - r.min_corner();
    rectangle_t channel_rectangle = input_rectangle;
    geometry::divide_value(channel_rectangle.max_corner(), shrinking_factor);
    geometry::divide_value(channel_rectangle.min_corner(), shrinking_factor);

    //geometry::strategy::within::franklin<point_t> strategy;
    //if(geometry::within(channel_rectangle, channel_size_rectangle, strategy) == false)
    if(rectangle_is_smaller_or_equal(channel_rectangle, channel_size_rectangle) == false)
    {
        log_error() << "Requested a rectangle outside of the channel size boundaries" << std::endl;
        log_error() << "input rectangle " << input_rectangle << std::endl;
        log_error() << "channel_rectangle " << channel_rectangle << std::endl;
        log_error() << "channel_size_rectangle " << channel_size_rectangle << std::endl;
        throw std::runtime_error("Requested a rectangle outsize the channel size boundaries");
    }

    const int num_channels = channels.shape()[0];
    point_t rectangle_size = channel_rectangle.max_corner();
    geometry::subtract_point(rectangle_size, channel_rectangle.min_corner());

    feature_vector.setZero(rectangle_size.x()*rectangle_size.y()*num_channels);

    int i = 0;
    // feature vector is composed of multiple channels
    for(int c = 0; c<num_channels; c += 1)
    {
        const channel_t channel = channels[c];
        for(int y=channel_rectangle.min_corner().y(); y<channel_rectangle.max_corner().y(); y+=1)
        {
            for(int x=channel_rectangle.min_corner().x(); x<channel_rectangle.max_corner().x(); x += 1)
            {
                assert(i < feature_vector.size());
                feature_vector(i) = channel[y][x];
                i += 1;
            } // end of "for each column"
        } // end of "for each row"

    } // end of "for each channel"

    // feature_vector has been modified
    return;
}
#endif // if OBJECTS_DETECTION_LIB is not defined


const IntegralChannelsForPedestrians::integral_channels_t &IntegralChannelsForPedestrians::get_integral_channels() const
{
    return integral_channels;
}


const IntegralChannelsForPedestrians::integral_channels_t &IntegralChannelsForPedestrians::get_integral_channels()
{
    return integral_channels;
}


const IntegralChannelsForPedestrians::channels_t &IntegralChannelsForPedestrians::get_channels() const
{
    return channels;
}


const IntegralChannelsForPedestrians::input_channels_t &IntegralChannelsForPedestrians::get_input_channels_uint8()
{
    return input_channels;
}


const IntegralChannelsForPedestrians::channels_t &IntegralChannelsForPedestrians::get_input_channels_uint16()
{
    throw std::runtime_error("IntegralChannelsForPedestrians does not implement get_channels_uint16");
    static IntegralChannelsForPedestrians::channels_t empty;
    return empty;
}


} // end of namespace doppia
