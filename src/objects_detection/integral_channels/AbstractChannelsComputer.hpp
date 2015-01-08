#ifndef DOPPIA_ABSTRACTCHANNELSCOMPUTER_HPP
#define DOPPIA_ABSTRACTCHANNELSCOMPUTER_HPP

#include <Eigen/Core>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/multi_array.hpp>
#include <boost/cstdint.hpp>


namespace doppia
{

// FIXME this should be an enum
const static int
FILTER = 0,
STUFF = 1,
THING = 2,
MULTI = 3;

class AbstractChannelsComputer
{
public:

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8c_view_t input_image_view_t;

    typedef Eigen::aligned_allocator<boost::uint8_t> aligned_uint8_allocator;

    /// channels as computed from the input image, before shrinking
    typedef boost::multi_array<boost::uint8_t, 3, aligned_uint8_allocator> input_channels_t;
    typedef input_channels_t::reference input_channel_t;

    // uint8_t is enough when resizing factor is 1
    // when using resizing factor 4, uint16_t allows to avoid additional quantization
    // (in practice, we always use resizing factor 4)
    //typedef boost::uint8_t channels_value_t;
    typedef boost::uint16_t channels_value_t;
    typedef boost::multi_array<channels_value_t, 3> channels_t;
    typedef channels_t::reference channel_t;
    typedef channels_t::const_reference const_channel_t;

    typedef boost::multi_array_ref<channels_value_t, 3> channels_ref_t;
    typedef channels_ref_t::reference channel_ref_t;

public:
    AbstractChannelsComputer();
    virtual ~AbstractChannelsComputer();

    /// how many channels are computed ?
    virtual size_t get_num_channels() const = 0;
    virtual int get_channel_type() const;

    /// set the image, and re-allocate as needed
    /// the implementation might keep on a reference to the input_view
    /// (should be kept valid until after the compute() call)
    virtual void set_image_file_name(const std::string &image_file_name_);
    virtual void set_image(const input_image_view_t &input_view,
                           const std::string &image_file_path = std::string());

    /// use to propagate information about training examples
    virtual void set_image_data(const int cutout_x1_, const int cutout_x2_,
                                const int cutout_y1_, const int cutout_y2_,
                                const int original_image_width_, const int original_image_height_);

    /// crunch numbers
    virtual void compute() = 0;

    virtual const input_channels_t &get_input_channels_uint8();

    virtual const channels_t &get_input_channels_uint16();
    std::string image_file_name;
protected:

    input_image_view_t input_view;


    int
        original_image_width,
        original_image_height,
        cutout_x1,
        cutout_x2,
        cutout_y1,
        cutout_y2;

    input_channels_t input_channels;
    channels_t channels_uint16;

    virtual void set_input_image(const input_image_view_t &input_view) = 0;

    void allocate_channels(const input_image_view_t::point_t &dimensions);

    void uint8_to_uint16_channels(const input_channels_t &uint8_channels, channels_t &uint16_channels) const;

};

} // end of namespace doppia

#endif // DOPPIA_ABSTRACTCHANNELSCOMPUTER_HPP
