#include "AbstractChannelsComputer.hpp"

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

#include <iostream>


namespace doppia {

AbstractChannelsComputer::AbstractChannelsComputer()
{
    // nothing to do here
    return;
}


AbstractChannelsComputer::~AbstractChannelsComputer()
{
    // nothing to do here
    return;
}


/// Default implementation simply disregards the input path
void AbstractChannelsComputer::set_image(
        const AbstractChannelsComputer::input_image_view_t &input_view,
        const std::string &image_file_path)
{

    try
    {
        if(image_file_path != "")
        {
            image_file_name = image_file_path;
			//TODO - find a better way to do this:
            /*std::cout << image_file_path << std::endl;
            typedef boost::char_separator<char> separator_t;
            typedef boost::tokenizer< separator_t > tokenizer_t;
            separator_t filename_sep(".");
            tokenizer_t filename_tokens(image_file_path, filename_sep);

            tokenizer_t::iterator filename_token = filename_tokens.begin(); ++filename_token;
            if((*filename_token).length() > 3)
            {
                separator_t imagedata_sep("_");
                tokenizer_t imagedata_tokens(*filename_token, imagedata_sep);
                tokenizer_t::iterator imagedata_token = imagedata_tokens.begin();
                cutout_x1 = atoi((*imagedata_token).c_str()); ++imagedata_token;
                cutout_y1 = atoi((*imagedata_token).c_str()); ++imagedata_token;
                cutout_x2 = atoi((*imagedata_token).c_str()); ++imagedata_token;
                cutout_y2 = atoi((*imagedata_token).c_str()); ++imagedata_token;
                original_image_width = atoi((*imagedata_token).c_str()); ++imagedata_token;
                original_image_height = atoi((*imagedata_token).c_str()); ++imagedata_token;
            }
            else
            {
                cutout_x1 = 0;
                cutout_y1 = 0;
                cutout_x2 = input_view.width();
                cutout_y2 = input_view.height();
                original_image_width = cutout_x2;
                original_image_height = cutout_y2;
            }*/
        }
        else
        {
            // empty image file path
        }
    }
    catch (std::exception & e)
    {
        std::cout << "Something went wrong while parsing filename: " << image_file_path << std::endl;
        std::cout << e.what() << std::endl;
    }

    set_input_image(input_view);
    return;
}


void AbstractChannelsComputer::set_image_data(const int cutout_x1_, const int cutout_x2_,
                                              const int cutout_y1_, const int cutout_y2_,
                                              const int original_image_width_, const int original_image_height_)
{

    cutout_x1 = cutout_x1_;
    cutout_y1 = cutout_y1_;
    cutout_x2 = cutout_x2_;
    cutout_y2 = cutout_y2_;
    original_image_width = original_image_width_;
    original_image_height = original_image_height_;

    return;
}


void AbstractChannelsComputer::set_image_file_name(const std::string &image_file_name_)
{
	image_file_name = image_file_name_;
	return;
}


int AbstractChannelsComputer::get_channel_type() const
{
    return doppia::FILTER; //default channel type
}


const AbstractChannelsComputer::input_channels_t &AbstractChannelsComputer::get_input_channels_uint8()
{
    return input_channels;
}


const AbstractChannelsComputer::channels_t &AbstractChannelsComputer::get_input_channels_uint16()
{
    return channels_uint16;
}


void AbstractChannelsComputer::allocate_channels(const input_image_view_t::point_t &dimensions)
{
    const size_t num_channels = get_num_channels();

    // re-allocate the channels
    if ((input_channels.shape()[0] != num_channels)
            or (input_channels.shape()[1] != static_cast<size_t>(dimensions.y))
            or (input_channels.shape()[2] != static_cast<size_t>(dimensions.x)))
    {
        input_channels.resize(boost::extents[num_channels][dimensions.y][dimensions.x]);
        channels_uint16.resize(boost::extents[num_channels][dimensions.y][dimensions.x]);
    }

    return;
}


void AbstractChannelsComputer::uint8_to_uint16_channels(
        const AbstractChannelsComputer::input_channels_t &uint8_channels,
        AbstractChannelsComputer::channels_t &uint16_channels) const
{

    assert(get_num_channels() <= uint8_channels.shape()[0]);
    uint16_channels.resize(
                //boost::extents[uint8_channels.shape()[0]]
                boost::extents[get_num_channels()] // in case num channels is not all the channels
            [uint8_channels.shape()[1]]
            [uint8_channels.shape()[2]]);

    //uint16_channels = uint8_channels; // copies data (and does implicit cast)

    for(size_t c = 0; c < uint16_channels.shape()[0]; c += 1)
    {
        for(size_t y = 0; y < uint16_channels.shape()[1]; y += 1)
        {
            for(size_t x = 0; x < uint16_channels.shape()[2]; x += 1)
            {
                // converts from uint8_t range to uint16_t
                uint16_channels[c][y][x] = uint8_channels[c][y][x] << 8;
            }
        }
    } // end of "for each channel"

    return;
}


} // end of namespace doppia
