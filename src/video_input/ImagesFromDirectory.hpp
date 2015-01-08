#ifndef IMAGESFROMDIRECTORY_HPP
#define IMAGESFROMDIRECTORY_HPP

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>

#include <string>

namespace doppia {


/// Helper class that will read a sequence of files from a directory
/// This class is different from VideoFromFiles in that:
/// a) It does not assume that all images have the same size
/// b) It only reads one image at a time, instead of a stereo pair
/// This class is mainly used for monocular input processing (e.g. objects detection),
/// while VideoFromFiles is used for stereo input (e.g. stixels estimation)
/// @see VideoFromFiles
class ImagesFromDirectory
{
public:

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8c_view_t input_image_view_t;
    typedef input_image_view_t::point_t dimensions_t;

    ImagesFromDirectory(const boost::filesystem::path &path);
    ~ImagesFromDirectory();

    bool next_frame();
    const input_image_view_t &get_image() const;
    const std::string &get_image_name() const;
    const std::string &get_image_path() const;

    int get_current_frame_number() const;

protected:

    int frames_counter;
    boost::filesystem::directory_iterator the_directory_iterator;
    input_image_t input_image;
    input_image_view_t input_image_view;
    std::string input_image_name, input_image_path;

};

/// simple helper method, that will check for the filename extension before opening the image
boost::gil::rgb8c_view_t open_image(const std::string &filename, boost::gil::rgb8_image_t &image);

} // end of namespace doppia

#endif // IMAGESFROMDIRECTORY_HPP
