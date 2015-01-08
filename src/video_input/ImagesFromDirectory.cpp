#include "ImagesFromDirectory.hpp"

#include "helpers/Log.hpp"

#include <boost/version.hpp>
#include <boost/filesystem.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io_new/pnm_read.hpp>

#include <stdexcept>

namespace
{

using namespace std;
using namespace boost;

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ImagesFromDirectory");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ImagesFromDirectory");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "ImagesFromDirectory");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ImagesFromDirectory");
}


} // end of anonymous namespace



namespace doppia {

using namespace std;
using namespace boost;


gil::rgb8c_view_t open_image(const string &file_path, gil::rgb8_image_t &image)
{
    const string extension = filesystem::extension(file_path);
    if(extension == ".png")
    {
        try
        {
            gil::png_read_and_convert_image(file_path, image);
        }
        catch(std::exception &e)
        {
            log_error() << "Failed to read png image " << file_path << std::endl;
            throw e;
        }
    }
    else if(extension == ".pnm")
    {
        try
        {
            gil::read_and_convert_image(file_path, image, gil::pnm_tag());
        }
        catch(std::exception &e)
        {
            log_error() << "Failed to read pnm image " << file_path << std::endl;
            throw e;
        }
    }
    else if((extension == ".jpg") or (extension == ".jpeg"))
    {
        try
        {
            gil::jpeg_read_and_convert_image(file_path, image);
        }
        catch(std::exception &e)
        {
            log_error() << "Failed to read jpeg image " << file_path << std::endl;
            throw e;
        }
    }
    else
    {
        log_error() << "Received unsupported image extension " << extension << std::endl;
        log_error() << "Filename: " << file_path << std::endl;
        throw std::invalid_argument("Received unsupported image extension");
    }

    return gil::view(image);
}



ImagesFromDirectory::ImagesFromDirectory(const boost::filesystem::path &path)
    : frames_counter(0)
{
    if( filesystem::is_directory(path) == false)
    {
        throw std::invalid_argument("ImagesFromDirectory expects to receive a path to a directory");
    }

    the_directory_iterator = filesystem::directory_iterator(path);
    return;
}

ImagesFromDirectory::~ImagesFromDirectory()
{
    // nothing to do here
    return;
}

bool ImagesFromDirectory::next_frame()
{
    filesystem::directory_iterator directory_end_it;
    if(the_directory_iterator == directory_end_it)
    {
        return false;
    }

    input_image_path = the_directory_iterator->path().string();

    // set the image name
#if BOOST_VERSION <= 104400
    input_image_name = the_directory_iterator->path().filename();
#else
    input_image_name = the_directory_iterator->path().filename().string();
#endif

    // read the image, set the image view
    input_image_view = open_image(the_directory_iterator->path().string(), input_image);
    log_debug() << "read file " << the_directory_iterator->path() << std::endl;

    // move iterator to next image
    ++the_directory_iterator;
    frames_counter+=1;
    return true;
}

const ImagesFromDirectory::input_image_view_t &ImagesFromDirectory::get_image() const
{
    return input_image_view;
}

const std::string &ImagesFromDirectory::get_image_name() const
{
    return input_image_name;
}

const string &ImagesFromDirectory::get_image_path() const
{
    return input_image_path;
}

int ImagesFromDirectory::get_current_frame_number() const
{
    return frames_counter;
}
} // namespace doppia
