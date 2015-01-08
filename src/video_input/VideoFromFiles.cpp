
#include "VideoFromFiles.hpp"

#include "calibration/StereoCameraCalibration.hpp"

#include "helpers/get_option_value.hpp"

#include <limits>

#include <boost/filesystem.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <omp.h>

namespace doppia
{

using namespace std;
using namespace boost;

program_options::options_description VideoFromFiles::get_args_options()
{
    program_options::options_description desc("VideoFromFiles options");

    desc.add_options()

            // @param mask string containing directory and filename, except
            // an %d for sprintf to be replaced by frame number, e.g. image_%08d.pgm

            ("video_input.left_filename_mask",
             program_options::value<string>(),
             "sprintf mask for left image files input. Will receive the frame number as input. Example: the_directory/left_%05d.png")

            ("video_input.right_filename_mask",
             program_options::value<string>(),
             "sprintf mask for right image files input. Will receive the frame number as input. Example: the_directory/right_%05d.png")

            ("video_input.frame_rate",
             program_options::value<int>()->default_value(15), "video input frame rate")

            ("video_input.frame_width",
             program_options::value<int>()->default_value(640), "video input frame width in pixels")

            ("video_input.frame_height",
             program_options::value<int>()->default_value(480), "video input frame height in pixels" )

            ("video_input.start_frame",
             program_options::value<int>()->default_value(0), "first image to read")

            ("video_input.end_frame",
             program_options::value<int>(), "last image to read, if omited will read all files matching the masks")
            ;


    return desc;
}


VideoFromFiles::VideoFromFiles(const program_options::variables_map &options,
                               const shared_ptr<StereoCameraCalibration> &stereo_calibration_p)
    : AbstractVideoInput(options),
      read_future_frame_start_barrier(2), // only two threads are involved
      read_future_frame_ended_barrier(2)
{

    total_number_of_frames = -1;

    this->stereo_calibration_p = stereo_calibration_p;
    // this->preprocessor_p is left uninitialzied until ::set_preprocessor is called

    left_filename_mask = get_option_value<string>(options, "video_input.left_filename_mask");
    right_filename_mask = get_option_value<string>(options, "video_input.right_filename_mask");

    if(left_filename_mask.empty() or right_filename_mask.empty())
    {
        throw std::runtime_error("Left and right input filename masks should be non empty strings");
    }

    left_filename_format = boost::format(left_filename_mask);
    right_filename_format = boost::format(right_filename_mask);

#if BOOST_VERSION >= 104400 
    // expected_args was only defined in version 1.44
    if(left_filename_format.expected_args() > 1
       or right_filename_format.expected_args() > 1)
    {
        throw std::runtime_error("Found more than one \%i argument in the left (or right) input filename masks");
    }

    if(left_filename_format.expected_args() != right_filename_format.expected_args())
    {
        throw std::runtime_error("left and right input filename masks should have the same number of arguments");
    }
#endif

    start_frame = get_option_value<int>(options, "video_input.start_frame");
    end_frame = get_option_value<int>(options, "video_input.end_frame");

    future_image_frame_number = -1;
    found_future_image = false;

    image_reading_thread = boost::thread(&VideoFromFiles::read_future_frame_thead, this);

    // do the first acquisition
    const bool found_frames = this->set_frame(start_frame);

    if(left_image_view.dimensions() != right_image_view.dimensions())
    {
        throw std::runtime_error("Left and right input images do not have the same dimensions");
    }

    if(found_frames and ((left_image_view.size() == 0) or  (right_image_view.size() == 0)))
    {
        throw std::runtime_error("Read an empty image file");
    }

    if(found_frames==false)
    {
        throw std::runtime_error("Could not read the first input frames pair");
    }

    return;
}


VideoFromFiles::~VideoFromFiles()
{
    // we stop the reading thread, before destroying the object
    image_reading_thread.interrupt();

    // we make sure the thread has done everything it should
    // this is necessary to avoid "'!pthread_mutex_destroy(&internal_mutex)' failed" exceptions
    // see https://bbs.archlinux.org/viewtopic.php?id=130195
    image_reading_thread.join();

    return;
}


const StereoCameraCalibration &VideoFromFiles::get_stereo_calibration() const
{
    if(preprocessor_p)
    {
        return preprocessor_p->get_post_processing_calibration();
    }
    else
    {
        return *stereo_calibration_p;
    }
}


void VideoFromFiles::set_preprocessor(const shared_ptr<AbstractPreprocessor> &preprocessor)
{
    this->preprocessor_p = preprocessor;
    return;
}


const shared_ptr<AbstractPreprocessor> &VideoFromFiles::get_preprocessor() const
{
    return preprocessor_p;
}


/// Advance in stream, return true if successful
bool VideoFromFiles::next_frame()
{
    return this->set_frame(current_frame_number + 1);
}


/// Go back in stream
bool VideoFromFiles::previous_frame()
{
    return this->set_frame(current_frame_number - 1);
}


int get_number_of_matching_files(boost::format &filename_format, const int start_frame_index)
{
    int num_matching_files = -1;

    int t_frame_index = start_frame_index;
    boost::filesystem::path t_path;

    bool end_of_game = false;
    while(end_of_game == false)
    {
        filename_format % t_frame_index;
        t_path = filename_format.str();

        if( boost::filesystem::exists(t_path) )
        {
            num_matching_files += 1;
        }
        else
        {
            end_of_game = true;
        }
    };

    return num_matching_files;
}


int VideoFromFiles::get_number_of_frames()
{

    // total_number_of_frames was not yet calculated
    if(total_number_of_frames < 0)
    {

        const bool dummy_number_of_frames = false;
        if(dummy_number_of_frames)
        {
            total_number_of_frames = current_frame_number;
        }
        else
        {
            if(end_frame != std::numeric_limits<int>::max())
            {
                total_number_of_frames = end_frame - start_frame;
            }
            else
            {
                const int left_frames = get_number_of_matching_files(left_filename_format, start_frame);
                const int right_frames = get_number_of_matching_files(right_filename_format, start_frame);
                total_number_of_frames = std::min(left_frames, right_frames);
            }
        }
    }

    return total_number_of_frames;
}


/// Set current absolute frame
bool VideoFromFiles::set_frame(const int frame_number)
{
    //const bool enable_parallel_read = true;
    const bool enable_parallel_read = false;

    if( (future_image_frame_number != frame_number) or
        (enable_parallel_read == false) )
    { // no future was launched to retrieve the images before hand

        const bool return_value  = read_frame_from_disk(frame_number,
                                                        left_image, right_image,
                                                        left_image_view, right_image_view);

        if(return_value == false)
        {
            return false;
        }
        else
        {
            // we continue
        }

        this->current_frame_number = frame_number;

    }
    else
    { // launched a future to read the images before hand --
        assert(frame_number == future_image_frame_number);

        // make sure future image reading has finished
        read_future_frame_ended_barrier.wait();

        this->current_frame_number = frame_number;
        if(found_future_image == false)
        {
            // the reading failed
            return false;
        }
        else
        {
            // copy the data from the future to the present
            boost::gil::copy_pixels(future_left_image_view, boost::gil::view(left_image));
            boost::gil::copy_pixels(future_right_image_view, boost::gil::view(right_image));
        }

    }

    // launch right away the reading of the next frame ---
    if(enable_parallel_read)
    {
        read_future_frame_start_barrier.wait();
    }

    // preprocess the acquired images ---
    if(this->preprocessor_p.get() != NULL)
    {
        /*  static int num_iterations = 0;
        static double cumulated_time = 0;

        const int num_iterations_for_timing = 50;
        const double start_wall_time = omp_get_wtime();
*/
        preprocessor_p->run(left_image_view, 0, boost::gil::view(this->left_image));
        preprocessor_p->run(right_image_view, 1, boost::gil::view(this->right_image));
        /*
        cumulated_time += omp_get_wtime() - start_wall_time;
        num_iterations += 1;

        const bool silent_mode = false;
        if((silent_mode == false) and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average preprocessor_p->run(...) speed  %.2lf [Hz] (in the last %i iterations)\n",
                   num_iterations / cumulated_time, num_iterations );
        }
        */
    }

    return true;
}


bool VideoFromFiles::read_frame_from_disk(const int frame_number,
                                          input_image_t &left_image, input_image_t &right_image,
                                          input_image_view_t &left_view, input_image_view_t &right_view)
{

    using namespace boost::filesystem;
    using boost::format;

    if(frame_number < start_frame or frame_number > end_frame )
    {
        printf("Requested frame number %i but frames should be in range (%i, %i)\n", frame_number, start_frame, end_frame);
        return false;
    }

    // frame_number is in a correct range --
    path left_image_path, right_image_path;

#if BOOST_VERSION >= 104400 
    // expected_args was only defined in version 1.44
    if(left_filename_format.expected_args() == 1)
#else
    if(true)
#endif
    { // already checked that right_filename_format.expected_args() == left_filename_format.expected_args()
        left_image_path = str( format(left_filename_format) % frame_number) ;
        right_image_path = str( format(right_filename_format) % frame_number);
    }
    else
    { // can only be expected_args() == 0
        left_image_path = str( format(left_filename_format) ) ;
        right_image_path = str( format(right_filename_format) );
    }


    if ((exists(left_image_path) == false) or (exists(right_image_path) == false) )
    {
        const bool print_not_found = true;
        if(print_not_found)
        {
            if(exists(left_image_path) == false)
            {
                printf("File %s not found\n", left_image_path.string().c_str());
            }

            if(exists(right_image_path) == false)
            {
                printf("File %s not found\n", right_image_path.string().c_str());
            }
        }
        return false;
    }

    // files exist --

    // let us read the images --
    const bool print_read_files = (frame_number == start_frame);
//    const bool print_read_files = true;

    if(print_read_files)
    {
        printf("Reading files:\n%s\n%s\n", left_image_path.string().c_str(), right_image_path.string().c_str());
    }

    if(left_view.size() == 0 or right_view.size() == 0)
    {
        // if views are empty, do memory allocation and create views
        boost::gil::png_read_and_convert_image(left_image_path.string(), left_image);
        boost::gil::png_read_and_convert_image(right_image_path.string(), right_image);

        left_view = boost::gil::const_view(left_image);
        right_view = boost::gil::const_view(right_image);
    }
    else
    {
        // update the views
        boost::gil::png_read_and_convert_view(left_image_path.string(), boost::gil::view(left_image));
        boost::gil::png_read_and_convert_view(right_image_path.string(), boost::gil::view(right_image));

        // no need to update this->left_image_view, this->right_image_view since they should still point the same image
        left_view = boost::gil::const_view(left_image);
        right_view = boost::gil::const_view(right_image);
    }

    return true;
}


void VideoFromFiles::read_future_frame_thead()
{
    while(true)
    { // the thread will end by a call to thread::interrupt

        read_future_frame_start_barrier.wait();

        future_image_frame_number = this->current_frame_number + 1;

        found_future_image = read_frame_from_disk(future_image_frame_number,
                                                  future_left_image, future_right_image,
                                                  future_left_image_view, future_right_image_view);

        read_future_frame_ended_barrier.wait();
    }
    return;
}


const AbstractVideoInput::input_image_view_t &VideoFromFiles::get_left_image()
{

    return this->left_image_view;
}


const AbstractVideoInput::input_image_view_t &VideoFromFiles::get_right_image()
{

    return this->right_image_view;
}


} // end of doppia namespace


