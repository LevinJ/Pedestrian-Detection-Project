#include "ComputeFeatureChannelsApplication.hpp"

#include "applications/EmptyGui.hpp"
#include "objects_detection/integral_channels/AbstractChannelsComputer.hpp"
//#include "objects_detection/integral_channels/ChannelsComputerFactory.hpp"

#include "objects_detection/integral_channels/AbstractGpuIntegralChannelsComputer.hpp"
#include "objects_detection/integral_channels/IntegralChannelsComputerFactory.hpp"


#include "video_input/ImagesFromDirectory.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"
#include "helpers/progress_display_with_eta.hpp"

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>
#include <boost/filesystem/operations.hpp>

#include <omp.h>

namespace doppia {

using namespace std;
using namespace  boost;

MODULE_LOG_MACRO("ComputeFeatureChannelsApplication")
//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string ComputeFeatureChannelsApplication::get_application_title()
{
    return  "Compute feature channels. Rodrigo Benenson @ MPI-Inf. 2013-2014.";
}


ComputeFeatureChannelsApplication::ComputeFeatureChannelsApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}


ComputeFeatureChannelsApplication::~ComputeFeatureChannelsApplication()
{
    // nothing to do here
    return;
}


program_options::options_description ComputeFeatureChannelsApplication::get_options_description()
{
    program_options::options_description desc("ComputeFeatureChannelsApplication options");

    const std::string application_name = "compute_feature_channels";
    BaseApplication::add_args_options(desc, application_name);

    desc.add_options()

            ("process_folder",
             program_options::value<string>(),
             "for evaluation purposes, will process images on a folder. Normal video input will be ignored")

            ("silent_mode",
             program_options::value<bool>()->default_value(false),
             "if true, no status information will be printed at run time (use this for speed benchmarking)")

            ;

    return desc;
}


void ComputeFeatureChannelsApplication::get_all_options_descriptions(program_options::options_description &desc)
{
    desc.add(ComputeFeatureChannelsApplication::get_options_description());
    //desc.add(ImagesFromDirectory::get_args_options());
    //desc.add(ChannelsComputerFactory::get_args_options());
    desc.add(IntegralChannelsComputerFactory::get_options_description());

    return;
}


/// helper method called by setup_problem
void ComputeFeatureChannelsApplication::setup_logging(std::ofstream &log_file,
                                                      const program_options::variables_map &options)
{
    if(log_file.is_open())
    {
        // the logging is already setup
        return;
    }

    // set base logging rules --
    BaseApplication::setup_logging(log_file, options);

    const bool silent_mode = get_option_value<bool>(options, "silent_mode");
    if(silent_mode == false)
    {
        // set our own stdout rules --
        logging::LogRuleSet &rules_for_stdout = logging::get_log().console_log().rule_set();
        rules_for_stdout.add_rule(logging::InfoMessage, "ComputeFeatureChannelsApplication");
#if defined(DEBUG)
        rules_for_stdout.add_rule(logging::DebugMessage, "*"); // we are debugging this application
#else
        rules_for_stdout.add_rule(logging::InfoMessage, "*"); // "production mode"
#endif
    }

    return;
}


void ComputeFeatureChannelsApplication::setup_problem(const program_options::variables_map &options)
{
    // parse the application specific options --
    silent_mode = get_option_value<bool>(options, "silent_mode");

    // instantiate the different processing modules --
    const filesystem::path folder_to_process = get_option_value<string>(options, "process_folder");
    directory_input_p.reset(new ImagesFromDirectory(folder_to_process));

    num_files_to_process = std::distance(filesystem::directory_iterator(folder_to_process),
                                         filesystem::directory_iterator());

    //channels_computer_p.reset(ChannelsComputerFactory::new_instance(options));
    channels_computer_p.reset(
                static_cast<AbstractChannelsComputer *>(IntegralChannelsComputerFactory::new_instance(options)));


    BOOST_STATIC_ASSERT((boost::is_same<ImagesFromDirectory::input_image_view_t,
                         AbstractChannelsComputer::input_image_view_t>::value));

    return;
} // end of ComputeFeatureChannelsApplication::setup_problem


AbstractGui* ComputeFeatureChannelsApplication::create_gui(const program_options::variables_map &options)
{

    //const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    bool use_empty_gui = true;
    if(options.count("gui.disabled") > 0)
    {
        use_empty_gui = get_option_value<bool>(options, "gui.disabled");
    }

    AbstractGui *gui_p = NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        //gui_p = new ComputeWhiteningMatrixGui(*this, options);
        throw std::runtime_error("ComputeGeodesicChannelsGui not yet implemented");
    }

    return gui_p;
}


int ComputeFeatureChannelsApplication::get_current_frame_number() const
{
    int current_frame_number = 0;
    if(directory_input_p)
    {
        current_frame_number = directory_input_p->get_current_frame_number();
    }

    return current_frame_number;
}


void save_channels_to_file(const AbstractChannelsComputer::channels_ref_t& channels,
                           const boost::filesystem::path &file_path)
{
    const size_t
            num_channels = channels.shape()[0],
            channel_size_y = channels.shape()[1],
            channel_size_x = channels.shape()[2];

    typedef boost::gil::gray16_image_t channel_image_t;

    BOOST_STATIC_ASSERT((boost::is_same< boost::gil::channel_type<channel_image_t>::type,
                         AbstractChannelsComputer::channels_ref_t::element>::value));

    //BOOST_STATIC_ASSERT_MSG((boost::is_same< boost::gil::channel_type<channel_image_t>::type,
    //                         AbstractChannelsComputer::channels_ref_t::element>::value),
    //                        "The channels_t and channels_t types are not compatible");

    channel_image_t channels_image(channel_size_x*num_channels, channel_size_y);
    channel_image_t::view_t channels_image_view = gil::view(channels_image);

    //printf("channels_image size width, height == %li, %li\n",
    //       channels_image.width(), channels_image.height());

    for(size_t c = 0; c < num_channels; c += 1 )
    {
        for(size_t y = 0; y < channel_size_y; y += 1 )
        {
            for(size_t x = 0; x < channel_size_x; x += 1 )
            {
                const size_t
                        image_x = c*channel_size_x + x,
                        image_y = y;

                //channels_image_view(image_x, image_y)[0] = static_cast<boost::uint16_t>(image_y*1024);
                channels_image_view(image_x, image_y)[0] = channels[c][y][x];

            } // end of "for each column"
        } // end of "for each row"
    } // end of "for each channel"


    gil::png_write_view(file_path.string(), gil::const_view(channels_image));

    log.debug() << "Created geodesic channels file " << file_path << std::endl;

    return;
}


// in this anonymous namespace we copy paste some functions defined elsewhere,
// this is part of a HACK used to generate figured needed for the paper.
namespace
{

void allocate_shrunk_channels(const AbstractChannelsComputer::channels_t &channels,
                              const int shrinking_factor,
                              AbstractChannelsComputer::channels_t &shrunk_channels)
{

    const size_t
            num_channels = channels.shape()[0],
            channel_size_y = channels.shape()[1],
            channel_size_x = channels.shape()[2];

    size_t
            shrunk_size_y = channel_size_y,
            shrunk_size_x = channel_size_x;

    //channel_size = input_image.dimensions() / shrinking_factor;
    // +shrinking_factor/2 to round-up
    if(shrinking_factor == 4)
    {
        shrunk_size_x = (( (channel_size_x+1) / 2) + 1) / 2;
        shrunk_size_y = (( (channel_size_y+1) / 2) + 1) / 2;
    }
    else if(shrinking_factor == 2)
    {
        shrunk_size_x = (channel_size_x+1) / 2;
        shrunk_size_y = (channel_size_y+1) / 2;
    }
    else
    {
        shrunk_size_x = channel_size_x;
        shrunk_size_y = channel_size_y;
    }


    // allocate the channel images
    shrunk_channels.resize(boost::extents[num_channels][shrunk_size_y][shrunk_size_x]);
    return;
}


/// *channel_t are reference types
template<size_t shrinking_factor>
void shrink_channel(AbstractChannelsComputer::channels_t::const_reference input_channel,
                    AbstractChannelsComputer::channels_t::reference shrunk_channel)
{
    assert(shrunk_channel.shape()[0]*shrinking_factor == input_channel.shape()[0]);
    assert(shrunk_channel.shape()[1]*shrinking_factor == input_channel.shape()[1]);

    for(size_t y = 0; y < shrunk_channel.shape()[0]; y += 1)
    {
        for(size_t x = 0; x < shrunk_channel.shape()[1]; x += 1)
        {
            boost::uint32_t sum = 0;

            for(size_t yy = 0; yy < shrinking_factor; yy += 1)
            {
                const size_t t_y = y*shrinking_factor + yy;
                for(size_t xx = 0; xx < shrinking_factor; xx += 1)
                {
                    const size_t t_x = x*shrinking_factor + xx;
                    sum += input_channel[t_y][t_x];
                }
            }

            sum /= (shrinking_factor*shrinking_factor);
            shrunk_channel[y][x] = sum;
        } // end of "for each column"
    } // end of "for each row"

    return;
}


template<int shrinking_factor>
void shrink_channels(const AbstractChannelsComputer::channels_t &input_channels,
                     AbstractChannelsComputer::channels_t &shrunk_channels)
{
    assert(input_channels.shape()[0] == shrunk_channels.shape()[0]);

#pragma omp parallel for
    // we compute all channels in parallel
    for(size_t c = 0; c < input_channels.shape()[0]; c += 1)
    {
        AbstractChannelsComputer::channels_t::const_reference input_channel = input_channels[c];
        AbstractChannelsComputer::channels_t::reference shrunk_channel = shrunk_channels[c];

        shrink_channel<shrinking_factor>(input_channel, shrunk_channel);
    } // end of "for each channel"

    return;
}


void shrink_channels(const AbstractChannelsComputer::channels_t &input_channels,
                     const int shrinking_factor,
                     AbstractChannelsComputer::channels_t &shrunk_channels)
{
    switch(shrinking_factor)
    {
    case 4:
        shrink_channels<4>(input_channels, shrunk_channels);
        break;
    case 2:
        shrink_channels<2>(input_channels, shrunk_channels);
        break;
    case 1:
        shrunk_channels = input_channels; // copy operation
        break;
    default:
        throw std::invalid_argument("IntegralChannelsFromFiles::compute received an unhandled shrinking factor");
    }

    return;
}


/// copy paste the interior of the channels into their border
void patch_borders(AbstractChannelsComputer::channels_t &channels)
{

    const size_t
            height = channels.shape()[1],
            width = channels.shape()[2];

    for(size_t c = 0; c < channels.shape()[0]; c += 1)
    {
        // fix left and right borders
        for(size_t y = 0; y < height; y += 1)
        {
            channels[c][y][0] = channels[c][y][1];
            channels[c][y][width - 1] = channels[c][y][width - 2];
        }

        // fix top and bottom borders
        for(size_t x = 0; x < width; x += 1)
        {
            channels[c][0][x] = channels[c][1][x];
            channels[c][height - 1][x] = channels[c][height - 2][x];
        }

    } // end of "for each channel"

    return;
}


} // end of (nested) anonymous namespace

void ComputeFeatureChannelsApplication::main_loop()
{

    const bool should_print = not silent_mode;
    if(silent_mode)
    {
        printf("The application is running in silent mode. "
               "No information will be printed until all the frames have been processed.\n");
    }


    int num_iterations = 0;
    const int num_iterations_for_timing = 10;
    //const int num_iterations_for_timing = 100;
    double cumulated_processing_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = directory_input_p->next_frame();

    AbstractChannelsComputer::channels_t channels;
    ImagesFromDirectory::input_image_view_t input_view = directory_input_p->get_image();
    channels.resize(boost::extents[channels_computer_p->get_num_channels()][input_view.height()][input_view.width()]);

    progress_display_with_eta progress_display(num_files_to_process);
    bool end_of_game = false;

    while(video_input_is_available and (not end_of_game))
    {

        // update video input --
        ImagesFromDirectory::input_image_view_t input_view = directory_input_p->get_image();

        // we start measuring the time before uploading the data to the GPU
        const double start_processing_wall_time = omp_get_wtime();

        if((static_cast<size_t>(input_view.height()) != channels.shape()[1])
                or (static_cast<size_t>(input_view.width()) != channels.shape()[2]))
        {
            printf("First image size (height, width) == %zix%zi\n", channels.shape()[1], channels.shape()[2]);
            printf("Current input image size (height, width) == %lix%li\n", input_view.height(), input_view.width());
            throw std::invalid_argument("ComputeFeatureChannelsApplication"
                                        "expects all input images to have the same size");
        }

        channels_computer_p->set_image(input_view);
        channels_computer_p->compute();

        const AbstractChannelsComputer::channels_t &computed_channels = \
                channels_computer_p->get_input_channels_uint16();

        assert(channels.shape()[0] == computed_channels.shape()[0]);
        assert(channels.shape()[1] == computed_channels.shape()[1]);
        assert(channels.shape()[2] == computed_channels.shape()[2]);

        channels = computed_channels; // copy the elements

        // Hack to create paper image
        if(false) //and ((input_view.width() == 96) and (input_view.height() == 160)))
        {
            patch_borders(channels);

            const int shrinking_factor = 4;

            AbstractChannelsComputer::channels_t shrunk_channels;
            allocate_shrunk_channels(channels, shrinking_factor, shrunk_channels);
            shrink_channels(channels, shrinking_factor, shrunk_channels);

            /*typedef boost::multi_array_types::index_range range;
            AbstractChannelsComputer::channels_t::const_array_view<3>::type detection_channels_view =
                    shrunk_channels[
                    boost::indices[range()]
                    [range(16/shrinking_factor, (16+128)/shrinking_factor)]
                    [range(16/shrinking_factor, (16+64)/shrinking_factor)] ];

            channels.resize(boost::extents[detection_channels_view.shape()[0]]
                    [detection_channels_view.shape()[1]]
                    [detection_channels_view.shape()[2]]);

            channels = detection_channels_view; // copy the data*/

            channels.resize(boost::extents[shrunk_channels.shape()[0]]
                    [shrunk_channels.shape()[1]]
                    [shrunk_channels.shape()[2]]);

            channels = shrunk_channels; // copy the data

        }


        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;


        // save the channels in the corresponding file --
        std::string filename = directory_input_p->get_image_name();
        filename.append(".png"); // the final filename will be of the kind "something.jpeg.png"
        //const boost::filesystem::path output_filename = output_path / filename;
        const boost::filesystem::path output_filename = get_recording_path() / filename;
        save_channels_to_file(channels, output_filename);

        // update user interface --
        end_of_game = update_gui();

        num_iterations += 1;

        // false since progress_display will provide the corresponding information
        if(false and should_print and ((num_iterations % num_iterations_for_timing) == 0))
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }


        ++progress_display; // update progress display

        // retrieve next input image
        video_input_is_available = directory_input_p->next_frame();

    } // end of "while video input and not end of game"


    printf("Processed a total of %i input frames\n", num_iterations);
    if(cumulated_processing_time > 0)
    {
        printf("Average processing time per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }


    return;
} // end of void ComputeFeatureChannelsApplication::main_loop



} // end of namespace doppia
