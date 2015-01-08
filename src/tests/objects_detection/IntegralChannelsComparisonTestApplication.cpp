#include "IntegralChannelsComparisonTestApplication.hpp"


#include "objects_detection/integral_channels/IntegralChannelsForPedestrians.hpp"
#include "objects_detection/integral_channels/GpuIntegralChannelsForPedestrians.hpp"

#include "video_input/ImagesFromDirectory.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_matrix.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/for_each_multi_array.hpp"

#include <boost/foreach.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <boost/test/test_tools.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <limits>
#include <algorithm>
#include <omp.h>

namespace doppia {

using namespace boost;


std::string IntegralChannelsComparisonTestApplication::get_application_title()
{
    return "integral_channels_comparison_test_application";
}


IntegralChannelsComparisonTestApplication::IntegralChannelsComparisonTestApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}


IntegralChannelsComparisonTestApplication::~IntegralChannelsComparisonTestApplication()
{
    // nothing to do here
    return;
}


/// helper method used by the user interfaces when recording screenshots
/// this number is expected to change with that same frequency that update_gui is called
int IntegralChannelsComparisonTestApplication::get_current_frame_number() const
{
    // no need to provide true numbers
    return 0;
}


void IntegralChannelsComparisonTestApplication::get_all_options_descriptions(program_options::options_description &desc)
{

    desc.add(BaseApplication::get_options_description(get_application_title()));

    //desc.add(ObjectsDetectionApplication::get_args_options());
    //desc.add(ObjectsDetectionGui::get_args_options());
    //desc.add(VideoInputFactory::get_args_options());

    // Subset of ObjectsDetectionApplication
    desc.add_options()
            ("process_folder",
             program_options::value<string>(),
             "for evaluation purposes, will process images on a folder. Normal video input will be ignored")
            ;

    return;
}


void IntegralChannelsComparisonTestApplication::setup_problem(const program_options::variables_map &options)
{
    const filesystem::path folder_to_process = get_option_value<string>(options, "process_folder");
    directory_input_p.reset(new ImagesFromDirectory(folder_to_process));

    cpu_integral_channels_computer_p.reset(new IntegralChannelsForPedestrians()); // options));
    gpu_integral_channels_computer_p.reset( new GpuIntegralChannelsForPedestrians()); // options));

    return;
}




AbstractGui* IntegralChannelsComparisonTestApplication::create_gui(const program_options::variables_map &/*options*/)
{
    // no gui
    return NULL;
}

void IntegralChannelsComparisonTestApplication::main_loop()
{

    int num_iterations = 0;
    const int num_iterations_for_timing = 10;
    double cumulated_processing_time = 0;
    double start_wall_time = omp_get_wtime();

    bool video_input_is_available = false;
    video_input_is_available = directory_input_p->next_frame();

    // for each input image
    while(video_input_is_available)
    {
        // update video input --
        const AbstractVideoInput::input_image_view_t &
                input_view = directory_input_p->get_image();

        const double start_processing_wall_time = omp_get_wtime();

        printf("Processing image %s\n", directory_input_p->get_image_name().c_str());

        // compute the channel responses and comparison statistics --
        process_frame(input_view);

        cumulated_processing_time += omp_get_wtime() - start_processing_wall_time;

        // print timing information --
        num_iterations += 1;
        if((num_iterations % num_iterations_for_timing) == 0)
        {
            printf("Average iteration speed  %.4lf [Hz] (in the last %i iterations)\n",
                   num_iterations_for_timing / (omp_get_wtime() - start_wall_time) , num_iterations_for_timing );
            start_wall_time = omp_get_wtime(); // we reset timer
        }

        // retrieve the next test image --
        video_input_is_available = directory_input_p->next_frame();

    } // end of "while video input"

    // print global timing information --
    printf("Processed a total of %i input frames\n", num_iterations);

    if(cumulated_processing_time > 0)
    {
        printf("Average speed per iteration %.2lf [Hz] (in the last %i iterations)\n",
               num_iterations / cumulated_processing_time , num_iterations );
    }

    // if we reach this point, everything went well --
    printf("IntegralChannelsComparisonTestApplication test passed\n");

    return;
}


void save_channels_to_file(const GpuIntegralChannelsForPedestrians::channels_t &channels,
                           const string file_path)
{

    const size_t num_channels = channels.shape()[0];
    const size_t channel_size_x = channels.shape()[2], channel_size_y = channels.shape()[1];

    gil::rgb8_image_t channels_image(channel_size_x*num_channels, channel_size_y);
    gil::rgb8_view_t channels_image_view = gil::view(channels_image);

    for(size_t i=0; i < num_channels; i += 1 )
    {
        GpuIntegralChannelsForPedestrians::channels_t::const_reference channel = channels[i];

        gil::rgb8_view_t channel_view =
                gil::subimage_view(channels_image_view,
                                   channel_size_x*i, 0, channel_size_x, channel_size_y);

        // reconstruct "non integral image" from integral image
        Eigen::MatrixXf channel_matrix = Eigen::MatrixXf::Zero(channel_size_y, channel_size_x);

        for(size_t y=0; y < channel_size_y; y+=1)
        {
            for(size_t x=0; x < channel_size_x; x+=1)
            {
                channel_matrix(y,x) = channel[y][x];
            } // end of "for each column"
        } // end of "for each row"

        // copy matrix to overall image
        draw_matrix(channel_matrix, channel_view);
    } // end of "for each channel image"

    gil::png_write_view(file_path, gil::const_view(channels_image));
    return;
}


void IntegralChannelsComparisonTestApplication::process_frame(const AbstractVideoInput::input_image_view_t &input_view)
{

    cpu_integral_channels_computer_p->set_image(input_view);
    gpu_integral_channels_computer_p->set_image(input_view);

    const int
        #if defined(DEBUG)
            num_iterations_for_cpu_timing = 1,
        #else
            num_iterations_for_cpu_timing = 100,
        #endif
            num_iterations_for_gpu_timing = 25*num_iterations_for_cpu_timing;
            //num_iterations_for_gpu_timing = 5*num_iterations_for_cpu_timing;
    double cpu_time, gpu_time;

    {
        printf("Computing CPU integral channels.\n");
        const double start_wall_time = omp_get_wtime();
        for(int i=0; i < num_iterations_for_cpu_timing; i+=1)
        {
            cpu_integral_channels_computer_p->compute();
        }
        cpu_time = omp_get_wtime() - start_wall_time;
    }

    {
        printf("Computing GPU integral channels.\n");
        const double start_wall_time = omp_get_wtime();
        for(int i=0; i < num_iterations_for_gpu_timing; i+=1)
        {
            gpu_integral_channels_computer_p->compute();
        }
        gpu_time = omp_get_wtime() - start_wall_time;
    }


    printf("Cpu speed %.4lf [Hz] / Gpu speed %.4lf [Hz] (compute call, averaged over %i/%i iterations)\n",
           num_iterations_for_cpu_timing / cpu_time,
           num_iterations_for_gpu_timing / gpu_time,
           num_iterations_for_cpu_timing, num_iterations_for_gpu_timing );

    printf("Comparing integral channels.\n");
    typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;
    const integral_channels_t
            &cpu_integral_channels = cpu_integral_channels_computer_p->get_integral_channels(),
            &gpu_integral_channels = gpu_integral_channels_computer_p->get_integral_channels();

    BOOST_REQUIRE(cpu_integral_channels.num_dimensions() == 3);
    BOOST_REQUIRE(cpu_integral_channels.num_dimensions() == gpu_integral_channels.num_dimensions());
    BOOST_REQUIRE(cpu_integral_channels.shape()[0] == gpu_integral_channels.shape()[0]);
    BOOST_REQUIRE(cpu_integral_channels.shape()[1] == gpu_integral_channels.shape()[1]);
    BOOST_REQUIRE(cpu_integral_channels.shape()[2] == gpu_integral_channels.shape()[2]);

    const size_t magnitude_channel_index = 6;
    float
            max_magnitude_diff = -std::numeric_limits<float>::max(),
            min_magnitude_diff = std::numeric_limits<float>::max();

    Eigen::MatrixXf cpu_channel, gpu_channel;

    bool found_one_difference = false;
    for(size_t channel_index = 0; channel_index < cpu_integral_channels.shape()[0]; channel_index += 1)
    {
        // reconstruct "non integral image" from integral image
        get_channel_matrix(cpu_integral_channels, channel_index, cpu_channel);
        get_channel_matrix(gpu_integral_channels, channel_index, gpu_channel);

        BOOST_REQUIRE(cpu_channel.cols() == gpu_channel.cols());
        BOOST_REQUIRE(cpu_channel.rows() == gpu_channel.rows());

        bool stop_this_channel = false;
        for(size_t row_index = 0;
            (row_index < static_cast<size_t>(cpu_channel.rows())) and (not stop_this_channel);
            row_index +=1 )
        {
            for(size_t col_index = 0;
                (col_index < static_cast<size_t>(cpu_channel.cols())) and (not stop_this_channel);
                col_index +=1 )
            {
                const float
                        &cpu_value = cpu_channel(row_index, col_index),
                        &gpu_value = gpu_channel(row_index, col_index);

                if(cpu_value != gpu_value)
                {
                    if(not found_one_difference)
                    {
                        printf("At (channnel, row, col) == (%zi, %zi, %zi), "
                               "cpu_value == %.3f, gpu_value == %.3f\n",
                               channel_index, row_index, col_index,
                               cpu_value, gpu_value);

                        found_one_difference = true;
                        stop_this_channel = true;
                    }


                    if(channel_index == magnitude_channel_index)
                    {
                        //float diff = cpu_value/32; // HERE IS THE PROBLEM
                        float diff = cpu_value; // HERE IS THE PROBLEM
                        diff -= gpu_value;
                        min_magnitude_diff = std::min(diff, min_magnitude_diff);
                        max_magnitude_diff = std::max(diff, max_magnitude_diff);

                        if(diff > 1000)
                        {
                            printf("At (channnel, row, col) == (%zi, %zi, %zi), "
                                   "cpu_value == %.3f, gpu_value == %.3f, diff == %.3f\n",
                                   channel_index, row_index, col_index,
                                   cpu_value, gpu_value, diff);
                        }
                    }

                }

                if(stop_this_channel)
                {
                    printf("Skipping the rest of channel %zi.\n", channel_index);
                }

            } // end of "for each column"
        } // end of "for each row"
    } // end of "for each channel"


    if(found_one_difference)
    {
        save_integral_channels_to_file(cpu_integral_channels, "cpu_integral_channels.png");
        save_integral_channels_to_file(gpu_integral_channels, "gpu_integral_channels.png");
        save_channels_to_file(gpu_integral_channels_computer_p->get_channels(), "gpu_channels.png");

        cv::Mat
                gpu_result = cv::imread("gpu_integral_channels.png"),
                cpu_result = cv::imread("cpu_integral_channels.png"),
                diff_result, diff_result_normalized;
        cv::absdiff(gpu_result, cpu_result, diff_result);
        {
            double min_diff, max_diff;
            cv::minMaxLoc(diff_result, &min_diff, &max_diff);
            printf("min/max gpu/cpu diff value (from image) == %.3f, %.3f\n", min_diff, max_diff);
            printf("min/max gpu/cpu diff value (from magnitude channel) == %.3f, %.3f\n", min_magnitude_diff, max_magnitude_diff);

        }

        cv::normalize(diff_result, diff_result_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
        cv::imwrite("integral_channels_diff.png", diff_result_normalized);

        printf("Created cpu/gpu_integral_channels.png, gpu_channels.png and integral_channels_diff.png "
               "to visualize the difference\n");
        BOOST_FAIL("the gpu and cpu integral channels are not identical");
    }

    printf("\n");

    return;
}



} // end of namespace doppia
