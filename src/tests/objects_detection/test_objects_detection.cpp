
/// Helper application that tests the FastestPedestrianDetectorInTheWest with respect to IntegralChannelsDetector

#define BOOST_TEST_MODULE ObjectsDetection
#include <boost/test/unit_test.hpp>

#include "DetectorsComparisonTestApplication.hpp"
#include "IntegralChannelsComparisonTestApplication.hpp"
#include "VeryFastDetectorScaleStatisticsApplication.hpp"
#include "image_processing/integrate.hpp"
#include "image_processing/fast_rgb_to_luv.hpp"

#include "applications/objects_detection/ObjectsDetectionApplication.hpp"
#include "objects_detection/integral_channels/AngleBinComputer.hpp"

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/cstdint.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <omp.h>

using namespace doppia;
using namespace boost;

mt19937 random_generator;
uniform_int<> uint8_distribution(0,255);

variate_generator<mt19937&, uniform_int<> > pixel_value_generator(random_generator, uint8_distribution);


BOOST_AUTO_TEST_CASE(TestIntegrate)
{
    // create a random image --
    multi_array<uint8_t, 2> random_image(extents[50][80]);

    for(size_t y=0; y < random_image.shape()[0]; y+=1 )
    {
        for(size_t x=0; x < random_image.shape()[1]; x+=1 )
        {
            random_image[y][x] = pixel_value_generator();

        } // end of "for each col"
    } // end of "for each row"

    // compute the integral --
    multi_array<uint32_t, 2> image_integral(extents[random_image.shape()[0] + 1][random_image.shape()[1] + 1]);
    doppia::integrate(random_image, image_integral);

    // compute the integral via opencv --
    gil::gray8c_view_t random_image_view =
            gil::interleaved_view(random_image.shape()[1], random_image.shape()[0],
                                  reinterpret_cast<gil::gray8c_pixel_t *>(random_image.origin()),
                                  random_image.shape()[1]*sizeof(multi_array<uint8_t, 2>::element));


    const gil::opencv::ipl_image_wrapper random_image_ipl = gil::opencv::create_ipl_image(random_image_view);

    const cv::Mat random_image_mat(random_image_ipl.get());
    cv::Mat image_integral_mat;
    cv::integral(random_image_mat, image_integral_mat);

    const gil::gray32c_view_t integral_mat_view =
            gil::interleaved_view(image_integral_mat.cols, image_integral_mat.rows,
                                  reinterpret_cast<gil::gray32c_pixel_t*>(image_integral_mat.data),
                                  static_cast<size_t>(image_integral_mat.step));

    // compare both images --
    BOOST_REQUIRE(static_cast<size_t>(image_integral_mat.rows) == image_integral.shape()[0]);
    BOOST_REQUIRE(static_cast<size_t>(image_integral_mat.cols) == image_integral.shape()[1]);

    for(size_t y=0; y < image_integral.shape()[0]; y+=1 )
    {
        for(size_t x=0; x < image_integral.shape()[1]; x+=1 )
        {
            BOOST_REQUIRE( (*integral_mat_view.at(x,y)) == image_integral[y][x]);
        } // end of "for each col"
    } // end of "for each row"


    printf("TestIntegrate passed. Yey!\n\n");

} // end of "BOOST_AUTO_TEST_CASE TestIntegrate"



BOOST_AUTO_TEST_CASE(FpdwVsChnFtrsTestCase)
{
    // we run both detectors over a set of images, with the same parameters
    // with a non-cascaded detectors, both detectors should obtain similar scores (with a given margin of error).

    char arg0[] = "test_objects_detection";
    char arg1[] = "-c";
    char arg2[] = "fpdw_vs_chnftrs.config.ini";

    char *argv[] = { arg0, arg1, arg2 };
    int argc = sizeof(argv) / sizeof(argv[0]);

    scoped_ptr<AbstractApplication> application_p;
    application_p.reset(new DetectorsComparisonTestApplication());

    application_p->main(argc, argv);

} // end of "BOOST_AUTO_TEST_CASE FpdwVsChnFtrsTestCase"



BOOST_AUTO_TEST_CASE(VeryFastDetectorScaleStatistics)
{
    // we run the very_fast detector and collect statistics across scales (for a single pixel)

    char arg0[] = "test_objects_detection";
    char arg1[] = "-c";
    char arg2[] = "very_fast_detector_scale_statistics.config.ini";

    char *argv[] = { arg0, arg1, arg2 };
    int argc = sizeof(argv) / sizeof(argv[0]);

    scoped_ptr<AbstractApplication> application_p(new VeryFastDetectorScaleStatisticsApplication());

    application_p->main(argc, argv);

} // end of "BOOST_AUTO_TEST_CASE VeryFastDetectorScaleStatistics"


BOOST_AUTO_TEST_CASE(ComputeSoftCascade)
{

    char arg0[] = "test_objects_detection";
    char arg1[] = "-c";
    char arg2[] = "compute_soft_cascade.config.ini";

    char *argv[] = { arg0, arg1, arg2 };
    int argc = sizeof(argv) / sizeof(argv[0]);

    scoped_ptr<ObjectsDetectionApplication> application_p( new ObjectsDetectionApplication());

    application_p->main(argc, argv);

} // end of "BOOST_AUTO_TEST_CASE ComputeSoftCascade"


BOOST_AUTO_TEST_CASE(GpuIntegralChannelsVsCpuIntegralChannelsTestCase)
{
    // we run both detectors over a set of images, with the same parameters
    // with a non-cascaded detectors, both detectors should obtain similar scores (with a given margin of error).

    char arg0[] = "test_objects_detection";
    char arg1[] = "-c";
    char arg2[] = "gpu_vs_cpu_integral_channels.config.ini";

    char *argv[] = { arg0, arg1, arg2 };
    int argc = sizeof(argv) / sizeof(argv[0]);

    scoped_ptr<AbstractApplication> application_p;
    application_p.reset(new IntegralChannelsComparisonTestApplication());

    application_p->main(argc, argv);

} // end of "BOOST_AUTO_TEST_CASE GpuIntegralChannelsVsCpuIntegralChannelsTestCase"


BOOST_AUTO_TEST_CASE(FastRgbToLuvTestCase)
{

    const std::string test_image_path = "rgb_to_luv_test_image.png";
    cv::Mat input_image = cv::imread(test_image_path), opencv_luv, fast_luv;

    const int num_iterations_for_timing = 50;
    double opencv_time, fast_time;
    // compute use opencv
    {
        const double start_wall_time = omp_get_wtime();
        for(int i=0; i < num_iterations_for_timing; i+=1)
        {
            cv::cvtColor(input_image, opencv_luv, CV_RGB2Luv);
        }
        opencv_time = omp_get_wtime() - start_wall_time;
    }

    // compute using our fast code
    {
        fast_luv.create(input_image.rows, input_image.cols, input_image.type());

        const gil::rgb8c_view_t rgb_view =
                gil::interleaved_view(input_image.cols, input_image.rows,
                                      reinterpret_cast<gil::rgb8c_pixel_t*>(input_image.data),
                                      static_cast<size_t>(input_image.step));
        const gil::dev3n8_view_t luv_view =
                gil::interleaved_view(fast_luv.cols, fast_luv.rows,
                                      reinterpret_cast<gil::dev3n8_pixel_t*>(fast_luv.data),
                                      static_cast<size_t>(fast_luv.step));

        const double start_wall_time = omp_get_wtime();
        for(int i=0; i < num_iterations_for_timing; i+=1)
        {
            fast_rgb_to_luv(rgb_view, luv_view);
        }
        fast_time = omp_get_wtime() - start_wall_time;
    }

    printf("OpenCv speed %.4lf [Hz], our code speed %.4lf [Hz] (averaged over %i iterations)\n",
           num_iterations_for_timing / opencv_time,
           num_iterations_for_timing / fast_time,
           num_iterations_for_timing );

    // comparison output
    {
        cv::Mat luv_diff, opencv_luv_in_rgb, fast_luv_in_rgb;
        cv::absdiff(opencv_luv, fast_luv, luv_diff);
        cv::Scalar t_sum = cv::sum(luv_diff);
        const float summed_diff = t_sum[0] + t_sum[1] + t_sum[2] + t_sum[3];
        printf("summed_diff %.3f\n", summed_diff);


        cv::cvtColor(opencv_luv, opencv_luv_in_rgb, CV_Luv2RGB);
        cv::cvtColor(fast_luv, fast_luv_in_rgb, CV_Luv2RGB);

        cv::imwrite("opencv_luv.png", opencv_luv);
        cv::imwrite("fast_luv.png", fast_luv);

        cv::imwrite("opencv_luv_in_rgb.png", opencv_luv_in_rgb);
        cv::imwrite("fast_luv_in_rgb.png", fast_luv_in_rgb);

        printf("Created opencv_luv.png and fast_luv.png to compare visually\n");
    }

} // end of "BOOST_AUTO_TEST_CASE FastRgbToLuvTestCase"


BOOST_AUTO_TEST_CASE(AngleBinComputerTestCase)
{

    const int num_angle_bins = 6;
    //const int num_angle_bins = 2;
    AngleBinComputer<num_angle_bins> angle_bin_computer;
    const float angle_quantum = M_PI/num_angle_bins;

    uniform_real<float> float_distribution(-1000.0f, 1000.0f);
    variate_generator<mt19937&, uniform_real<float> > float_value_generator(random_generator, float_distribution);

    // FIXME something is wrong with the angles binning

    const int num_test_samples = 100000;
    for(int c=0; c < num_test_samples; c+=1)
    {
        const float dx = float_value_generator(), dy = float_value_generator();

        float angle = atan2(dy, dx) + (angle_quantum/2);
        if(angle < 0)
        {
            angle += M_PI; // reflect negative angles
        }
        const int atan2_angle_index = static_cast<int>(angle / angle_quantum) % num_angle_bins;

        const int computed_angle_index = angle_bin_computer(dy, dx);
        if(atan2_angle_index != computed_angle_index)
        {
            printf("Test failed after %i tests, when dx == %.3f and dy == %.3f\n", c, dx, dy);
            printf("atan2_angle_index == %i\n", atan2_angle_index);
            printf("atan2 angle == %.3f°\n", angle*180/M_PI);
            printf("atan2 angle / angle_quantum == %.3f\n", angle / angle_quantum);
            printf("computed_angle_index == %i\n", computed_angle_index);

            for(int i=0; i < num_angle_bins; i+=1)
            {
                float angle = atan2(angle_bin_computer.bin_vectors[i][1], angle_bin_computer.bin_vectors[i][0]);
                printf("angle_bin_computer.bin_vectors[%i] == %.3f°\n", i, angle*180/M_PI);
                const float dot_product = std::abs(dx*angle_bin_computer.bin_vectors[i][0] + dy*angle_bin_computer.bin_vectors[i][1]);
                printf("angle_bin_computer.bin_vectors[%i] dot product == %.3f\n", i, dot_product);
            }
        }

        BOOST_REQUIRE_MESSAGE(atan2_angle_index == computed_angle_index,
                              "AngleBinComputer computed the wrong result");

    } // end of all random tests

} // end of "BOOST_AUTO_TEST_CASE AngleBinComputerTestCase"


BOOST_AUTO_TEST_CASE(GpuResizeTestCase)
{

    typedef cv::gpu::FilterEngine_GPU filter_t;
    typedef cv::Ptr<filter_t> filter_shared_pointer_t;

    // any texture image is good for the test
    //const std::string test_image_path = "rgb_to_luv_test_image.png";
    const std::string test_image_path = "resizing_test_image.png";
    cv::Mat input_rgb_image = cv::imread(test_image_path), input_image;

    cv::gpu::GpuMat input_gpu_image, shrink_buffer, shrunk_image_a, shrunk_image_b;

    // Based on the Nvidia forum answer, nppiResize_8u_C1R would actually do averaging of the input pixels
    // this test verifies if this is true or not
    // http://forums.nvidia.com/index.php?showtopic=210066

    cv::cvtColor(input_rgb_image, input_image, CV_RGB2GRAY);

    BOOST_REQUIRE(input_image.type() == CV_8UC1);

    //input_image.setTo(10); // for baseline check

    input_gpu_image.upload(input_image);
    shrink_buffer.create(input_image.rows, input_image.cols, input_image.type());
    shrunk_image_a.create((((input_image.rows + 1)/2) + 1)/2, (((input_image.cols+1)/2)+1)/2, input_image.type());

    // option A
    {
        filter_shared_pointer_t pyrDown4_smoothing_filter_p =
                cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(4, 4));
        //cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(16, 16));
        //cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(8, 8));

        // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
        pyrDown4_smoothing_filter_p->apply(input_gpu_image, shrink_buffer);
        //shrink_buffer = input_gpu_image; // for comparing INTER_NEAREST to INTER_LINEAR
        cv::gpu::resize(shrink_buffer, shrunk_image_a,
                        shrunk_image_a.size(), 0, 0,
                        cv::INTER_NEAREST);
    }

    // option B
    {
        cv::gpu::resize(input_gpu_image, shrunk_image_b,
                        shrunk_image_a.size(), 0, 0,
                        cv::INTER_LINEAR);
    }

    BOOST_REQUIRE(shrunk_image_a.size() == shrunk_image_b.size());

    // are they both the same ?
    {
        cv::Mat cpu_a, cpu_b, cpu_diff, cpu_diff_cropped, cpu_diff_normalized;
        shrunk_image_a.download(cpu_a);
        shrunk_image_b.download(cpu_b);

        cv::absdiff(cpu_a, cpu_b, cpu_diff);

        cpu_diff_cropped = cpu_diff(cv::Rect(16, 16, cpu_diff.cols - 16*2, cpu_diff.rows - 16*2));

        cv::Scalar total_sum = cv::sum(cpu_diff), cropped_sum = cv::sum(cpu_diff_cropped);
        const float
                summed_diff = total_sum[0] + total_sum[1] + total_sum[2] + total_sum[3],
                summed_diff_cropped = cropped_sum[0] + cropped_sum[1] + cropped_sum[2] + cropped_sum[3];
        printf("summed_diff %.3f\n", summed_diff);
        printf("summed_diff_cropped %.3f\n", summed_diff_cropped);

        if(summed_diff > 0)
        {
            cv::normalize(cpu_diff, cpu_diff_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
            cv::imwrite("resize_diff.png", cpu_diff_normalized);

            cv::normalize(cpu_diff_cropped, cpu_diff_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
            cv::imwrite("resize_diff_cropped.png", cpu_diff_normalized);

            cv::imwrite("resize_a.png", cpu_a);
            cv::imwrite("resize_b.png", cpu_b);

            printf("Created resize_diff.png, resize_diff_cropped.png, resize_a.png and resize_b.png to visualize the differences\n");
        }

        BOOST_CHECK(summed_diff == 0);
        BOOST_REQUIRE(summed_diff_cropped == 0);
    }


} // end of "BOOST_AUTO_TEST_CASE GpuResizeTestCase"




BOOST_AUTO_TEST_CASE(CpuResizeTestCase)
{

    typedef cv::FilterEngine filter_t;
    typedef cv::Ptr<filter_t> filter_shared_pointer_t;

    // any texture image is good for the test
    //const std::string test_image_path = "rgb_to_luv_test_image.png";
    const std::string test_image_path = "resizing_test_image.png";
    cv::Mat input_rgb_image = cv::imread(test_image_path);

    cv::Mat input_image, shrink_buffer, shrunk_image_a, shrunk_image_b;

    // Based on the Nvidia forum answer, nppiResize_8u_C1R would actually do averaging of the input pixels
    // this test verifies if this is true or not
    // http://forums.nvidia.com/index.php?showtopic=210066

    cv::cvtColor(input_rgb_image, input_image, CV_RGB2GRAY);

    BOOST_REQUIRE(input_image.type() == CV_8UC1);

    //input_image.setTo(10); // for baseline check

    shrink_buffer.create(input_image.rows, input_image.cols, input_image.type());
    shrunk_image_a.create((((input_image.rows + 1)/2) + 1)/2, (((input_image.cols+1)/2)+1)/2, input_image.type());

    // option A
    {
        filter_shared_pointer_t pre_resizing_smoothing_filter_p =
                cv::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(4, 4));
        //cv::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(8, 8));
        //cv::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(16, 16));

        // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
        pre_resizing_smoothing_filter_p->apply(input_image, shrink_buffer);
        cv::resize(shrink_buffer, shrunk_image_a,
                   shrunk_image_a.size(), 0, 0,
                   cv::INTER_NEAREST);
    }

    // option B
    {
        cv::resize(input_image, shrunk_image_b,
                   shrunk_image_a.size(), 0, 0,
                   cv::INTER_AREA);
    }

    BOOST_REQUIRE(shrunk_image_a.size() == shrunk_image_b.size());

    // are they both the same ?
    {
        cv::Mat cpu_a, cpu_b, cpu_diff, cpu_diff_cropped, cpu_diff_normalized;
        cpu_a = shrunk_image_a;
        cpu_b = shrunk_image_b;

        cv::absdiff(cpu_a, cpu_b, cpu_diff);

        cpu_diff_cropped = cpu_diff(cv::Rect(16, 16, cpu_diff.cols - 16*2, cpu_diff.rows - 16*2));

        cv::Scalar total_sum = cv::sum(cpu_diff), cropped_sum = cv::sum(cpu_diff_cropped);
        const float
                summed_diff = total_sum[0] + total_sum[1] + total_sum[2] + total_sum[3],
                summed_diff_cropped = cropped_sum[0] + cropped_sum[1] + cropped_sum[2] + cropped_sum[3];
        printf("summed_diff %.3f\n", summed_diff);
        printf("summed_diff_cropped %.3f\n", summed_diff_cropped);

        if(summed_diff > 0)
        {
            cv::normalize(cpu_diff, cpu_diff_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
            cv::imwrite("resize_diff.png", cpu_diff_normalized);

            cv::normalize(cpu_diff_cropped, cpu_diff_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
            cv::imwrite("resize_diff_cropped.png", cpu_diff_normalized);

            cv::imwrite("resize_a.png", cpu_a);
            cv::imwrite("resize_b.png", cpu_b);

            printf("Created resize_diff.png, resize_diff_cropped.png, resize_a.png and resize_b.png to visualize the differences\n");
        }

        BOOST_CHECK(summed_diff == 0);
        BOOST_REQUIRE(summed_diff_cropped == 0);
    }


} // end of "BOOST_AUTO_TEST_CASE CpuResizeTestCase"




BOOST_AUTO_TEST_CASE(GpuVsCpuResizeTestCase)
{
    // This test is critical.
    // Our experiment have shown that the classification performance is _very_ sensitive to the resizing methods.
    // As such _every_ image resizing (in any python or C++ program) should be done
    // in exactly the same way to ensure top performance.

    // GpuIntegralChannelsDetector::resize_input_and_compute_integral_channels
    // is the method that "dictates" what should be used everywhere.
    // Current version uses
    // cv::gpu::resize(input_gpu_mat, resized_input_gpu_mat, scaled_size);
    // I assume we use "vanilla resize" for performance reasons.
    // (although when resizing images, we are already dropping speed significantly)
    // Seems like a reasonable "let us keep it simple" choice.


    const std::string test_image_path = "resizing_test_image.png";
    cv::Mat input_rgb_image = cv::imread(test_image_path);

    cv::gpu::GpuMat input_gpu_mat, resized_input_gpu_mat;

    cv::Mat cpu_resized_image, gpu_resized_image;

    float
            image_width = input_rgb_image.cols,
            image_height = input_rgb_image.rows;

    std::vector<float> scaling_factors;
    scaling_factors.push_back(0.3);
    scaling_factors.push_back(0.5);
    scaling_factors.push_back(1.0);
    scaling_factors.push_back(1.3);
    scaling_factors.push_back(1.5);
    scaling_factors.push_back(2);
    scaling_factors.push_back(3);

    typedef boost::tuple<float, cv::Size> factor_and_size_t;
    std::vector<factor_and_size_t> factor_and_sizes;

    BOOST_FOREACH(const float factor, scaling_factors)
    {
        factor_and_sizes.push_back(
                    boost::make_tuple(factor,
                                      cv::Size(image_width * factor,
                                               image_height * factor))
                    );
    } // end of "for each scaling factor"


    BOOST_FOREACH(const factor_and_size_t &factor_and_size, factor_and_sizes)
    {

        const float scaling_factor = factor_and_size.get<0>();
        const cv::Size &destination_size = factor_and_size.get<1>();

        // resize via cpu --
        cv::resize(input_rgb_image, cpu_resized_image, destination_size);

        // resize via gpu --
        input_gpu_mat.upload(input_rgb_image);
        cv::gpu::resize(input_gpu_mat, resized_input_gpu_mat, destination_size);
        resized_input_gpu_mat.download(gpu_resized_image);


        // are they both the same ? --
        {
            cv::Mat cpu_gpu_diff, cpu_gpu_diff_normalized;

            cv::absdiff(cpu_resized_image, gpu_resized_image, cpu_gpu_diff);

            cv::Scalar total_sum = cv::sum(cpu_gpu_diff);

            // cv::Scalar has at most 4 elements, all initialized to zero
            const float
                    summed_diff = total_sum[0] + total_sum[1] + total_sum[2] + total_sum[3];

            if(summed_diff > 0)
            {
                printf("Resizing towards (%i, %i, factor %.3f) failed. summed_diff == %.3f\n",
                       destination_size.width, destination_size.height, scaling_factor, summed_diff);

                if(false)
                {
                    cv::normalize(cpu_gpu_diff, cpu_gpu_diff_normalized, 255, 0, cv::NORM_MINMAX, CV_32FC1);
                    cv::imwrite("resize_diff.png", cpu_gpu_diff_normalized);

                    cv::imwrite("resize_cpu.png", cpu_resized_image);
                    cv::imwrite("resize_gpu.png", gpu_resized_image);

                    printf("Created resize_diff.png, resize_cpu.png and resize_gpu.png to visualize the differences\n");
                }
            }
            else
            {
                printf("Resizing towards (%i, %i, factor %.3f) passed\n",
                       destination_size.width, destination_size.height, scaling_factor);
            }
            BOOST_CHECK(summed_diff == 0);
            //BOOST_REQUIRE(summed_diff == 0);
        }

    } // end of "for each destination size"


} // end of "BOOST_AUTO_TEST_CASE GpuVsCpuResizeTestCase"
