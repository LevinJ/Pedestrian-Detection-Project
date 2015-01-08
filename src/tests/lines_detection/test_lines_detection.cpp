
#define BOOST_TEST_MODULE LinesDetection
#include <boost/test/unit_test.hpp>

/// Small application that tests the LinesDetection class
/// Generate 5 random lines and verify their detection

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view.hpp>

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/format.hpp>

#include <boost/gil/extension/io/png_io.hpp>
#include "image_processing/OpenCvLinesDetector.hpp"
#include "image_processing/IrlsLinesDetector.hpp"

#include <string>
#include <cstdio>

using namespace doppia;


boost::mt19937 random_generator;
boost::uniform_int<> line_value_distribution(10,255);
boost::normal_distribution<> noise_value_distribution;
boost::bernoulli_distribution<> row_selection_distribution(0.3);

boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
line_value_generator(random_generator, line_value_distribution);

boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
noise_value_generator(random_generator, noise_value_distribution);

boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >
row_selection_generator(random_generator, row_selection_distribution);




void set_noise(boost::gil::gray8_view_t &image_view, const float noise_level)
{
    for(int y=0; y < image_view.height(); y+=1)
    {
        for(int x=0; x < image_view.width(); x+=1)
        {
            const bool row_selected = row_selection_generator();

            if(row_selected)
            {
                const int pixel_value = std::abs(noise_level*noise_value_generator());
                image_view.at(x, y)[0] = pixel_value;
            }
        } // end of "for each column"
    } // end of "for each row"

    return;
}

void dummy_draw_line(boost::gil::gray8_view_t &image_view, const AbstractLinesDetector::line_t &line)
{

    for(int x=0; x < image_view.width(); x+=1)
    {
        const float line_y = line.direction()(0) * x + line.origin()(0);
        if(false and x == 100)
        {
            printf("x == %i => y == %.2f\n", x, line_y);
        }
        if(line_y >= 0 and line_y < (image_view.height() - 1))
        {
            const int y = static_cast<int>(line_y);
            image_view.at(x, y)[0] = line_value_generator();
        }
    } // end of "for each x value along the line"

    if(line.direction()(0) != 0)
    {
        for(int y=0; y < image_view.height(); y+=1)
        {
            const float line_x = (y - line.origin()(0)) / line.direction()(0);

            if(false and y == 100)
            {
                printf("y == %i => x == %.2f\n", y, line_x);
            }
            if(line_x >= 0 and line_x < (image_view.width() - 1))
            {
                const int x = static_cast<int>(line_x);
                image_view.at(x, y)[0] = line_value_generator();
            }
        } // end of "for each x value along the line"
    }


    return;
}


bool same_line(const AbstractLinesDetector::line_t &line_a, const AbstractLinesDetector::line_t &line_b,
               const float direction_tolerance, const float origin_tolerance)
{

    return
            std::abs(line_a.direction()(0) - line_b.direction()(0)) < direction_tolerance and
            std::abs(line_a.origin()(0) - line_b.origin()(0)) < origin_tolerance;
}

/// noise_level should be a value between 0-255
void draw_test_lines(boost::gil::gray8_view_t &image_view,
                     const AbstractLinesDetector::lines_t &lines,
                     const float noise_level)
{
    boost::gil::fill_pixels(image_view, 0);

    if(noise_level > 0)
    {
        assert(noise_level <= 255);
        set_noise(image_view, noise_level);
    }

    for(std::size_t i=0; i < lines.size(); i+=1  )
    {
        dummy_draw_line(image_view, lines[i]);
    }


    const bool save_test_image = true;
    if(save_test_image)
    {
        static int id = 0;
        const std::string filename = boost::str( boost::format("lines_detection_test_image_%i.png") % id);
        printf("Created file %s\n", filename.c_str());
        boost::gil::png_write_view(filename, image_view);
        id += 1;
    }

    return;
}


void check_detected_lines(const AbstractLinesDetector::lines_t &source_lines,
                          const AbstractLinesDetector::lines_t &detected_lines)
{

    const float origin_resolution = 1;
    const float direction_resolution = (M_PI/180)*1;
    const float direction_tolerance = direction_resolution*10; // radians
    const float origin_tolerance = origin_resolution*10; // pixels

    printf("direction_tolerance == %.3f, origin_tolerance == %.3f\n",
           direction_tolerance, origin_tolerance);


    printf("source_lines.size() == %zi, detected_lines.size() == %zi\n",
           source_lines.size(), detected_lines.size());
    BOOST_REQUIRE( source_lines.size() <= detected_lines.size() );

    for(std::size_t i=0; i < detected_lines.size(); i+=1  )
    {
        printf("detected_lines[%zi] (direction, origin) == (%.3f, %.3f)\n",
               i, detected_lines[i].direction()(0), detected_lines[i].origin()(0));
    }

    for(std::size_t i=0; i < source_lines.size(); i+=1  )
    {
        printf("source_lines[%zi] (direction, origin) == (%.3f, %.3f)\n",
               i, source_lines[i].direction()(0), source_lines[i].origin()(0));

        const AbstractLinesDetector::lines_t::const_iterator it =
                std::find_if(detected_lines.begin(), detected_lines.end(),
                             boost::bind(&same_line,
                                         source_lines[i], _1,
                                         direction_tolerance, origin_tolerance));

        BOOST_REQUIRE_MESSAGE( it != detected_lines.end(), "did not detect one of the source lines");
    }

    printf("\n\n");
    return;
}

BOOST_AUTO_TEST_CASE(OpenCvLinesDetectionTestCase)
{

    boost::gil::gray8_image_t test_image(500, 500);
    boost::gil::gray8_view_t test_image_view = boost::gil::view(test_image);
    boost::gil::gray8c_view_t test_image_const_view = boost::gil::const_view(test_image);

    AbstractLinesDetector::lines_t source_lines, detected_lines;

    // setup test lines --
    {
        AbstractLinesDetector::line_t t_line;
        t_line.direction()(0) = 0.5;
        t_line.origin()(0) = 250;
        source_lines.push_back(t_line);

        t_line.direction()(0) = 5;
        t_line.origin()(0) = 100;
        source_lines.push_back(t_line);

        t_line.direction()(0) = 0;
        t_line.origin()(0) = 300;
        source_lines.push_back(t_line);
    }

    // draw test lines --
    const float noise_level = 0;
    draw_test_lines(test_image_view, source_lines, noise_level);

    // detects lines --
    const float direction_resolution = (M_PI/180)*1;
    const float origin_resolution = 1;
    const int detection_threshold = 70;

    const int input_image_threshold = 100; // intensity threshold

    {
        boost::scoped_ptr<AbstractLinesDetector> lines_detector_p;
        lines_detector_p.reset(
                    new OpenCvLinesDetector(input_image_threshold,
                                            direction_resolution, origin_resolution, detection_threshold) );

        (*lines_detector_p)(test_image_const_view, detected_lines);
    }

    // check --
    check_detected_lines(source_lines, detected_lines);

    return;
} // end of "BOOST_AUTO_TEST_CASE OpenCvLinesDetectionTestCase"


void irls_lines_detection_test_case(const float noise_level_fraction, const bool use_initial_estimate)
{
    // seed the random generator
    random_generator.seed(std::time(NULL));

    boost::gil::gray8_image_t test_image(500, 500);
    boost::gil::gray8_view_t test_image_view = boost::gil::view(test_image);
    boost::gil::gray8c_view_t test_image_const_view = boost::gil::const_view(test_image);

    AbstractLinesDetector::lines_t source_lines, detected_lines;

    // setup test line --
    // Irls detects only one single line
    AbstractLinesDetector::line_t initial_line_estimate;
    {
        AbstractLinesDetector::line_t t_line;
        t_line.direction()(0) = 0.5;
        t_line.origin()(0) = 250;
        source_lines.push_back(t_line);

        initial_line_estimate = t_line;
        initial_line_estimate.direction()(0) += 0.4;
        initial_line_estimate.origin()(0) -= 50;
    }



    const int max_intensity_value = 200;
    const int num_iterations = 5;
    const float max_tukey_c = 50, min_tukey_c = 15;

    // draw test lines --
    const float noise_level = max_intensity_value*noise_level_fraction;
    draw_test_lines(test_image_view, source_lines, noise_level);

    // detects lines --
    {
        boost::scoped_ptr<IrlsLinesDetector> lines_detector_p;
        lines_detector_p.reset(new IrlsLinesDetector(max_intensity_value, num_iterations,
                                                     max_tukey_c, min_tukey_c) );

        if(use_initial_estimate)
        {
            lines_detector_p->set_initial_estimate(initial_line_estimate);
        }

        (*lines_detector_p)(test_image_const_view, detected_lines);
    }

    // check --
    check_detected_lines(source_lines, detected_lines);

    return;
}

BOOST_AUTO_TEST_CASE(IrlsLinesDetectionNoiseFreeTestCase)
{
    const float noise_level_fraction = 0;
    const bool use_initial_estimate = false;
    irls_lines_detection_test_case(noise_level_fraction, use_initial_estimate);

    return;
} // end of "BOOST_AUTO_TEST_CASE IrlsLinesDetectionNoiseFreeTestCase"

BOOST_AUTO_TEST_CASE(IrlsLinesDetectionNoisePresentTestCase)
{
    // 0.1 is too easy, 0.25 is border line and
    // 0.3 will not be solved by simple least squares
    const float noise_level_fraction = 0.3;
    const bool use_initial_estimate = true;
    irls_lines_detection_test_case(noise_level_fraction, use_initial_estimate);

    return;
} // end of "BOOST_AUTO_TEST_CASE IrlsLinesDetectionNoisePresentTestCase"
