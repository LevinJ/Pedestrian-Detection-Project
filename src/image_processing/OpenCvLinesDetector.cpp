#include "OpenCvLinesDetector.hpp"

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <stdexcept>

namespace doppia {

using namespace std;

OpenCvLinesDetector::OpenCvLinesDetector()
    :
      input_image_threshold(150),
      direction_resolution((M_PI/180)*1),
      origin_resolution(1),
      detection_threshold(50)
{
    // nothing to do here
    return;
}

OpenCvLinesDetector::OpenCvLinesDetector(const float input_image_threshold_,
                                           const float direction_resolution_, const float origin_resolution_,
                                           const int detection_threshold_)
    : input_image_threshold(input_image_threshold_),
      direction_resolution(direction_resolution_),
      origin_resolution(origin_resolution_),
      detection_threshold(detection_threshold_)
{
    // nothing to do here
    return;
}

OpenCvLinesDetector::~OpenCvLinesDetector()
{
    // nothing to do here
    return;
}


void OpenCvLinesDetector::operator()(const source_view_t &src, lines_t &lines)
{
    using namespace boost::gil;

    // convert to opencv format --
    opencv::ipl_image_wrapper src_img = opencv::create_ipl_image(src);

    cv::Mat input_image(src_img.get());
    const double threshold_value = input_image_threshold; // 0 to 255
    const double max_value = 255;
    const int threshold_type = cv::THRESH_TOZERO; // any value bellow the threshold_value is set to zero
    // see http://opencv.willowgarage.com/documentation/cpp/miscellaneous_image_transformations.html#cv-threshold
    cv::threshold(input_image, input_image, threshold_value, max_value, threshold_type);

    // call opencv --
    vector<cv::Vec2f> found_lines;
    {

        const float rho_scale = 1;
        const float theta_scale = 1;

        // rho: distance resolution of the accumulator in pixels
        const double rho = origin_resolution*rho_scale;
        // theta: angle resolution of the accumulator in radians
        const double theta = direction_resolution*theta_scale;
        // threshold: the accumulator threshold parameter. Only those lines are returned that get enough votes
        const int threshold = detection_threshold;

        //cv::HoughLines(input_image, found_lines, rho, theta, threshold, rho_scale, theta_scale);
        cv::HoughLines(input_image, found_lines, rho, theta, threshold);
    }


    // convert to output format --
    lines.clear();
    for(std::size_t i=0; i < found_lines.size(); i+=1)
    {
        line_t t_line;
        const float rho = found_lines[i][0];
        const float theta = found_lines[i][1];

        double sin_theta, cos_theta;
        sincos(theta, &sin_theta, &cos_theta);

        if(sin_theta == 0)
        {
            // we simply skip this value (is this such a good idea ?)
            continue;
            //throw std::runtime_error("OpenCvLinesDetector::operator division by zero");
        }

        // see http://en.wikipedia.org/wiki/Hough_transform#Theory
        t_line.direction()(0) = - cos_theta / sin_theta;
        t_line.origin()(0) = rho / sin_theta;

        lines.push_back(t_line);
    }


    const bool show_found_lines = false;
    if(show_found_lines)
    {
        cv::Mat src_img_mat(src_img.get()), color_image;
        cv::cvtColor(src_img_mat, color_image, CV_GRAY2RGB);
        const size_t max_lines_to_draw = 1; //5; // 20

        for(size_t i = 0; i < std::min(found_lines.size(), max_lines_to_draw); i+=1)
        {

            CvPoint pt1, pt2;
            const bool draw_final_lines = true;
            if(draw_final_lines)
            {
                const line_t &t_line = lines.at(i);
                pt1.x = 0;
                pt1.y = t_line.origin()(0);
                pt2.x = 1000;
                pt2.y = pt1.y + pt2.x * t_line.direction()(0);
            }
            else
            {
                // use opencv lines instead of final lines
                const cv::Vec2f &t_line = found_lines.at(i);
                const float rho = t_line[0];
                const float theta = t_line[1];

                const double cos_theta = cos(theta), sin_theta = sin(theta);
                const double x0 = cos_theta*rho, y0 = sin_theta*rho;
                pt1.x = cvRound(x0 + 1000*(-sin_theta));
                pt1.y = cvRound(y0 + 1000*(cos_theta));
                pt2.x = cvRound(x0 - 1000*(-sin_theta));
                pt2.y = cvRound(y0 - 1000*(cos_theta));
            }

            const int thickness = 1;
            cv::line( color_image, pt1, pt2, CV_RGB(255,0,0), thickness, CV_AA);
        } // end of "for each found line"

        cv::imshow("OpenCvLinesDetector", color_image);
        cv::waitKey(0); // force a drawing
    } // end of "if show_found_lines"


    return;
}



} // namespace doppia
