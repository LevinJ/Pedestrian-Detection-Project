#include "FastReverseMapper.hpp"

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <cstdio>

namespace doppia {

FastReverseMapper::FastReverseMapper(const dimensions_t &dimensions, const shared_ptr<const CameraCalibration> &calibration_p)
    : ReverseMapper(dimensions, calibration_p)
{

    slow_lookup_table_x.create(input_dimensions.y, input_dimensions.x, CV_32F);
    slow_lookup_table_y.create(input_dimensions.y, input_dimensions.x, CV_32F);


    return;
}

FastReverseMapper::~FastReverseMapper()
{
    // nothing to do here
    return;
}

void FastReverseMapper::add_undistortion()
{
    ReverseMapper::add_undistortion();

    update_fast_lookup_table();
    return;
}


void FastReverseMapper::add_homography_projection(const HomographyMatrix &H, const bool H_is_centered)
{
    ReverseMapper::add_homography_projection(H, H_is_centered);

    update_fast_lookup_table();
    return;
}

void FastReverseMapper::warp(const input_image_view_t &input, const output_image_view_t &output) const
{

    // will violate the constness of the view
    boost::gil::opencv::ipl_image_wrapper src_wrapper = boost::gil::opencv::create_ipl_image(input);
    boost::gil::opencv::ipl_image_wrapper t_img_wrapper = boost::gil::opencv::create_ipl_image(t_img_view);


    const bool copy_data = false;
    cv::Mat src(src_wrapper.get(), copy_data);
    cv::Mat t_img_mat(t_img_wrapper.get(), copy_data);
    //cv::Mat t_img_mat;

    const int interpolation = cv::INTER_LINEAR; // cv::INTER_NEAREST
    const int border_mode = cv::BORDER_CONSTANT;

    // warp input to t_img --
    const bool use_fast_lookup = true; // false or true

    if(use_fast_lookup)
    {
        cv::remap(src, t_img_mat,
                  fast_lookup_table_map1, fast_lookup_table_map2, interpolation, border_mode);
    }
    else
    {
        cv::remap(src, t_img_mat,
                  slow_lookup_table_x, slow_lookup_table_y, interpolation, border_mode);
    }

    const bool debug_remap = false;
    if(debug_remap)
    {
        cv::imshow("src", src);
        cv::imshow("t_img", t_img_mat);
        cv::waitKey(10);
    }

    // copy t_img to output --
    copy_pixels(boost::gil::const_view(t_img), output);

    return;
}

const point2<float> &FastReverseMapper::warp(const point2<int> &point) const
{

    return ReverseMapper::warp(point);
}


void FastReverseMapper::update_fast_lookup_table()
{

    // copy lookup_table intp the slow_lookup_table ---
    const lookup_table_t &lookup_table = *lookup_table_p;

    assert(slow_lookup_table_x.cols == input_dimensions.x);
    assert(slow_lookup_table_x.rows == input_dimensions.y);
    assert(slow_lookup_table_x.type() == CV_32F);

    assert(slow_lookup_table_y.cols == input_dimensions.x);
    assert(slow_lookup_table_y.rows == input_dimensions.y);
    assert(slow_lookup_table_y.type() == CV_32F);


    for (int y = 0; y <  input_dimensions.y; y+=1)
    {
        for (int x = 0; x < input_dimensions.x; x+=1)
        {
            const point2<float> &the_point = lookup_table[y][x];

            slow_lookup_table_x.at<float>(y,x) = the_point.x;
            slow_lookup_table_y.at<float>(y,x) = the_point.y;

            if(false and (x == 100 and y == 100))
            {
                printf("FastReverseMapper::update_fast_lookup_table (%i, %i) -> (%.2f, %.2f)\n",
                       x,y,
                       slow_lookup_table_x.at<float>(y,x), slow_lookup_table_y.at<float>(y,x));
            }

        } // end of for each x
    } // end of for each y


    // convert slow_lookup_table into fast_lookup_table ---
    cv::convertMaps(slow_lookup_table_x, slow_lookup_table_y,
                    fast_lookup_table_map1, fast_lookup_table_map2, CV_16SC2);

    assert(fast_lookup_table_map1.cols == input_dimensions.x);
    assert(fast_lookup_table_map1.rows == input_dimensions.y);

    assert(fast_lookup_table_map2.cols == input_dimensions.x);
    assert(fast_lookup_table_map2.rows == input_dimensions.y);

    return;
}

} // end of namespace doppia
