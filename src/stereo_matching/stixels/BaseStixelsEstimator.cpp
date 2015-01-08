#include "BaseStixelsEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "helpers/Log.hpp"

#include <cstdio>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "BaseStixelsEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "BaseStixelsEstimator");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "BaseStixelsEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "BaseStixelsEstimator");
}

} // end of anonymous namespace



namespace doppia {

using namespace std;

BaseStixelsEstimator::BaseStixelsEstimator(
        const MetricStereoCamera &camera,
        const float expected_object_height_,
        const int minimum_object_height_in_pixels_,
        const int stixel_width_)
    : stereo_camera(camera),
      expected_object_height(expected_object_height_),
      minimum_object_height_in_pixels(minimum_object_height_in_pixels_),
      stixel_width(stixel_width_)
{
    // nothing to do here
    return;
}



BaseStixelsEstimator::~BaseStixelsEstimator()
{
    // nothing to do here
    return;
}


int BaseStixelsEstimator::get_stixel_width() const
{
    return stixel_width;
}


const BaseStixelsEstimator::v_given_disparity_t &BaseStixelsEstimator::get_v_given_disparity() const
{
    return v_given_disparity;
}


const BaseStixelsEstimator::disparity_given_v_t &BaseStixelsEstimator::get_disparity_given_v() const
{
    return disparity_given_v;
}


void BaseStixelsEstimator::set_v_disparity_line_bidirectional_maps(const int num_rows, const int num_disparities)
{
    // cost volume data is organized as y (rows), x (columns), disparity
    const int max_v = num_rows - 1;
    const int max_disparity = num_disparities - 1;

    v_given_disparity.resize(num_disparities);
    disparity_given_v.resize(num_rows);

    const float v_origin = the_v_disparity_ground_line.origin()(0);
    const float direction = the_v_disparity_ground_line.direction()(0);
    const float direction_inverse = 1 / direction;

    const bool print_mapping = false;
    static bool first_print = true;

    for(int d=0; d < num_disparities; d += 1)
    {
        v_given_disparity[d] =
                std::max(0, std::min(max_v, static_cast<int>(
                                         direction*d + v_origin )));

        if(print_mapping and first_print)
        {
            printf("v value at disparity %i == %i\n", d, v_given_disparity[d]);
        }
    } // end of "for each disparity"

    for(int v=0; v < num_rows; v += 1)
    {
        const float d = (v - v_origin) * direction_inverse;
        disparity_given_v[v] =
                std::max(0, std::min(max_disparity, static_cast<int>(d)));

        if(print_mapping and first_print)
        {
            printf("disparity value at v %i == %i\n", v,  disparity_given_v[v]);
            first_print = false;
        }
    } // end of "for each row"

    return;
} // end of StixelsEstimator::set_v_disparity_line_bidirectional_maps


void BaseStixelsEstimator::set_v_given_disparity(const int num_rows, const int num_disparities)
{

    const int minimum_v = 0;
    expected_v_given_disparity.resize(num_disparities, minimum_v);
    top_v_for_stixel_estimation_given_disparity.resize(num_disparities, minimum_v);

    if(expected_object_height <= 0)
    {
        // if zero or negative we use the whole top area of the image
        // thus minimum_v_given_disparity[d] == 0 is good enough
        return;
    }

    const int max_v = num_rows - 1;
    const MetricCamera &left_camera = stereo_camera.get_left_camera();

    for(int d=0; d < num_disparities; d+=1)
    {
        int &expected_v = expected_v_given_disparity[d];
        int &top_v_for_stixel_estimation = top_v_for_stixel_estimation_given_disparity[d];

        const int max_minimum_v = std::max(0, v_given_disparity[d] - minimum_object_height_in_pixels);
        if(d == 0)
        {
            expected_v = max_minimum_v;
            top_v_for_stixel_estimation = max_minimum_v;
        }
        else
        {
            const float depth = stereo_camera.disparity_to_depth(d);

            const float height_for_stixel_estimation = expected_object_height;

            // the "expected top of object"
            {
                const Eigen::Vector2f uv_point =
                        left_camera.project_ground_plane_point(the_ground_plane,
                                                               0, depth,
                                                               expected_object_height);
                expected_v = static_cast<int>(uv_point[1]);
            }

            // the top_v_for_stixel_estimation
            {
                const Eigen::Vector2f uv_point =
                        left_camera.project_ground_plane_point(the_ground_plane,
                                                               0, depth,
                                                               height_for_stixel_estimation);
                top_v_for_stixel_estimation = static_cast<int>(uv_point[1]);
            }

            // check up and lower image bounds
            expected_v = std::min(std::max(0, expected_v), max_v);
            top_v_for_stixel_estimation = std::min(std::max(0, top_v_for_stixel_estimation), max_v);

            // check constraint with respect to v_given_disparity (ground line estimate)
            expected_v = std::min(expected_v, max_minimum_v);
            top_v_for_stixel_estimation = std::min(top_v_for_stixel_estimation, max_minimum_v);


        } // end of "if d == 0"

        // print before assertion failure
        if(false and v_given_disparity[d] <= expected_v)
        {
            printf("At d == %i, v_given_disparity[d] == %i, expected_v == %i and top_v_for_stixel_estimation == %i\n",
                   d, v_given_disparity[d], expected_v, top_v_for_stixel_estimation);
        }

        assert(v_given_disparity[d] > expected_v);
        assert(v_given_disparity[d] > top_v_for_stixel_estimation);

    } // end of "for each disparity value"

    const bool print_details = false;
    if(print_details)
    {
        log_debug() << "expected_v_given_disparity[0] == " <<
                       expected_v_given_disparity[0] << std::endl;
        log_debug() << "expected_v_given_disparity[num_disparities - 1] == " <<
                       expected_v_given_disparity[num_disparities - 1] << std::endl;

        log_debug() << "v height at disparity 0 == " <<
                       (v_given_disparity[0] - expected_v_given_disparity[0]) << std::endl;

        log_debug() << "v height at disparity num_disparities - 1 == " <<
                       (v_given_disparity[num_disparities -1] - expected_v_given_disparity[num_disparities -1]) << std::endl;
    }

    return;
} // end of StixelsEstimator::set_mininum_v_given_disparity


} // end of namespace doppia
