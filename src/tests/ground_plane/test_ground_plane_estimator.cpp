
#define BOOST_TEST_MODULE GroundPlaneEstimator
#include <boost/test/unit_test.hpp>

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include <string>
#include <cstdio>

using namespace doppia;
using namespace std;

BOOST_AUTO_TEST_CASE(LineAndGroundPlaneConvertionTestCase)
{
    const string stereo_calibration_path = "../../video_input/calibration/stereo_calibration_bahnhof.proto.txt";
    StereoCameraCalibration stereo_calibration(stereo_calibration_path);
    GroundPlaneEstimator ground_plane_estimator(stereo_calibration);

    GroundPlane ground_plane_prior;
    ground_plane_prior.set_from_metric_units(-10*M_PI/180, 0, 1.2);

    const GroundPlane &the_ground_plane = ground_plane_prior;
    const GroundPlaneEstimator::line_t the_line = ground_plane_estimator.ground_plane_to_v_disparity_line(the_ground_plane);
    const GroundPlane new_ground_plane = ground_plane_estimator.v_disparity_line_to_ground_plane(the_line);

    printf("the_ground_plane height == %.4f [meters], pitch == %.4f [degrees]\n",
           the_ground_plane.get_height(), the_ground_plane.get_pitch() * 180/M_PI);

    printf("new_ground_plane height == %.4f [meters], pitch == %.4f [degrees]\n",
           new_ground_plane.get_height(), new_ground_plane.get_pitch() * 180/M_PI);

    BOOST_REQUIRE_MESSAGE( the_ground_plane.isApprox(new_ground_plane),
                          "after converting to line ground plane is not identical to the original. It should be identical.");

    return;
} // end of "BOOST_AUTO_TEST_CASE"

