#include "MetricCamera.hpp"

#include "video_input/calibration/CameraCalibration.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"

#include "helpers/xyz_indices.hpp"

#include <cstdio>
#include <iostream>

namespace doppia {


MetricCamera::MetricCamera(const CameraCalibration &calibration_, const Pose &camera_pose)
    : own_camera_calibration_p(new CameraCalibration(calibration_)),
      calibration(*own_camera_calibration_p)
{
    // we set the new pose
    own_camera_calibration_p->get_pose() = camera_pose;
    compute_helper_matrices();
    return;
}


MetricCamera::MetricCamera(const CameraCalibration &calibration_)
    : calibration(calibration_)
{
    compute_helper_matrices();
    return;
}


void MetricCamera::compute_helper_matrices()
{
    // precompute some useful definitions -
    const Pose &camera_pose = calibration.get_pose();

    transposed_R = camera_pose.rotation.transpose();
    up_axis = transposed_R * Eigen::Vector3f(0, 1, 0);
    forward_axis = transposed_R * Eigen::Vector3f(0, 0, 1);
    left_axis = transposed_R * Eigen::Vector3f(-1, 0, 0);

    camera_focal_point = -(transposed_R * camera_pose.translation);


    const InternalCalibrationMatrix &K = calibration.get_internal_calibration();
    // pre-multiplied K * rotation and K*translation
    KR = K * camera_pose.R;
    Kt = K * camera_pose.t;

    KR_inverse = KR.inverse();

    ProjectionMatrix Rt;
    Rt = ProjectionMatrix::Zero();
    Rt.topLeftCorner<3, 3>() = get_calibration().get_pose().rotation;
    Rt.col(3) = get_calibration().get_pose().translation;

    P = K*Rt; // set P the projection matrix
    return;
}

MetricCamera::~MetricCamera()
{
    // nothing to do here
    return;
}


const CameraCalibration &MetricCamera::get_calibration() const
{
    return calibration;
}


Eigen::Vector2f MetricCamera::project_3d_point(const Eigen::Vector3f &point3d) const
{
    const Eigen::Vector3f t = KR * point3d + Kt;

    return Eigen::Vector2f( t(i_x)/t(i_z), t(i_y)/t(i_z) );
}


Eigen::Vector3f MetricCamera::back_project_2d_point_to_3d( const Eigen::Vector2f& point_2d, const float depth ) const
{
    Eigen::Vector3f homogeneous_point_2d( point_2d( i_x ) * depth, point_2d( i_y ) * depth, depth );
    //    Eigen::Vector3f point_3d = KR.inverse() * ( homogeneous_point_2d - Kt );
    Eigen::Vector3f point_3d = KR_inverse * ( homogeneous_point_2d - Kt );

    const float z = point_3d( i_z );

    point_3d = point_3d * (depth / z); // FIXME is this last step correct ?

    return point_3d;
}


MetricCamera::Line3d MetricCamera::back_project_2d_point_to_3d_ray(const Eigen::Vector2f &point_2d) const
{
    // See Multiple view geometry, p. 148
    const Eigen::Vector3f &camera_center = get_calibration().get_pose().translation;
    const InternalCalibrationMatrix &K = get_calibration().get_internal_calibration();
    const Eigen::Vector3f point_3d_on_image_plane = transposed_R*(K.inverse() * point_2d.homogeneous());

    return Line3d::Through(camera_center, point_3d_on_image_plane);
}


MetricCamera::Plane3d MetricCamera::get_principal_plane() const
{
    // see http://en.wikipedia.org/wiki/Pinhole_camera_model for a definition of principal plane
    const Eigen::Vector3f principal_plane_normal(P(2, 0), P(2, 1), P(2, 2));
    const float principal_plane_offset = P(2,3);

    return Plane3d(principal_plane_normal, principal_plane_offset);
}




Eigen::Vector3f MetricCamera::from_ground_to_3d(const GroundPlane &ground_plane,
                                                const float x, const float y, const float height) const
{
    //Eigen::Vector3f xyz_point = camera_focal_point + forward_axis*y + left_axis*x;
    //const Eigen::Vector3f closest_point_on_plane = ground_plane.projection(xyz_point);
    //xyz_point = closest_point_on_plane + ground_plane.normal()*height;

    if(false)
    { // mini test

        const float d1 = ground_plane.signedDistance(camera_focal_point);
        const Eigen::Vector3f xyz_point = camera_focal_point + forward_axis*10; // 10 meters foward
        const float d2 = ground_plane.signedDistance(xyz_point);

        //printf("d@0 == %.3f, d@10 == %.3f, d@10 <= d@0 %s\n", d1, d2, (d2 <= d1)? "true": "false");
        printf("camera looking %s\n", (d2 <= d1)? "down": "up");
        //assert(d2 <= d1);
    }

    const Eigen::Vector3f plane_zero_zero = ground_plane.projection(camera_focal_point);
    const Eigen::Vector3f &ground_left_axis = left_axis;

    const Eigen::Vector3f ground_plane_normal = ground_plane.normal();
    const Eigen::AngleAxis<float> aa(M_PI/2, left_axis);
    const Eigen::Vector3f ground_forward_axis =  aa * ground_plane_normal;

    const Eigen::Vector3f object_bottom = plane_zero_zero + ground_forward_axis*y + ground_left_axis*x;
    const Eigen::Vector3f xyz_point = object_bottom + ground_plane.normal()*height;

    return xyz_point;
}

Eigen::Vector3f MetricCamera::from_ground_to_3d(const GroundPlane &ground_plane,
                                                const Eigen::Vector3f &point_on_ground_coordinates) const
{
    return from_ground_to_3d(ground_plane,
                             point_on_ground_coordinates(i_x),
                             point_on_ground_coordinates(i_y),
                             point_on_ground_coordinates(i_z));
}

Eigen::Vector3f MetricCamera::from_ground_to_3d(const GroundPlane &ground_plane, const float x, const float y) const
{
    return from_ground_to_3d(ground_plane, x, y, 0);
}


/// x,y and height are in [meters]
Eigen::Vector2f MetricCamera::project_ground_plane_point(const GroundPlane &ground_plane,
                                                         const Eigen::Vector3f &point_on_ground_coordinates) const
{
    return project_ground_plane_point(ground_plane,
                                      point_on_ground_coordinates(i_x),
                                      point_on_ground_coordinates(i_y),
                                      point_on_ground_coordinates(i_z));
}


/// x,y and height are in [meters]
Eigen::Vector2f MetricCamera::project_ground_plane_point(const GroundPlane &ground_plane,
                                                         const float x, const float y)  const
{
    return project_ground_plane_point(ground_plane, x, y, 0);
}




/// x,y and height are in [meters]
Eigen::Vector2f MetricCamera::project_ground_plane_point(
        const GroundPlane &ground_plane,
        const float x, const float y, const float height)  const
{
    const Eigen::Vector3f xyz_point = from_ground_to_3d(ground_plane, x, y, height);
    const Eigen::Vector2f uv_point = project_3d_point(xyz_point);
    return uv_point;
}



} // end of namespace doppia
