#ifndef METRICCAMERA_HPP
#define METRICCAMERA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/scoped_ptr.hpp>

namespace doppia {

// forward declarations
class CameraCalibration;
class GroundPlane;
class Pose;

/// Utility object that allows to do multiple kinds of geometric
/// transformations based on the camera calibration
class MetricCamera
{
public:

    typedef Eigen::Matrix<float, 3,3> RotationMatrix;
    typedef Eigen::Matrix<float, 3, 4> ProjectionMatrix;
    typedef Eigen::ParametrizedLine<float, 3> Line3d;
    typedef Eigen::Hyperplane<float, 3> Plane3d;


    MetricCamera(const CameraCalibration &calibration);

    /// Build a metric camera using a specific pose (different from the one set during calibration)
    MetricCamera(const CameraCalibration &calibration, const Pose &camera_pose);

    ~MetricCamera();

    const CameraCalibration &get_calibration() const;

    /// Project a 3d point to the image plane
    /// @note the input point should be in the same reference frame than the
    /// camera.get_pose() vector, <em>not in the camera reference frame</em>.
    /// This is specially important when using the right camera of a stereo setup
    Eigen::Vector2f project_3d_point(const Eigen::Vector3f &point) const;

    /// Project a 3d point on a ground plane into the corresponding 2d point in the image plane
    /// The x,y input coordinates are defined over the plane and
    /// the z axis moves along the plane normal.
    /// y moves toward the front of the camera.
    /// In that sense the z axis corresponds to the height over the plane
    /// x,y and height are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const float x, const float y, const float height) const;

    /// x,y and height are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const Eigen::Vector3f &point_on_ground_coordinates) const;

    /// We assume z == 0
    /// x and y are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const float x, const float y) const;

    /// from image plane 2d to camera coordinates 3d
    Eigen::Vector3f back_project_2d_point_to_3d( const Eigen::Vector2f& point_2d, const float depth) const;

    /// from image plane 2d to ray in 3d
    Line3d back_project_2d_point_to_3d_ray(const Eigen::Vector2f& point_2d) const;


    /// see http://en.wikipedia.org/wiki/Pinhole_camera_model for a definition of principal plane
    Plane3d get_principal_plane() const;


    /// convert a point relative to the ground to a point relative to the camera reference frame
    Eigen::Vector3f from_ground_to_3d(const GroundPlane &ground_plane,
                                      const float x, const float y, const float height) const;

    /// convert a point relative to the ground to a point relative to the camera reference frame
    /// x,y and height are in [meters]
    Eigen::Vector3f from_ground_to_3d(const GroundPlane &ground_plane,
                                      const Eigen::Vector3f &point_on_ground_coordinates) const;

    /// convert a point relative to the ground to a point relative to the camera reference frame
    /// We assume z == 0
    /// x and y are in [meters]
    Eigen::Vector3f from_ground_to_3d(const GroundPlane &ground_plane,
                                      const float x, const float y) const;


protected:

    void compute_helper_matrices();

    /// will be set only if the MetricCamera has its own pose
    boost::scoped_ptr<CameraCalibration> own_camera_calibration_p;

    const CameraCalibration &calibration;

    Eigen::Vector3f up_axis, forward_axis, left_axis;
    RotationMatrix transposed_R;
    Eigen::Vector3f camera_focal_point;

    /// Pre-multiplied rotation matrix
    RotationMatrix KR;
    RotationMatrix KR_inverse;

    /// Pre-multiplied translation vector
    Eigen::Vector3f Kt;

    /// Projection matrix
    ProjectionMatrix P;
};

} // end of namespace doppia

#endif // METRICCAMERA_HPP
