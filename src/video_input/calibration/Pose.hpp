#ifndef POSE_HPP
#define POSE_HPP


#include <Eigen/Core>

namespace doppia
{

typedef Eigen::Matrix<float, 3, 3> RotationMatrix;
typedef Eigen::Vector3f TranslationVector;


/// R and t are such that X_camera = R * X_world + t
class Pose
{
public:
    Pose(const RotationMatrix &R, const TranslationVector &t);

    /// copy constructor
    Pose(const Pose &pose);
    ~Pose();

    Pose &operator=(const Pose &pose);

    /// Global rotation (world -> camera)
    RotationMatrix rotation;
    RotationMatrix &R; // alias reference

    /// Global translation (world -> camera)
    TranslationVector translation;
    TranslationVector &t; // alias reference

};

} // end of namespace doppia

#endif // POSE_HPP
