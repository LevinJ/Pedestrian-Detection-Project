#ifndef GROUNDPLANE_HPP
#define GROUNDPLANE_HPP


#include <Eigen/Geometry>

namespace doppia {

/// Stores the ground plane in metric units
class GroundPlane : public Eigen::Hyperplane<float, 3>
{
public:
    /// Eigen::Hyperplane already defines (and documents) almost everything we need
    /// http://eigen.tuxfamily.org/dox-devel/classEigen_1_1Hyperplane.html

    /// Default constructor without initialization
    explicit GroundPlane();

    /// Constructs a plane from its normal \a n and distance to the origin \a d
    /// such that the algebraic equation of the plane is \f$ n \cdot x + d = 0 \f$.
    /// \warning the vector normal is assumed to be normalized.
    ////
    GroundPlane(const VectorType& n, Scalar d);

    /// picth and roll in [radians], height in [meters]
    void set_from_metric_units(const float pitch, const float roll, const float height);

    /// @returns the distance to the ground in [meters]
    const float &get_height() const;
    float &get_height();

    /// we assume no roll
    float get_pitch() const;

};

} // end of namespace doppia

#endif // GROUNDPLANE_HPP
