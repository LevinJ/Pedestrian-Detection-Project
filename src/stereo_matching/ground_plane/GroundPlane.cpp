#include "GroundPlane.hpp"

#include "helpers/xyz_indices.hpp"

#include <cmath>

namespace doppia {


GroundPlane::GroundPlane()
    :  Eigen::Hyperplane<float, 3>()
{
    // nothing to do here
    return;
}



GroundPlane::GroundPlane(const GroundPlane::VectorType& n, GroundPlane::Scalar d)
    :  Eigen::Hyperplane<float, 3>(n,d)
{
    // nothing to do here
    return;
}


void GroundPlane::set_from_metric_units(const float pitch, const float roll, const float height)
{
    GroundPlane &the_plane = (*this);

    Eigen::Vector3f normal;

    // ground plane normal should be similar to x = 0, y = -1, z =0;
    normal(i_x) = sin(roll);
    normal(i_z) = sin(pitch);
    // ensure a vector of length 1
    normal(i_y) = -sqrt(1 - (normal(i_x)* normal(i_x) +  normal(i_z)* normal(i_z)));

    the_plane = GroundPlane(normal, height);

    return;
}

const float &GroundPlane::get_height() const
{
    return this->offset();
}

float &GroundPlane::get_height()
{
    return this->offset();
}

float GroundPlane::get_pitch() const
{
    // we assume no roll
    return std::asin(this->normal()(i_z));
}

} // end of namespace doppia
