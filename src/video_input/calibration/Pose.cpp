#include "Pose.hpp"


namespace doppia {

Pose::Pose(const RotationMatrix &_R, const TranslationVector &_t):
    rotation(_R), R(rotation), translation(_t), t(translation)
{
    // nothing to do here
    return;
}


Pose::Pose(const Pose &pose):
    rotation(pose.R), R(rotation), translation(pose.t), t(translation)
{
    // nothing to do here
    return;
}


Pose::~Pose()
{
    // nothing to do here
    return;
}

Pose &Pose::operator =(const Pose &pose)
{
    rotation = pose.rotation; // R still points to the proper instance
    translation = pose.translation; // T still points to the proper instance
    return *this;
}


} // end of namespace doppia
