#ifndef ABSTRACTGROUNDPLANEESTIMATOR_HPP
#define ABSTRACTGROUNDPLANEESTIMATOR_HPP

#include "GroundPlane.hpp"
#include "image_processing/AbstractLinesDetector.hpp"

namespace doppia {

class AbstractGroundPlaneEstimator
{
public:

    typedef AbstractLinesDetector::line_t line_t;

    AbstractGroundPlaneEstimator();
    virtual ~AbstractGroundPlaneEstimator();

    virtual void compute() = 0;

    /// @returns the estimated ground plane
    virtual const GroundPlane &get_ground_plane() const = 0;

    /// @returns the line that describes the mapping from disparities to v value
    virtual const line_t &get_ground_v_disparity_line() const = 0;
};

} // end of namespace doppia

#endif // ABSTRACTGROUNDPLANEESTIMATOR_HPP
