#ifndef DOPPIA_FIXEDWINDOWNONMAXIMALSUPPRESSION_HPP
#define DOPPIA_FIXEDWINDOWNONMAXIMALSUPPRESSION_HPP

#include "AbstractNonMaximalSuppression.hpp"

namespace doppia {

/// Helper function used during the BMVC2012 experiments
/// always returns a single window, at a fixed (hardcoded) position
class FixedWindowNonMaximalSuppression : public AbstractNonMaximalSuppression
{
public:
    FixedWindowNonMaximalSuppression();
    ~FixedWindowNonMaximalSuppression();

    /// Provide the (large) set of raw detections from the sliding window (or similar) detector
    void set_detections(const detections_t &detections);

    void compute();

protected:

    detections_t candidate_detections;

};

} // end namespace doppia

#endif // DOPPIA_FIXEDWINDOWNONMAXIMALSUPPRESSION_HPP
