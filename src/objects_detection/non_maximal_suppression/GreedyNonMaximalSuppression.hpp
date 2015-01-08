#ifndef GREEDYNONMAXIMALSUPPRESSION_HPP
#define GREEDYNONMAXIMALSUPPRESSION_HPP

#include "AbstractNonMaximalSuppression.hpp"

#include <boost/program_options.hpp>
#include <list>
#include <string>

namespace doppia {

/// This class implements the greedy non maximal suppression described by
/// Piotr Dollar, "Integral Channel Features - Addendum", 2009.
/// Section 1.2.1, the greddy* variant
class GreedyNonMaximalSuppression : public AbstractNonMaximalSuppression
{
public:

    static boost::program_options::options_description get_args_options();

    GreedyNonMaximalSuppression(const boost::program_options::variables_map &options);
    GreedyNonMaximalSuppression(const float minimal_overlap_threshold, const std::string overlap_method_);
    ~GreedyNonMaximalSuppression();

    void set_detections(const detections_t &detections);

    void compute();

protected:

    const float minimal_overlap_threshold;
    const std::string overlap_method;
    typedef std::list<detection_t> candidate_detections_t;
    candidate_detections_t candidate_detections;

};


/// Helper function used in bootstrapped_lib
/// compute the overlap between two detections, using the P. Dollar overlap criterion
/// this is _not_ the PASCAL VOC overlap criterion
float compute_overlap(const GreedyNonMaximalSuppression::detection_t &a,
                      const GreedyNonMaximalSuppression::detection_t &b, 
					  const std::string &method="dollar");

} // end of namespace doppia

#endif // GREEDYNONMAXIMALSUPPRESSION_HPP
