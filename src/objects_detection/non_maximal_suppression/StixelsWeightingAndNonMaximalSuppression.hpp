#ifndef DOPPIA_STIXELSWEIGHTINGANDNONMAXIMALSUPPRESSION_HPP
#define DOPPIA_STIXELSWEIGHTINGANDNONMAXIMALSUPPRESSION_HPP

#include "AbstractNonMaximalSuppression.hpp"

#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include <boost/scoped_ptr.hpp>

#include <vector>

namespace doppia {


/// This object will receive detections, re-weight their score using the stixels data term evidence,
/// and then apply non-maximal suppression
class StixelsWeightingAndNonMaximalSuppression : public AbstractNonMaximalSuppression
{
public:

    static boost::program_options::options_description get_args_options();

    StixelsWeightingAndNonMaximalSuppression(const boost::program_options::variables_map &options);
    ~StixelsWeightingAndNonMaximalSuppression();

    void set_u_disparity_cost(const StixelsEstimator::u_disparity_cost_t &u_disparity_cost);

    void set_disparity_given_v(const std::vector<int> &disparity_given_v);

    void set_detections(const detections_t &detections);

    void compute();

protected:

    boost::scoped_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p;

    StixelsEstimator::u_disparity_cost_t u_disparity_cost;
    std::vector<int> disparity_given_v;

    typedef std::vector<detection_t> candidate_detections_t;
    candidate_detections_t candidate_detections;

};

} // end of namespace doppia

#endif // DOPPIA_STIXELSWEIGHTINGANDNONMAXIMALSUPPRESSION_HPP
