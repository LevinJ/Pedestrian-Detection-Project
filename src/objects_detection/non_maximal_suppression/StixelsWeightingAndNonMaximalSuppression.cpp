#include "StixelsWeightingAndNonMaximalSuppression.hpp"

#include "GreedyNonMaximalSuppression.hpp"

#include <boost/foreach.hpp>

#include <cstdio>

namespace doppia {

using namespace std;
using namespace boost;
using namespace boost::program_options;


options_description
StixelsWeightingAndNonMaximalSuppression::get_args_options()
{
    options_description desc("StixelsWeightingAndNonMaximalSuppression options");

    desc.add_options()
            ;

    return desc;
}

StixelsWeightingAndNonMaximalSuppression::StixelsWeightingAndNonMaximalSuppression(const variables_map &options)
{
    non_maximal_suppression_p.reset(new GreedyNonMaximalSuppression(options));
    return;
}


StixelsWeightingAndNonMaximalSuppression::~StixelsWeightingAndNonMaximalSuppression()
{
    // nothing to do here
    return;
}


void StixelsWeightingAndNonMaximalSuppression::set_u_disparity_cost(const StixelsEstimator::u_disparity_cost_t &u_disparity_cost_)
{
    // FIXME dummy implementation, we simply copy the data,
    // should be more efficient and keep pointers, or something like that...
    u_disparity_cost = u_disparity_cost_;


    return;
}

void StixelsWeightingAndNonMaximalSuppression::set_disparity_given_v(const std::vector<int> &disparity_given_v_)
{
    disparity_given_v = disparity_given_v_;
    return;
}

void StixelsWeightingAndNonMaximalSuppression::set_detections(const AbstractNonMaximalSuppression::detections_t &detections)
{
    // we copy the data (since we are going to re-weight the scores)
    candidate_detections = detections;
    return;
}

void StixelsWeightingAndNonMaximalSuppression::compute()
{
    assert(non_maximal_suppression_p);
    assert(disparity_given_v.empty() == false);
    assert(u_disparity_cost.size() > 0);


    // we re-weight the detections
    {
        const float max_u_disparity_cost = u_disparity_cost.maxCoeff();

        BOOST_FOREACH(detection_t &detection, candidate_detections)
        {
            const detection_t::rectangle_t &box = detection.bounding_box;
            const size_t v = std::min<size_t>(detection.bounding_box.max_corner().y(), disparity_given_v.size() - 1);
            const int disparity = disparity_given_v[v];
            if(disparity >= 0)
            {
                float cost = 0;

                const int
                        //delta_d = 2,
                        delta_d = 10, // 10 is better than 2, 20 is like 10
                        //delta_d = 50, // 50 is like 10
                        min_d = std::max<int>(0, disparity - delta_d),
                        max_d = std::min<int>(disparity + delta_d, u_disparity_cost.cols() - 1);

                const int
                        min_u = std::max<int>(0, box.min_corner().x()),
                        max_u = std::min<int>(box.max_corner().x(), u_disparity_cost.rows());
                for(int u=min_u; u < max_u; u+=1)
                {
                    const bool search_around = true;
                    if(search_around)
                    {
                        float minimum_local_cost = std::numeric_limits<float>::max();
                        for(int d=min_d; d <= max_d; d+=1)
                        {
                            const float t_cost = u_disparity_cost(d, u);
                            minimum_local_cost = std::min(minimum_local_cost, t_cost);
                        }
                        cost += minimum_local_cost;
                    }
                    else
                    {
                        cost += u_disparity_cost(disparity, u);
                    }

                } // end of "for each column in the detection window"

                cost /= (max_u - min_u); // we average the cost

                const bool threshold_scores = false;
                const float score_threshold = 35;

                if(threshold_scores)
                {
                    if(cost < score_threshold)
                    {
                        // we keep the same score
                        detection.score = detection.score;
                    }
                    else
                    {
                        // we remove the detection
                        detection.score = -10*abs(detection.score);
                    }
                }
                else
                {
                    const bool use_bad_reweighting = false;
                    if(use_bad_reweighting)
                    {
                        // this reweighthing is worse the that one in the "else" block
                        cost -= score_threshold; // above zero is good, below zero is bad
                        cost /= -score_threshold; // max value is 1, some high negative value

                        detection.score = detection.score + 1*std::max(std::min(3.0f, cost), -3.0f)*abs(detection.score);
                    }
                    else
                    {
                        cost /= max_u_disparity_cost; // we normalize with the maximum cost
                        // here cost is in [0, 1]
                        cost = 1 - cost; // now cost is good at 1 and bad at 0
                        detection.score = std::max<float>(0, detection.score - 0.2*cost + 1); // FIXME 2 is a hardcoded parameter
                    }
                }
            }
            else
            {
                // no valid disparity, nothing to do
            }

        } // end of "for each detection"

    }

    if(non_maximal_suppression_p)
    {
        // we provide them to the underlying non-maximimal suppression module
        non_maximal_suppression_p->set_detections(candidate_detections);
        non_maximal_suppression_p->compute();

        maximal_detections = non_maximal_suppression_p->get_detections();
    }
    else
    {
        maximal_detections = candidate_detections;
    }

    return;
}


} // namespace doppia
