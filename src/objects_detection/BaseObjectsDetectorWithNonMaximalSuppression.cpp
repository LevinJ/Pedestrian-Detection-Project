#include "BaseObjectsDetectorWithNonMaximalSuppression.hpp"

#include "non_maximal_suppression/AbstractNonMaximalSuppression.hpp"

namespace doppia {

BaseObjectsDetectorWithNonMaximalSuppression::BaseObjectsDetectorWithNonMaximalSuppression(
    const boost::program_options::variables_map &options,
    const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p_)
    : AbstractObjectsDetector(options),
      non_maximal_suppression_p(non_maximal_suppression_p_)
{
    // nothing to do here
    return;
}

BaseObjectsDetectorWithNonMaximalSuppression::~BaseObjectsDetectorWithNonMaximalSuppression()
{
    // nothing to do here
    return;
}

const AbstractObjectsDetector::detections_t &BaseObjectsDetectorWithNonMaximalSuppression::get_detections()
{
    if(non_maximal_suppression_p)
    {
        return non_maximal_suppression_p->get_detections();
    }
    else
    {
        return detections;
    }

    return detections;
}


void BaseObjectsDetectorWithNonMaximalSuppression::compute_non_maximal_suppresion()
{

    if(non_maximal_suppression_p)
    {
        non_maximal_suppression_p->set_detections(detections);
        non_maximal_suppression_p->compute();
    }
    else
    {
        // detections == detections
    }

    return;
}

} // end of namespace doppia
