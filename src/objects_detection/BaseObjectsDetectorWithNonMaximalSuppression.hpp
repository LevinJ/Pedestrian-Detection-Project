#ifndef BASEOBJECTSDETECTORWITHNONMAXIMALSUPPRESSION_HPP
#define BASEOBJECTSDETECTORWITHNONMAXIMALSUPPRESSION_HPP

#include "AbstractObjectsDetector.hpp"

namespace doppia {

class AbstractNonMaximalSuppression;

class BaseObjectsDetectorWithNonMaximalSuppression : public AbstractObjectsDetector
{
public:
    BaseObjectsDetectorWithNonMaximalSuppression(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p);
    ~BaseObjectsDetectorWithNonMaximalSuppression();

    const detections_t &get_detections();

protected:

    boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p;

    void compute_non_maximal_suppresion();
};


} // end of namespace doppia

#endif // BASEOBJECTSDETECTORWITHNONMAXIMALSUPPRESSION_HPP
