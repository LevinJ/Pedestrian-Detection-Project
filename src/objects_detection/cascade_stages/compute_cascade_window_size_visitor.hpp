#ifndef DOPPIA_COMPUTE_CASCADE_WINDOW_SIZE_HPP
#define DOPPIA_COMPUTE_CASCADE_WINDOW_SIZE_HPP

#include "objects_detection/ScaleData.hpp"

#include <boost/foreach.hpp>
#include <boost/variant/static_visitor.hpp>

#include <cstdio>
#include <stdexcept>

namespace doppia {

/// Helper visitor class used to verify if the stages lie inside an expected range
class compute_cascade_window_size_visitor : public boost::static_visitor<ScaleData::detection_window_size_t>
{

public:
    typedef ScaleData::detection_window_size_t detection_window_size_t;

    template<typename StageType>
    detection_window_size_t operator()(const std::vector<StageType> &stages) const;

protected:

    /// update the window size based on a single stage
    template<typename StageType>
    void update_window_size(const StageType &stage, detection_window_size_t& window_size) const;

    /// update the window size based on a single feature
    void update_window_size(const IntegralChannelsFeature &feature, detection_window_size_t& window_size) const;

};



// Templated implementations need to be on the header
template<typename StageType>
compute_cascade_window_size_visitor::detection_window_size_t
compute_cascade_window_size_visitor::operator()(const std::vector<StageType> &stages) const
{

    detection_window_size_t window_size(0, 0);


    BOOST_FOREACH(const StageType &stage, stages)
    {
        update_window_size(stage, window_size);
    }


    return window_size;
}

template<typename StageType>
void compute_cascade_window_size_visitor::update_window_size(const StageType &, detection_window_size_t& ) const
{
    printf("StageType == %s\n", typeid(StageType).name());
    throw std::runtime_error("compute_cascade_window_size_visitor::update_window_size "
                             "received an unknown (not yet implemented?) stage type");
    return;
}


// prototype declaration
template<>
void compute_cascade_window_size_visitor::update_window_size
<SoftCascadeOverIntegralChannelsModel::fast_stage_t>
(const SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage, detection_window_size_t &window_size) const;


} // end of namespace doppia

#endif // DOPPIA_COMPUTE_CASCADE_WINDOW_SIZE_HPP
