#ifndef ABSTRACTLINESDETECTOR_HPP
#define ABSTRACTLINESDETECTOR_HPP

#include <Eigen/Geometry>
#include <boost/gil/typedefs.hpp>

#include <vector>

namespace doppia {

/// Abstract class for a functor used to detect lines in an image
class AbstractLinesDetector
{
public:

    typedef Eigen::ParametrizedLine<float, 2> line_t;
    typedef std::vector<line_t> lines_t;

    typedef  boost::gil::gray8c_view_t source_view_t;

    AbstractLinesDetector();
    virtual ~AbstractLinesDetector();

    virtual void operator()(const source_view_t &src, lines_t &lines) = 0;
};

} // end of namespace doppia

#endif // ABSTRACTLINESDETECTOR_HPP
