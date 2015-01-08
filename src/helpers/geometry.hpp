#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

//#include <boost/geometry/geometries/point_xy.hpp>
//#include <boost/geometry/geometries/box.hpp>

// little trick to get proper compilation inside and outside the CUDA nvcc compiler
#if defined(USE_GPU)
#include <host_defines.h>
#else // not defined(USE_GPU)
#if not defined(__device__)
#define __device__
#endif
#if not defined(__host__)
#define __host__
#endif
#endif // defined(USE_GPU)

namespace doppia {

namespace geometry {
/// This namespace contains simplified versions of
///  boost::geometry::model::d2::point_xy<> and boost::geometry::model::box<>

// In order to reuse the same datastructures in the CUDA code,
// we define our simplified point and box classes, which conform to the boost::geometry API,
// but do not include lexical_cast in one of its dark corners
//typedef boost::geometry::model::d2::point_xy<coordinate_t> point_t;
//typedef boost::geometry::model::box<point_t> rectangle_t;

template<typename CoordinateType>
class point_xy
{
public:

    typedef CoordinateType coordinate_t;

    /// Default constructor, does not initialize anything
    __host__ __device__ inline point_xy()
    {}

    /// Constructor with x/y values
    __host__ __device__ inline point_xy(CoordinateType const& x, CoordinateType const& y)
        : m_x(x), m_y(y)
    {}

    /// Copy constructor
    __host__ __device__ inline point_xy(const point_xy &other)
        : m_x(other.x()), m_y(other.y())
    {}

    /// Get x-value
    __host__ __device__ inline CoordinateType const& x() const
    { return m_x; }

    /// Get y-value
    __host__ __device__ inline CoordinateType const& y() const
    { return m_y; }

    /// Set x-value
    __host__ __device__ inline void x(CoordinateType const& v)
    { m_x = v; }

    /// Set y-value
    __host__ __device__ inline void y(CoordinateType const& v)
    { m_y = v; }

    bool operator==(const point_xy &other) const
    {
        return (m_x == other.m_x) and (m_y == other.m_y);
    }

    bool operator!=(const point_xy &other) const
    {
        return (*this == other) == false;
    }

protected:

    CoordinateType m_x, m_y;
};


template<typename Point>
class box
{
    // BOOST_CONCEPT_ASSERT( (concept::Point<Point>) );
public:

    typedef Point point_type;

    __host__ __device__ inline box() {}

    ///    \brief Constructor taking the minimum corner point and the maximum corner point
    __host__ __device__ inline box(Point const& min_corner, Point const& max_corner)
    {
        m_min_corner = min_corner;
        m_max_corner = max_corner;
    }

    __host__ __device__ inline Point const& min_corner() const { return m_min_corner; }
    __host__ __device__ inline Point const& max_corner() const { return m_max_corner; }

    __host__ __device__ inline Point& min_corner() { return m_min_corner; }
    __host__ __device__ inline Point& max_corner() { return m_max_corner; }


    bool operator==(const box &other) const
    {
        return (m_min_corner == other.m_min_corner) and (m_max_corner == other.m_max_corner);
    }

    bool operator!=(const box &other) const
    {
        return (*this == other) == false;
    }

private:

    Point m_min_corner;
    Point m_max_corner;
};

} // end of namespace doppia::geometry

} // end of namespace doppia

#endif // GEOMETRY_HPP
