#include "fast_rgb_to_luv.hpp"

#include <boost/gil/gil_all.hpp>

#include <vector>
#include <cfloat>
#include <stdexcept>

namespace doppia {

/// cube root approximation using bit hack for 32-bit float
/// provides a very crude approximation
inline
float cbrt_5_f32(float f)
{
    unsigned int* p = reinterpret_cast<unsigned int *>(&f);
    *p = *p/3 + 709921077;
    return f;
}

/// iterative cube root approximation using Halley's method (float)
inline
float cbrta_halley_f32(const float a, const float R)
{
    const float a3 = a*a*a;
    const float b = a * (a3 + R + R) / (a3 + a3 + R);
    return b;
}

/// Code based on
/// http://metamerist.com/cbrt/cbrt.htm
/// cube root approximation using 2 iterations of Halley's method (float)
/// this is expected to be ~2.5x times faster than std::pow(x, 3)
inline
float fast_cube_root(const float d)
{
    float a = cbrt_5_f32(d);
    a = cbrta_halley_f32(a, d);
    return cbrta_halley_f32(a, d);
}

/// Helper table to construct a lookup table
/// will only compute the root for values in the range [0, 1]
class CubeRootTable
{
public:

    CubeRootTable(const int num_bins);
    float operator()(const float x) const;

protected:
    std::vector<float> lookup_table;
    size_t max_i;
};

CubeRootTable::CubeRootTable(const int num_bins)
    : lookup_table(num_bins),
      max_i(num_bins - 1)
{
    for(int i=0; i < num_bins; i+=1)
    {
        const float x = static_cast<float>(i)/max_i;
        lookup_table[i] = pow(x, 1.0f/3.0f);
    }
    return;
}

float CubeRootTable::operator()(const float x) const
{
    const size_t i = static_cast<size_t>(x*max_i);
    assert(i >= 0);
    assert(i < lookup_table.size());

    return lookup_table[i];
    //return lookup_table[std::min(i, max_i)];
}


/// this code is based on the equations from
/// http://software.intel.com/sites/products/documentation/hpc/ipp/ippi/ippi_ch6/ch6_color_models.html
/// and from
/// http://www.f4.fhtw-berlin.de/~barthel/ImageJ/ColorInspector//HTMLHelp/farbraumJava.htm
/// and from RGB2Luv_f
/// https://code.ros.org/trac/opencv/browser/trunk/opencv/modules/imgproc/src/color.cpp
/// will map from rgb8c_layout_t to boost::gil::dev3n8c_pixel_t
/// additional references
/// http://en.wikipedia.org/wiki/CIELUV
template<typename ChannelValue>
inline
boost::gil::pixel<ChannelValue, boost::gil::devicen_layout_t<3> >
rgb_to_luv(const boost::gil::pixel<ChannelValue, boost::gil::rgb_layout_t> &rgb_value)
{

    // static ensures a single global instance
    //static const CubeRootTable cube_root_table(256);
    //static const CubeRootTable cube_root_table(1024);
    static const CubeRootTable cube_root_table(2048);
    typedef  boost::gil::pixel<ChannelValue, boost::gil::devicen_layout_t<3> > luv_pixel_t;
    luv_pixel_t luv_value;

    const float
            r = rgb_value[0] / 255.0f,
            g = rgb_value[1] / 255.0f,
            b = rgb_value[2] / 255.0f;

    assert(r <= 1.0f);
    assert(g <= 1.0f);
    assert(b <= 1.0f);

    const float
            x = 0.412453f*r + 0.35758f*g + 0.180423f*b,
            y = 0.212671f*r + 0.71516f*g + 0.072169f*b,
            z = 0.019334f*r + 0.119193f*g + 0.950227f*b;

    const float
            x_n = 0.312713f, y_n = 0.329016f,
            uv_n_divisor = -2.f*x_n + 12.f*y_n + 3.f,
            u_n = 4.f*x_n / uv_n_divisor,
            v_n = 9.f*y_n / uv_n_divisor;

    const float
            uv_divisor = std::max((x + 15.f*y + 3.f*z), FLT_EPSILON),
            u = 4.f*x / uv_divisor,
            v = 9.f*y / uv_divisor;

    // opencv's rgb to luv is ~60 [Hz] on test_objects_detection (not considering multi-threading)
    //const float y_cube_root = pow(y, 1.0f/3.0f); // ~40 [Hz] on test_objects_detection
    //const float y_cube_root = fast_cube_root(y); // ~90 [Hz] on test_objects_detection
    const float y_cube_root = cube_root_table(y); // ~170 [Hz] on test_objects_detection

    const float
            l_value = std::max(0.f, ((116.f * y_cube_root) - 16.f)),
            u_value = 13.f * l_value * (u - u_n),
            v_value = 13.f * l_value * (v - v_n);

    // L in [0, 100], U in [-134, 220], V in [-140, 122]
    const float
            scaled_l = l_value * (255.f / 100.f),
            scaled_u = (u_value + 134.f) * (255.f / (220.f + 134.f )),
            scaled_v = (v_value + 140.f) * (255.f / (122.f + 140.f ));

    luv_value[0] = static_cast<boost::uint8_t>(scaled_l);
    luv_value[1] = static_cast<boost::uint8_t>(scaled_u);
    luv_value[2] = static_cast<boost::uint8_t>(scaled_v);

    return luv_value;
}

void fast_rgb_to_luv(const boost::gil::rgb8c_view_t &rgb_view,
                     const boost::gil::dev3n8_view_t &luv_view)
{

    using namespace boost::gil;

    if(rgb_view.dimensions() != luv_view.dimensions())
    {
        throw std::invalid_argument("rgb_to_luv expects views of the same dimensions");
    }

#pragma omp parallel for
    for(size_t row=0; row < static_cast<size_t>(rgb_view.height()); row +=1)
    {
        rgb8c_view_t::x_iterator rgb_row_it = rgb_view.row_begin(row);
        dev3n8_view_t::x_iterator luv_row_it = luv_view.row_begin(row);

        for(size_t col=0; col < static_cast<size_t>(rgb_view.width());
            col +=1, ++rgb_row_it, ++luv_row_it)
        {
            (*luv_row_it) = rgb_to_luv(*rgb_row_it);
        } // end of "for each column"
    } // end of "for each row"

    return;
}

void fast_rgb_to_luv(const boost::gil::rgb8c_view_t &rgb_view,
                     const boost::gil::dev3n8_planar_view_t &luv_view)
{

    using namespace boost::gil;

    if(rgb_view.dimensions() != luv_view.dimensions())
    {
        throw std::invalid_argument("rgb_to_luv expects views of the same dimensions");
    }

#pragma omp parallel for
    for(size_t row=0; row < static_cast<size_t>(rgb_view.height()); row +=1)
    {
        rgb8c_view_t::x_iterator rgb_row_it = rgb_view.row_begin(row);
        dev3n8_planar_view_t::x_iterator luv_row_it = luv_view.row_begin(row);

        for(size_t col=0; col < static_cast<size_t>(rgb_view.width());
            col +=1, ++rgb_row_it, ++luv_row_it)
        {
            (*luv_row_it) = rgb_to_luv(*rgb_row_it);

        } // end of "for each column"
    } // end of "for each row"

    return;
}


void fast_rgb_to_luv(const boost::gil::rgb16c_view_t &rgb_view,
                     const boost::gil::dev3n16_view_t &luv_view)
{

    using namespace boost::gil;

    if(rgb_view.dimensions() != luv_view.dimensions())
    {
        throw std::invalid_argument("rgb_to_luv expects views of the same dimensions");
    }

#pragma omp parallel for
    for(size_t row=0; row < static_cast<size_t>(rgb_view.height()); row +=1)
    {
        rgb16c_view_t::x_iterator rgb_row_it = rgb_view.row_begin(row);
        dev3n16_view_t::x_iterator luv_row_it = luv_view.row_begin(row);

        for(size_t col=0; col < static_cast<size_t>(rgb_view.width());
            col +=1, ++rgb_row_it, ++luv_row_it)
        {
            (*luv_row_it) = rgb_to_luv(*rgb_row_it);
        } // end of "for each column"
    } // end of "for each row"

    return;
}


} // end of namespace doppia

