#ifndef COST_FUNCTIONS_HPP
#define COST_FUNCTIONS_HPP

#include "boost/gil/typedefs.hpp"
#include "boost/gil/color_base_algorithm.hpp"

#include <stdexcept>

namespace doppia
{

using namespace boost;

template <typename ReturnType>
class SadCostFunctionT {

public:

    typedef ReturnType cost_t;

    ReturnType operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const;
    ReturnType operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const;

    template<typename PixelType>
    ReturnType get_maximum_cost_per_pixel() const;

};

typedef SadCostFunctionT<float> SadCostFunction;

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
// perfomance critical functions should be inline

template <>
inline float SadCostFunctionT<float>::operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const {

    const float delta_r = pixel_a[0] - pixel_b[0];
    const float delta_g = pixel_a[1] - pixel_b[1];
    const float delta_b = pixel_a[2] - pixel_b[2];

    // SAD
    const float distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    //const float distance_saturation = 120;

    //return min(distance, distance_saturation);
    return distance;
}

template <>
inline float SadCostFunctionT<float>::operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const {

    const float delta = pixel_a[0] - pixel_b[0];

    // SAD
    return std::abs(delta);
}


template <>
inline boost::int32_t SadCostFunctionT<boost::int32_t>::operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const {

    const int32_t delta_r = pixel_a[0] - pixel_b[0];
    const int32_t delta_g = pixel_a[1] - pixel_b[1];
    const int32_t delta_b = pixel_a[2] - pixel_b[2];

    // SAD
    const int32_t distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    //const float distance_saturation = 120;

    //return min(distance, distance_saturation);
    return distance;
}

template <>
inline boost::int32_t SadCostFunctionT<boost::int32_t>::operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const {

    const int32_t delta = pixel_a[0] - pixel_b[0];

    // SAD
    return std::abs(delta);
}


template <>
inline boost::uint8_t SadCostFunctionT<boost::uint8_t>::operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const {

    const int16_t delta_r = pixel_a[0] - pixel_b[0];
    const int16_t delta_g = pixel_a[1] - pixel_b[1];
    const int16_t delta_b = pixel_a[2] - pixel_b[2];

    // SAD
    const uint16_t distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    const uint8_t cost = distance / 3;

    //const int8_t distance_saturation = 120;
    //return std::min(cost, distance_saturation);
    return cost;
}

template <>
inline boost::uint8_t SadCostFunctionT<boost::uint8_t>::operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const {

     // SAD
    const int16_t delta = pixel_a[0] - pixel_b[0];
    const uint8_t cost = std::abs(delta);

    //const int8_t distance_saturation = 120;
    //return std::min(cost, distance_saturation);
    return cost;
}


// compute the raw SAD (without doing a division by the number of channels)
inline uint16_t sad_cost_uint16(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b)
{
    const int16_t delta_r = pixel_a[0] - pixel_b[0];
    const int16_t delta_g = pixel_a[1] - pixel_b[1];
    const int16_t delta_b = pixel_a[2] - pixel_b[2];

    // SAD
    const uint16_t distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    //const uint8_t cost = distance / 3;
    const uint16_t &cost = distance; // we skip the /3

    return cost;
}

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

class SsdCostFunction {

public:

    float operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const;

    float operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const;

    template<typename PixelType>
    float get_maximum_cost_per_pixel() const;

};

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
// perfomance critical functions should be inline

inline float SsdCostFunction::operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const {

    const float delta_r = pixel_a[0] - pixel_b[0];
    const float delta_g = pixel_a[1] - pixel_b[1];
    const float delta_b = pixel_a[2] - pixel_b[2];

    // SSD
    const float distance = delta_r*delta_r + delta_g*delta_g + delta_b*delta_b;
    //const float distance_saturation = 120;

    //return min(distance, distance_saturation);
    return distance;
}

inline float SsdCostFunction::operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const {

    const float delta = pixel_a[0] - pixel_b[0];

    // SSD
    return delta*delta;
}

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-


/// This is a dummy and slow implementation of the LCDM cost function proposed in
///  "Stereo vision for robotic applications in presence of non-ideal lighting conditions"
///  by L. Nalpantidis and A. Gasteratos. 2010

class LCDMCostFunction {

public:

    LCDMCostFunction();

    ~LCDMCostFunction();

    float operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const;

    float operator()(const gil::gray8c_pixel_t &pixel_a, const gil::gray8c_pixel_t &pixel_b) const;

    template<typename PixelType>
    float get_maximum_cost_per_pixel() const;

};

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
// perfomance critical functions should be inline

/**
     H is in [0, 360]
     S and L in [0, 1]
     */
inline
void rgb_to_hsl(const gil::rgb8c_pixel_t &pixel_a, float &h, float &s, float &l)
{
    const float
            R = pixel_a[0],
            G = pixel_a[1],
            B = pixel_a[2],
            nR = (R<0?0:(R>255?255:R))/255.0f,
            nG = (G<0?0:(G>255?255:G))/255.0f,
            nB = (B<0?0:(B>255?255:B))/255.0f,
            m = std::min(std::min(nR,nG),nB),
            M = std::max(std::max(nR,nG),nB);

    l = (m+M)/2;
    h = 0;
    s = 0;
    if (M!=m) {
        const float
                f = (nR==m)?(nG-nB):((nG==m)?(nB-nR):(nR-nG)),
                i = (nR==m)?3.0f:((nG==m)?5.0f:1.0f);
        h = (i-f/(M-m));
        if (h>=6)
        {
            h-=6;
        }
        h*=60;
        s = (2*l<=1)?((M-m)/(M+m)):((M-m)/(2-M-m));
    }

    return;
}

inline float LCDMCostFunction::operator()(const gil::rgb8c_pixel_t &pixel_a, const gil::rgb8c_pixel_t &pixel_b) const {


    float h1,s1,l1, h2,s2,l2;

    rgb_to_hsl(pixel_a, h1, s1, l1);
    rgb_to_hsl(pixel_b, h2, s2, l2);

    /* Is this intentional?
     * I foolishly assume that this should be a transformation from degrees to radians,
     * in which case it should be *= M_PI / 180.0f.
     * But I have no idea if this is what is intended...
     */
    h1 *= M_PI / 360.0f;
    h2 *= M_PI / 360.0f;

    // square(LCDM)
    const float distance2 = (s1*s1 + s2*s2 - 2*s1*s2*cos((h1 - h2)));
    // distance2 value is between 4 and 0
    //*255.0/4.0;

    return distance2;
}

// the gray8c_pixel_t version is not inlined, see cpp


// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// This cost function is inspired by the following paper
/// "The Gradient - A Powerful and Robust Cost Function for Stereo Matching"
/// Simon Hermann and Tobi Vaudrey. IEEE Publ. IVCNZ 2010

class GradientCostFunction {

public:

    /// the color_weight will be 1-gradient_weight
    /// the final cost will be color_cost*color_weight + gradient_cost*gradient_weight
    GradientCostFunction(const float gradient_weight);

    ~GradientCostFunction();

    /// dev5 pixel is assumed to be r,g,b, dx, dy
    float operator()(const gil::dev5n8c_pixel_t &pixel_a, const gil::dev5n8c_pixel_t &pixel_b) const;

    /// dev3 pixel is assumed to be gray, dx, dy
    float operator()(const gil::dev3n8c_pixel_t &pixel_a, const gil::dev3n8c_pixel_t &pixel_b) const;

    template<typename PixelType>
    float get_maximum_cost_per_pixel() const;

protected:

    const float gradient_weight;
    const float color_weight;

};

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
// perfomance critical functions should be inline

inline float GradientCostFunction::operator()(const gil::dev5n8c_pixel_t &pixel_a,
                                              const gil::dev5n8c_pixel_t &pixel_b) const
{

    const float delta_r = pixel_a[0] - pixel_b[0];
    const float delta_g = pixel_a[1] - pixel_b[1];
    const float delta_b = pixel_a[2] - pixel_b[2];

    // SAD
    const float color_distance = std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    //const float distance_saturation = 120;

    //color_distance = std::min(distance, distance_saturation);

    const float delta_dx = pixel_a[3] - pixel_b[3];
    const float delta_dy = pixel_a[4] - pixel_b[4];
    const float gradient_distance = std::abs(delta_dx) + std::abs(delta_dy);

    const float distance = color_weight*color_distance + gradient_weight*gradient_distance;
    return distance;
}

inline float GradientCostFunction::operator()(const gil::dev3n8c_pixel_t &pixel_a,
                                              const gil::dev3n8c_pixel_t &pixel_b) const {

    const float delta = pixel_a[0] - pixel_b[0];
    const float gray_distance = std::abs(delta); // SAD

    const float delta_dx = pixel_a[1] - pixel_b[1];
    const float delta_dy = pixel_a[2] - pixel_b[2];
    const float gradient_distance = std::abs(delta_dx) +  std::abs(delta_dy);

    const float distance = color_weight*gray_distance + gradient_weight*gradient_distance;
    return distance;
}


} // end of namespace doppia

#endif // COST_FUNCTIONS_HPP
