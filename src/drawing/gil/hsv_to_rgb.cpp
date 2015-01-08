#include "hsv_to_rgb.hpp"

namespace doppia {


/// @param hue should be in range [0,1]
/// @param saturation should be in range [0,1]
/// @param value should be in range [0,1]
boost::gil::rgb8c_pixel_t hsv_to_rgb(const float hue, const float saturation, const float value)
{

    boost::gil::rgb8_pixel_t pixel_value;

    float
            H = hue*360,
            S = saturation,
            V = value,
            R = 0, G = 0, B = 0;

    if (H==0 and S==0)
    {
        R = G = B = V;
    }
    else
    {
        H/=60;
        const int i = (int) std::floor(H);
        const float
                f = (i&1)?(H-i):(1-H+i),
                m = V*(1-S),
                n = V*(1-S*f);
        switch (i)
        {
        case 6 :
        case 0 :
            R = V;
            G = n;
            B = m;
            break;
        case 1 :
            R = n;
            G = V;
            B = m;
            break;
        case 2 :
            R = m;
            G = V;
            B = n;
            break;
        case 3 :
            R = m;
            G = n;
            B = V;
            break;
        case 4 :
            R = n;
            G = m;
            B = V;
            break;
        case 5 :
            R = V;
            G = m;
            B = n;
            break;
        }
    }
    R*=255;
    G*=255;
    B*=255;
    get_color(pixel_value, boost::gil::red_t())   = (R<0?0:(R>255?255:R));
    get_color(pixel_value, boost::gil::green_t()) = (G<0?0:(G>255?255:G));
    get_color(pixel_value, boost::gil::blue_t())  = (B<0?0:(B>255?255:B));

    return pixel_value;
}


void hsv_to_rgb( float H, float S, float V, float& R, float& G, float& B )
{
    R = 0;
    G = 0;
    B = 0;

    if (H==0 and S==0)
    {
        R = G = B = V;
    }
    else
    {
        H /= 60;
        const int i = (int) std::floor(H);
        const float
                f = (i&1)?(H-i):(1-H+i),
                m = V*(1-S),
                n = V*(1-S*f);
        switch (i)
        {
        case 6 :
        case 0 :
            R = V;
            G = n;
            B = m;
            break;
        case 1 :
            R = n;
            G = V;
            B = m;
            break;
        case 2 :
            R = m;
            G = V;
            B = n;
            break;
        case 3 :
            R = m;
            G = n;
            B = V;
            break;
        case 4 :
            R = n;
            G = m;
            B = V;
            break;
        case 5 :
            R = V;
            G = m;
            B = n;
            break;
        }
    }

    R*=255;
    G*=255;
    B*=255;

    return;
}


} // end of namespace doppia
