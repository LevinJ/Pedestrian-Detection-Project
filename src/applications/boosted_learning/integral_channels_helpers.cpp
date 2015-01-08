#include "integral_channels_helpers.hpp"

namespace boosted_learning {


/// the out integral channel will be resized to the required dimensions
void get_integral_channels(const integral_channels_t &in,
                           const point_t &modelWindowSize, const point_t &dataOffset, const int resizing_factor,
                           integral_channels_t &out)
{
    get_integral_channels(in,
                          dataOffset.x(), dataOffset.y(),
                          modelWindowSize.x(), modelWindowSize.y(),
                          resizing_factor,
                          out);
    return;
}


void get_integral_channels(const integral_channels_t &in,
                           const int inX, const int inY, const int inW, const int inH,
                           const int resizing_factor,
                           integral_channels_t &out)
{
    int x, y, w, h;

    if (resizing_factor == 4)
    {
        x = (((inX + 1) / 2) + 1) / 2;
        y = (((inY + 1) / 2) + 1) / 2;
        w = (((inW + 1) / 2) + 1) / 2;
        h = (((inH + 1) / 2) + 1) / 2;
    }
    else if (resizing_factor == 2)
    {
        x = (inX + 1) / 2;
        y = (inY + 1) / 2;
        w = (inW + 1) / 2;
        h = (inH + 1) / 2;
    }
    else
    {
        x =  inX;
        y =  inY;
        w =  inW;
        h =  inH;
    }

    out.resize(boost::extents[in.shape()[0]][h+1][w+1]);// = integral_channels_t();

    for (size_t ch = 0; ch < in.shape()[0]; ch += 1)
    {
        for (size_t yy = y; yy < static_cast<size_t>(y + h + 1); yy += 1)
        {
            //std::copy(&(integral_channels[ch][yy][x]), &(integral_channels[ch][yy][x+w+1]), out[ch][yy-y].begin());
            for (size_t xx = x; xx < static_cast<size_t>(x + w + 1); xx += 1)
            {
                out[ch][yy-y][xx-x] = in[ch][yy][xx];
            } // end of "for each column"
        }// end of "for each row"

    } // end of "for each channel"

    return;
}

integral_channels_const_view_t
get_integral_channels_view(const integral_channels_t &in,
                           const int inX, const int inY, const int inW, const int inH,
                           const int resizing_factor)
{
    // generate view on the data of the size of the resized rectangle
    int x, y, w, h;

    if (resizing_factor == 4)
    {
        x = (((inX + 1) / 2) + 1) / 2;
        y = (((inY + 1) / 2) + 1) / 2;
        w = (((inW + 1) / 2) + 1) / 2;
        h = (((inH + 1) / 2) + 1) / 2;
    }
    else if (resizing_factor == 2)
    {
        x = (inX + 1) / 2;
        y = (inY + 1) / 2;
        w = (inW + 1) / 2;
        h = (inH + 1) / 2;
    }
    else
    {
        x =  inX;
        y =  inY;
        w =  inW;
        h =  inH;
    }

    typedef integral_channels_t::index_range range;
    integral_channels_const_view_t out = in[ boost::indices[range()][range(y, y+h+1)][range(x, x+w+1)] ];
    return out;
}

} // end of namespace boosted_learning
