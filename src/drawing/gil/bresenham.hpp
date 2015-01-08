#ifndef _bresenham_hpp_
#define _bresenham_hpp_

#include <cstdlib>
#include <algorithm>

namespace doppia {

// this methods are based on the code provided at www.reportbase.com/gil-tests.tar.gz

template <typename view_t, typename pixel_t>
inline void bresenham(const view_t& view, const pixel_t &pixel, int x1, int y1, int x2, int y2)
{

    const int max_x = view.width() - 1, max_y = view.height() - 1;
    const bool do_boundaries_check = true;

    int delta_x = std::abs(x2 - x1) << 1;
    int delta_y = std::abs(y2 - y1) << 1;

    // if x1 == x2 or y1 == y2, then it does not matter what we set here
    signed char ix = x2 > x1?1:-1;
    signed char iy = y2 > y1?1:-1;

    if(do_boundaries_check)
    {
        if(x1 >= 0 and x1 < max_x and y1 >= 0 and y1 < max_y)
        {
            view(x1, y1) = pixel;
        }
        else
        {
            // skip this pixel
        }
    }
    else
    {
        view(x1, y1) = pixel;
    }

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error = delta_y - (delta_x >> 1);

        while (x1 != x2)
        {
            if (error >= 0)
            {
                if (error || (ix > 0))
                {
                    y1 += iy;
                    error -= delta_x;
                }
                // else do nothing
            }
            // else do nothing

            x1 += ix;
            error += delta_y;

            if(do_boundaries_check)
            {
                if(x1 >= 0 and x1 < max_x and y1 >= 0 and y1 < max_y)
                {
                    view(x1, y1) = pixel;
                }
                else
                {
                    // skip this pixel
                }
            }
            else
            {
                view(x1, y1) = pixel;
            }
        }
    }
    else
    {
        // error may go below zero
        int error = delta_x - (delta_y >> 1);

        while (y1 != y2)
        {
            if (error >= 0)
            {
                if (error || (iy > 0))
                {
                    x1 += ix;
                    error -= delta_y;
                }
                // else do nothing
            }
            // else do nothing

            y1 += iy;
            error += delta_x;

            if(do_boundaries_check)
            {
                if(x1 >= 0 and x1 < max_x and y1 >= 0 and y1 < max_y)
                {
                    view(x1, y1) = pixel;
                }
                else
                {
                    // skip this pixel
                }
            }
            else
            {
                view(x1, y1) = pixel;
            }
        }
	
    }

    return;
} // end of function bresenham

template <typename view_t>
struct draw_bresenham
{
    typedef typename view_t::value_type value_type;
    const view_t& view;
    value_type pixel;
    draw_bresenham(const view_t& view, const value_type& pixel) :
        view(view),pixel(pixel) {}

    void operator()(int x, int y, int x1, int y1)
    {
        bresenham(view,pixel,x,y,x1,y1);
    }

    template <typename point_t>
    void operator()(point_t pt0, point_t pt1)
    {
        bresenham(view,pixel,pt0.x,pt0.y,pt1.x,pt1.y);
    }
};

} // end of namespace doppia

#endif
