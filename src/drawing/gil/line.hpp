#if not defined(LINE_HEADER_INCLUDED)
#define LINE_HEADER_INCLUDED

#include "bresenham.hpp"

namespace doppia {

template <typename view_t, typename pixel_t>
inline void draw_line(const view_t& view, const pixel_t &color, int x1, int y1, int x2, int y2)
{
    bresenham(view, color, x1, y1, x2, y2);
    return;
}


template <typename view_t, typename pixel_t, typename point_t>
inline void draw_line(const view_t& view, const pixel_t &color, const point_t pt0, const point_t pt1)
{
    bresenham(view, color, pt0.x, pt0.y, pt1.x, pt1.y);
    return;
}


/// we assume that rectangle_t is of type compatible with boost::geometry::box
template <typename view_t, typename pixel_t, typename rectangle_t>
void draw_rectangle(const view_t& view, const pixel_t &color, const rectangle_t &r)
{
    const int
            x1 = r.min_corner().x(), y1 = r.min_corner().y(),
            x2 = r.max_corner().x(), y2 = r.max_corner().y();

    draw_line(view, color, x1, y1, x1, y2);
    draw_line(view, color, x1, y2, x2, y2);
    draw_line(view, color, x2, y2, x2, y1);
    draw_line(view, color, x2, y1, x1, y1);
    return;
}

/// we assume that rectangle_t is of type compatible with boost::geometry::box
template <typename view_t, typename pixel_t, typename rectangle_t>
void draw_rectangle(const view_t& view, const pixel_t &color, const rectangle_t &r, int line_width)
{

    if (line_width <= 0)
    {
        line_width = 1;
    }

    for(line_width -=1; line_width >= 0; line_width -=1)
    {
        const int
                x1 = r.min_corner().x()+line_width, y1 = r.min_corner().y()+line_width,
                x2 = r.max_corner().x()-line_width, y2 = r.max_corner().y()-line_width;

        const typename rectangle_t::point_type
                mincorner(x1, y1), maxcorner(x2, y2);
        const rectangle_t rectangle_to_draw(mincorner, maxcorner);
        draw_rectangle(view, color, rectangle_to_draw);
    }
    return;
}

} // end of namespace doppia

#endif // not defined(LINE_HEADER_INCLUDED)
