#include "qx_basic.hpp"
#include "qx_csbp.hpp"
#include <limits>
#include <stdexcept>
#include <cassert>
#include <cstddef>

using std::size_t;

namespace {
    template<typename T>
    void delete_and_null_array(T*& ptr)
    {
        delete[] ptr;
        ptr = NULL;
    }

    int bpstereo_vec_min(short *in, int len)
    {
        int min_pos = 0;
        short min_val = in[0];
        for (int i= 1; i<len; i++)
        {
            if (in[i]<min_val)
            {
                min_val = in[i];
                min_pos = i;
            }
        }
        return min_pos;
    }

    void bpstereo_normalize(short *in, int len)
    {
        int val=0;
        for (int i=0;i<len;i++)
            val += in[i];
        val/=len;
        for (int i=0;i<len;i++)
            in[i] -= val;
    }
    


    /*
     * Select the len smallest costs and store them and the corresponding disparities in cost and disp in increasing order.
     * TODO This could and maybe should be done with a partial_sort, making it O(len log(len)) instead of O(len^2)
     */
    void qx_get_first_k_element_increase(short*q1,short*q2,short*q3,short*q4,short*p1,short*p2,short*p3,short*p4,
            short*cost,short*disp,short*coarse,short*in,short*disp_in,int len,int len_in)
    {
        for (int i=0;i<len;i++)
        {
            short fmin=in[i];
            int id=i;
            for (int j=i+1;j<len_in;j++)
            {
                if (in[j]<fmin)
                {
                    fmin=in[j];
                    id=j;
                }
            }
            cost[i]=coarse[id];
            disp[i]=disp_in[id];
            q1[i]=p1[id];
            q2[i]=p2[id];
            q3[i]=p3[id];
            q4[i]=p4[id];
            in[id]=in[i];
            disp_in[id]=disp_in[i];
            coarse[id]=coarse[i];
        }
    }

    void qx_get_first_k_element_increase_special(short *cost, short *disp, short *in, short *in_backup, short *disp_in, int len, int len_in)
    {
        /* 
         * Set disp_in to identity 
         * This is a parameter so that it can be preallocated.
         */
        for (int j=0;j<len_in;j++) 
            disp_in[j]=j;

        int nr_local_minimum=0;

        /* 
         * Find local minima in the disparity cost.
         * FIXME is it correct that the boundaries (i=0, i=len_in-1) are ignored?
         * If there are not enough minima, they will be included in the second loop
         * if they are small enough.
         */
        for (int i=1;i<len_in-1;i++)
        {
            if (nr_local_minimum < len && in_backup[i] < in_backup[i-1] && in_backup[i] < in_backup[i+1] )
            {
                /* Local minimum at disparity i, copy to cost and disp */
                cost[nr_local_minimum] = in[i];
                disp[nr_local_minimum] = disp_in[i];

                /*
                 * Do as if swapping the minima to the front of the array in and disp_in,
                 * but discard the minima values. This boils down to simply copying the
                 * front elements in here.
                 * At the end, in[i] for i >= nr_local_minimum will be all values that were 
                 * not minima.
                 */
                in[i]=in[nr_local_minimum];
                disp_in[i]=disp_in[nr_local_minimum];

                /* Increase minima count */
                nr_local_minimum++;
            }
        }

        /* 
         * Now fill with remaining len-nr_local_minimum smallest elements.
         * FIXME
         * This is quadratic in len, which might or might not be bad, but 
         * certainly isn't asymptotically optimal. 
         * It could be done by nth_element and partition in linear (in len) time.
         */
        for (int i=nr_local_minimum;i<len;i++)
        {
            short fmin=in[i];
            int id=i;
            for (int j=i+1;j<len_in;j++)
            {
                if (in[j]<fmin)
                {
                    fmin=in[j];
                    id=j;
                }
            }
            cost[i]=fmin;
            disp[i]=disp_in[id];

            /* Again, "swap" element to the front */
            in[id]=in[i];
            disp_in[id]=disp_in[i];
        }
    }

}

short * qx_csbp_base::get_messages(int scale, int x, int y, Neighbor nb)
{
    const int nbidx = static_cast<int>(nb);
    const int w = m_w_pyramid[scale];
    const int nr_plane = m_max_nr_plane_pyramid[scale];
    const int idx = ((y*w + x)*NrNeighbors + nbidx)*nr_plane;
    return &m_message[idx];
}

#if 0
const unsigned char * qx_csbp_base::get_image_pixel(const unsigned char *img, unsigned x, unsigned y) const
{
    return img + 3*(y * m_w + x);
}
#endif
    
short * qx_csbp_base::get_selected_disparities(int scale, int x, int y)
{
    const int w = m_w_pyramid[scale];
    const int nr_plane = m_max_nr_plane_pyramid[scale];
    const int idx = (y*w + x)*nr_plane;
    return &m_selected_disparity[idx];
}

short * qx_csbp_base::get_selected_data_cost(int scale, int x, int y)
{
    const int w = m_w_pyramid[scale];
    const int nr_plane = m_max_nr_plane_pyramid[scale];
    const int idx = (y*w + x)*nr_plane;
    return &m_data_cost_selected[idx];
}

short * qx_csbp_base::get_data_cost(int scale, int x, int y)
{
    /* The data cost array is indexed with m_max_nr_plane_pyramid[scale+1] */
    const int w = m_w_pyramid[scale];
    const int nr_plane = m_max_nr_plane_pyramid[scale+1];
    const int idx = (y*w + x)*nr_plane;
    return &m_data_cost[idx];
}

qx_csbp_base::integral_image_element_type *qx_csbp_base::get_integral_data_costs(int x, int y)
{
    const size_t idx = x * m_nr_planes;
    return m_integral_data_cost_volume_row_ptr[y] + idx;
}
 
const qx_csbp_base::integral_image_element_type *qx_csbp_base::get_integral_data_costs(int x, int y) const
{
    const size_t idx = x * m_nr_planes;
    return m_integral_data_cost_volume_row_ptr[y] + idx;
}

short qx_csbp_base::get_data_cost_for_pixel(int scale, int x, int y, int disp) const
{
    const int xlo = x << scale;
    const int xhi = (x+1) << scale;
    const int ylo = y << scale;
    const int yhi = (y+1) << scale;

    return static_cast<short>( (
          get_integral_data_cost(xhi, yhi, disp)
        - get_integral_data_cost(xlo, yhi, disp)
        - get_integral_data_cost(xhi, ylo, disp)
        + get_integral_data_cost(xlo, ylo, disp) ) );
}

size_t qx_get_number_of_scales(int nr_scales, int nr_planes)
{
    int log_nr_planes = static_cast<int>(log((double)nr_planes)/log(2.0));
    return min(nr_scales, log_nr_planes);
}

qx_csbp_base::qx_csbp_base(int h, int w, const options& opt, int *iterations)
    : m_integral_data_cost_volume((w+1)*(h+1)*opt.nr_planes),
      m_integral_data_cost_volume_row_ptr(h+1),
      m_nr_scale(qx_get_number_of_scales(opt.nr_scales, opt.nr_planes)),
      m_iteration(m_nr_scale),
      m_data_cost(2*w*h*opt.nr_planes_base_level),
      m_data_cost_selected(w*h*opt.nr_planes_base_level),
      m_selected_disparity(w*h*opt.nr_planes_base_level),
      m_max_nr_plane_pyramid(m_nr_scale),
      m_w_pyramid(m_nr_scale),
      m_h_pyramid(m_nr_scale),
      m_w(w),
      m_h(h),
      m_nr_planes(opt.nr_planes),
      m_cost_discontinuity_single_jump(opt.cost_discontinuity_single_jump),
      m_cost_max_discontinuity(opt.max_nr_jump * m_cost_discontinuity_single_jump * m_nr_planes / 16.0f),
      m_cost_max_data_term(opt.cost_max_data_term),
      m_temp(m_nr_planes),
      m_temp_2(m_nr_planes),
      m_temp_3(m_nr_planes),
      m_message(w*h*NrNeighbors*opt.nr_planes_base_level)
{
    /* 
     * If number of iterations per scale weren't specified, use defaults for each scale 
     */
    if(!iterations)
    {
        switch(m_nr_scale)
        {
            default: for(int i=m_nr_scale-1; i>5; --i) m_iteration[i] = Default_Nr_Iter5;
            case 5: m_iteration[4] = Default_Nr_Iter4;
            case 4: m_iteration[3] = Default_Nr_Iter3;
            case 3: m_iteration[2] = Default_Nr_Iter2;
            case 2: m_iteration[1] = Default_Nr_Iter1;
            case 1: m_iteration[0] = Default_Nr_Iter0;
            case 0: ;
        }
    }
    else
    {
        memcpy(m_iteration, iterations, m_nr_scale * sizeof(iterations[0]));
    }

    /* 
     * Build the pyramid information
     * Each coarser level has half the width and size of the next finer level,
     * but twice the number of disparities to consider.
     * The base level 
     */
    m_max_nr_plane_pyramid[0] = opt.nr_planes_base_level;
    m_w_pyramid[0] = m_w;
    m_h_pyramid[0] = m_h;

    for (int i=1;i<m_nr_scale;++i)
    {
        if(m_max_nr_plane_pyramid[i-1] > std::numeric_limits<int>::max()/2)
            throw std::runtime_error("Overflow in pyramid computation");

        /* 
         * Halve and truncate. This way, computing base positions is simply done by shifting,
         * without any bounds checking on the finest scale.
         */
        m_max_nr_plane_pyramid[i] = m_max_nr_plane_pyramid[i-1]<<1;
        m_h_pyramid[i] = (m_h_pyramid[i-1])>>1;
        m_w_pyramid[i] = (m_w_pyramid[i-1])>>1;
    }

    /*
     * Compute the pointers into the cost volume
     */
    for(int y=0; y<=m_h; ++y)
    {
        m_integral_data_cost_volume_row_ptr[y] = &m_integral_data_cost_volume[y*(m_w+1)*m_nr_planes];
    }
}

short * qx_csbp_base::compute_disparity()
{
    /* 
     * Zero out arrays.
     */
    fill(m_message.begin(), m_message.end(), 0);
    fill(m_data_cost.begin(), m_data_cost.end(), 0);
    fill(m_data_cost_selected.begin(), m_data_cost_selected.end(), 0);
    fill(m_integral_data_cost_volume.begin(), m_integral_data_cost_volume.end(), 0);

    /*
     * Compute the cost volume
     * This is implemented in subclasses.
     */
    build_integral_data_cost_volume();

    /* Hierarchical approach, iterate from coarse to fine */
    for (int scale=m_nr_scale-1; scale>=0; scale--)
    {
        /* Compute the data costs */
        if (scale==(m_nr_scale-1))
        {
            /* 
             * Compute the per-pixel data cost.
             * The cost is actually computed for m_max_nr_plane_pyramid[scale] disparities, of which
             * m_nr_planes are then selected as follows:
             *  1. up to m_nr_planes local minima are selected, discarding later (larger) disparities if there are too many.
             *  2. the rest is filled with the remaining lowest cost elements
             */
            compute_data_cost_init(scale);
        }
        else
        {
            compute_data_cost(scale);
            init_message(scale);
        }

        /* Propagate messages m_iteration[scale] times */
        for (int j=0;j<m_iteration[scale];j++)
        {
            compute_message(scale);
        }
    }

    /* We reuse the data cost array? */
    short *disp = &m_data_cost[0]; 
    compute_disparity(disp,0);
    return disp;
}

void qx_csbp_base::init_message(int scale_index)
{
    const int h = m_h_pyramid[scale_index];
    const int w = m_w_pyramid[scale_index];
    const int nr_plane = m_max_nr_plane_pyramid[scale_index];

    int h2 = m_h_pyramid[scale_index+1];
    int w2 = m_w_pyramid[scale_index+1];
    int h2_ = h2-1;
    int w2_ = w2-1;
    int nr_plane2 = m_max_nr_plane_pyramid[scale_index+1];

    /*
     * Loop over the image (backwards... any reason?)
     */
    for (int y=h-1;y>=0;y--)
    {
        // FIXME: proper rounding when computing the pyramid sizes (i.e. w[i+1] = (w[i]+1)/2) should make this test unnecessary
        // Rounding down only makes it (very slightly) easier to integrate the data cost
        int y2 = min(h2_,(y>>1));

        for (int x=w-1;x>=0;x--)
        {
            int x2 = min(w2_,(x>>1));
    
	    /* 
             * Copy the current disparities at pixel x,y into m_temp
             */
            memcpy(&m_temp[0], get_selected_disparities(scale_index+1, x2, y2), nr_plane2 * sizeof(m_temp[0]));

            /*
             * Get the messages from the four neighbors.
             * For every pixel x,y and neighbor n,
             * there are nr_plane entries in m_message.
             *
             * So considering image coordinates which start at the top left,
             *
             *    p1 is (x, y-1, South)
             *    p2 is (x-1, y, West)
             *    p3 is (x, y+1, North)
             *    p4 is (x+1, y, East)
             *
             * The messages p21, p22, p23, and p24 are the same on the next
             * smaller level (i.e. higher scale)
             */

            short *p1 = get_messages(scale_index, x, (y>0)?y-1:y, NbSouth);
            short *p2 = get_messages(scale_index, (x>0)?x-1:x, y, NbEast);
            short *p3 = get_messages(scale_index, x, (y<h-1)?y+1:y, NbNorth);
            short *p4 = get_messages(scale_index, (x<w-1)?x+1:x, y, NbWest);

            short *p21 = get_messages(scale_index+1, x2, (y2>0)?y2-1:y2,  NbSouth);
            short *p22 = get_messages(scale_index+1, (x2>0)?x2-1:x2, y2,  NbEast);
            short *p23 = get_messages(scale_index+1, x2, (y2<h2-1)?y2+1:y2, NbNorth);
            short *p24 = get_messages(scale_index+1, (x2<w2-1)?x2+1:x2, y2, NbWest);

            short *data_cost = get_data_cost(scale_index, x, y);

            /*
             * fill m_temp_2 with the sum of the data cost and the messages
             */
            for (int d=0;d<nr_plane2;d++)
            {
                m_temp_2[d]=data_cost[d]+p21[d]+p22[d]+p23[d]+p24[d];
            }

            short *data_cost_selected = get_selected_data_cost(scale_index, x, y);
            short *disparity_selected = get_selected_disparities(scale_index, x, y);

            /* Select the nr_plane "best" disparities and corresponding messages */
            qx_get_first_k_element_increase(p1,p2,p3,p4,p21,p22,p23,p24,data_cost_selected,disparity_selected,data_cost,
                                            &m_temp_2[0],&m_temp[0],nr_plane,m_max_nr_plane_pyramid[scale_index+1]);
        }
    }
}



void qx_csbp_base::compute_data_cost_init(int scale)
{
    const int h = m_h_pyramid[scale];
    const int w = m_w_pyramid[scale];
    const int nr_plane = m_max_nr_plane_pyramid[scale];

    /* Loop over the image */
    for (int y=0;y<h;y++)
    {
        for (int x=0;x<w;x++)
        {
            short *selected_disparity = get_selected_disparities(scale, x, y);
            short *data_cost = get_selected_data_cost(scale, x, y);

            /* Using m_temp instead of data_cost directly is crucial, some function later seems to depend on it
             * TODO refactor
             */
            /*
             * Loop over the base resolution rectangle defined by [x,x+1] x [y,y+1]
             * This integrates the data cost over the rectangle defined by this (super-)pixel
             */
            for (int d=0;d<m_nr_planes;d++)
            {
                m_temp[d] = get_data_cost_for_pixel(scale, x, y, d);
            }

            /* Copy m_temp to m_temp_3 */
            m_temp_3 = m_temp;

            /* Find k local minima in the data costs */
            qx_get_first_k_element_increase_special(data_cost, selected_disparity, &m_temp[0], &m_temp_3[0], &m_temp_2[0], nr_plane, m_nr_planes);
        }
    }
}

void qx_csbp_base::compute_data_cost(int scale)
{
    const int nr_plane2 = m_max_nr_plane_pyramid[scale+1];
    const int h2_ = m_h_pyramid[scale+1]-1;
    const int w2_ = m_w_pyramid[scale+1]-1;
    const int h = m_h_pyramid[scale];
    const int w = m_w_pyramid[scale];

    assert(((w-1)<<scale) < m_w);
    assert(((h-1)<<scale) < m_h);

    for (int y=0;y<h;y++)
    {
        const int y2 = min(h2_,(y>>1));

        for (int x=0;x<w;x++)
        {
            const int x2 = min(w2_,(x>>1));

            short *selected_disparity = get_selected_disparities(scale+1, x2, y2);
            short *data_cost = get_data_cost(scale, x, y);

            /*
             * Initialize data costs with 0
             */

            /* 
             * Integrate per pixel cost over the image rectangle defined by 
             * the (super-)pixel x,y
             */
            for (int di=0;di<nr_plane2;di++)
            {
                const int d = selected_disparity[di];

                /* 
                 * If the right pixel is outside the image, set the cost to cost_max.
                 * Otherwise, set the cost based on a pixel difference metric.
                 */
                m_temp[di] = get_data_cost_for_pixel(scale, x, y, d);
            }

            memcpy(data_cost, &m_temp[0], nr_plane2 * sizeof(data_cost[0]));
        }
    }
}

/* This is one of the few bottlenecks in the computation. The double loop over nr_plane can be costly, and it is called very often. */
void qx_csbp_base::compute_message_per_pixel_per_neighbor(short *comp_func_sub,short minimum,short *disp_left,short *disp_right,int nr_plane,int scale)
{
    for (int d=0;d<nr_plane;d++)
    {
        short cost_min=minimum+m_cost_max_discontinuity;
        for (int i=0;i<nr_plane;i++)
        {
            cost_min=min<short>(cost_min,comp_func_sub[i]+m_cost_discontinuity_single_jump*abs(disp_left[i]-disp_right[d]));
        }
        m_temp[d]=cost_min;
    }
    memcpy(comp_func_sub,&m_temp[0],sizeof(comp_func_sub[0])*nr_plane);
    bpstereo_normalize(comp_func_sub,nr_plane);
}

void qx_csbp_base::compute_message_per_pixel(short*c0,short *p0,short *p1,short *p2,short *p3,short *p4,short*d0,short*d1,short*d2,
                                        short*d3,short*d4,int y,int x,int nr_plane,int scale,int &count)
{
    short minimum[4] = { 30000, 30000, 30000, 30000 };
    short *p0u = p0;
    short *p0l = &(p0[nr_plane]);
    short *p0d = &(p0[nr_plane+nr_plane]);
    short *p0r = &(p0[nr_plane+nr_plane+nr_plane]);
    count++;
    for (int d=0;d<nr_plane;d++)
    {
        p0u[d] = c0[d] + p2[d] + p3[d] + p4[d];
        p0l[d] = c0[d] + p1[d] + p3[d] + p4[d];
        p0d[d] = c0[d] + p1[d] + p2[d] + p4[d];
        p0r[d] = c0[d] + p1[d] + p2[d] + p3[d];
        if (p0u[d]<minimum[0]) minimum[0] = p0u[d];
        if (p0l[d]<minimum[1]) minimum[1] = p0l[d];
        if (p0d[d]<minimum[2]) minimum[2] = p0d[d];
        if (p0r[d]<minimum[3]) minimum[3] = p0r[d];
        //m_comp_func_sub_prev[d]=p1[d]+p2[d]+p3[d]+p4[d];
    }
    compute_message_per_pixel_per_neighbor(p0u,minimum[0],d0,d1,nr_plane,scale);
    compute_message_per_pixel_per_neighbor(p0l,minimum[1],d0,d2,nr_plane,scale);
    compute_message_per_pixel_per_neighbor(p0d,minimum[2],d0,d3,nr_plane,scale);
    compute_message_per_pixel_per_neighbor(p0r,minimum[3],d0,d4,nr_plane,scale);
}

void qx_csbp_base::compute_message(int scale)
{
    const int h = get_height_at_scale(scale);
    const int w = get_width_at_scale(scale);
    const int nr_plane = get_nr_plane_at_scale(scale);

    const int yy=h-1;
    const int xx=w-1;

    int count = 0;

    /* 
     * Iterate over the image in a chessboard manner:
     * First the even fields, then the odd fields.
     */
    for (int i=0;i<2;i++)
    {
        for (int y=1;y<yy;y++)
        {
            for (int x=xx-1+(y+i)%2;x>=1;x-=2) //for(x=(y+i)%2+1;x<xx;x+=2)
            {
                short *c0 = get_selected_data_cost(scale, x, y);

                short *p0 = get_messages(scale, x, y, NbNorth);
                short *p1 = get_messages(scale, x, y-1, NbSouth);
                short *p2 = get_messages(scale, x-1, y, NbEast);
                short *p3 = get_messages(scale, x, y+1, NbNorth);
                short *p4 = get_messages(scale, x+1, y, NbWest);

                short *d0 = get_selected_disparities(scale, x, y);
                short *d1 = get_selected_disparities(scale, x, y-1);
                short *d2 = get_selected_disparities(scale, x-1, y);
                short *d3 = get_selected_disparities(scale, x, y+1);
                short *d4 = get_selected_disparities(scale, x+1, y);

                compute_message_per_pixel(c0,p0,p1,p2,p3,p4,d0,d1,d2,d3,d4,y,x,nr_plane,scale,count);
            }
        }
    }
}

int qx_csbp_base::compute_disparity(short *disparity, int scale)
{
    const int h=m_h_pyramid[scale];
    const int w=m_w_pyramid[scale];

    // reset to zero the disparity
    memset(disparity,0,sizeof(short)*h*w);

    //const int nr_plane = m_max_nr_plane_pyramid[scale];
    const int nr_plane = get_nr_plane_at_scale(scale);

    for (int y=1;y<h-1;y++)
    {
        for (int x=1;x<w-1;x++)
        {
            short *c0 = get_selected_data_cost(scale, x, y);

            short *p1 = get_messages(scale, x, y-1, NbSouth);
            short *p2 = get_messages(scale, x-1, y, NbEast);
            short *p3 = get_messages(scale, x, y+1, NbNorth);
            short *p4 = get_messages(scale, x+1, y, NbWest);

            for (int d=0;d<nr_plane;d++)
            {
                m_temp[d]=c0[d]+p1[d]+p2[d]+p3[d]+p4[d];
            }

            short *selected_disparity = get_selected_disparities(scale, x, y);
            int d0 = bpstereo_vec_min(&m_temp[0], nr_plane);
            disparity[y*w+x] = selected_disparity[d0];
        }
    }

    // set the horizontal borders of the disparity
    memcpy(&(disparity[(h-1)*w]),&(disparity[(h-2)*w]),sizeof(short)*w);
    memcpy(&(disparity[0]),&(disparity[w]),sizeof(short)*w);

    // set the vertical borders of the disparity
    for (int y=0;y<h;y++)
    {
        disparity[y*w+0]  = disparity[y*w+1];
        disparity[y*w+w-1]= disparity[y*w+w-2];
    }
    return 0;
}

void qx_csbp_pixelwise::build_integral_data_cost_volume()
{
    const int w = get_base_width();
    const int h = get_base_height();
    const int nr_planes = get_nr_planes();
    const short max_cost = get_cost_max_data_term();

    if(h == 0 || w == 0)
        return;

    for(int y=0; y<h; ++y)
    {
        for(int x=0; x<w; ++x)
        {
            set_left_pixel(x,y);
            const int max_valid_d = min(x+1,nr_planes); // before we hit the image boundary
            /* Get integral data costs for pixel to the left, upper left, and above */
            const integral_image_element_type *il = get_integral_data_costs(x, y+1);
            const integral_image_element_type *iul = get_integral_data_costs(x, y);
            const integral_image_element_type *iu = get_integral_data_costs(x+1, y);
            /* Get integral data costs to fill in */
            integral_image_element_type *ic = get_integral_data_costs(x+1, y+1);

            /* 
             * The integral cost for pixel (x,y) 
             * I(x,y) is I(x-1,y) + I(x,y-1) - I(x-1,y-1)
             */
            int d = 0;
            for (;d<max_valid_d;++d)
            {
                set_disparity(d);
                ic[d] = il[d] + iu[d] - iul[d] + min(max_cost,  cost_for_pixel_pair());
            }
            for(;d<nr_planes;d++)
            {
                ic[d] = il[d] + iu[d] - iul[d] + max_cost;
            }
        }
    }
}


