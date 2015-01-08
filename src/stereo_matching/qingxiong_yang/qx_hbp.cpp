#include "qx_basic.hpp"
#include "qx_hbp.hpp"

namespace
{
    void bpstereo_vec_min(short *in, int len, int &min_pos)
    {
        short min_val=in[0];
        min_pos= 0;
        for ( int i= 1; i<len; i++)
            if (in[i]<min_val)
            {
                min_val=in[i];
                min_pos= i;
            }
    }
    void bpstereo_normalize(short *in, int len)
    {
        int val=0;
        for (int i=0;i<len;i++)
            val+=in[i];
        val/=(short)len;
        for (int i=0;i<len;i++)
            in[i]-=val;
    }
}

qx_hbp::~qx_hbp()
{
}

qx_hbp::qx_hbp(int h,int w, const options& opts, int *iterations)
    : m_h(h),
      m_w(w),
      m_nr_plane(opts.nr_planes),
      m_nr_scale(opts.nr_scales),
      m_iteration(m_nr_scale),
      m_cost_discontinuity_single_jump(opts.cost_discontinuity_single_jump),
      m_cost_max_discontinuity(short(opts.max_nr_jump/16.0f*m_cost_discontinuity_single_jump*m_nr_plane)),
      m_cost_max_data_term(opts.cost_max_data_term),
      m_data_cost_pyramid(m_nr_scale),
      m_temp_f(m_nr_plane),
      m_temp_s(m_nr_plane),
      m_temp_s2(m_nr_plane),
      m_zero_messages(m_nr_plane),
      m_message_pyramid(m_nr_scale),
      m_h_pyramid(m_nr_scale),
      m_w_pyramid(m_nr_scale),
      m_disparity_map_pyramid(m_nr_scale)

{
    if (iterations==NULL)
    {
        switch(m_nr_scale)
        {
            default:
                for(int i=5; i<m_nr_scale; ++i) m_iteration[i]=QX_DEF_BP_NR_ITER5;
            case 5: m_iteration[4]=QX_DEF_BP_NR_ITER4;
            case 4: m_iteration[3]=QX_DEF_BP_NR_ITER3;
            case 3: m_iteration[2]=QX_DEF_BP_NR_ITER2;
            case 2: m_iteration[1]=QX_DEF_BP_NR_ITER1;
            case 1: m_iteration[0]=QX_DEF_BP_NR_ITER0;
            case 0: ;
        }
    }
    else
    {
        memcpy(m_iteration, iterations, m_nr_scale * sizeof(iterations[0]));
    }

    for (int i=0;i<m_nr_scale;i++)
    {
        m_h_pyramid[i] = (i==0) ? m_h : (m_h_pyramid[i-1]>>1);
        m_w_pyramid[i] = (i==0) ? m_w : (m_w_pyramid[i-1]>>1);

        qx_array_3d<short>(m_h_pyramid[i], m_w_pyramid[i], m_nr_plane).swap(m_data_cost_pyramid[i]);
        qx_array_4d<short>(m_h_pyramid[i], m_w_pyramid[i], NrNeighbors, m_nr_plane).swap(m_message_pyramid[i]);
        qx_array_2d<short>(m_h_pyramid[i], m_w_pyramid[i]).swap(m_disparity_map_pyramid[i]);
    }

    for (int i=0; i<m_nr_plane; i++)
    {
        m_zero_messages[i] = 0;
    }
}

int qx_hbp::disparity(qx_array_2d<short>::type disparity, qx_array_3d<short>::const_type corr_vol)
{
    compute_data_cost_pyramid(corr_vol);
    for (int i=m_nr_scale-1;i>=0;i--)
    {
        init_message(i);
        for (int j=0;j<m_iteration[i];j++)
        {
            compute_message(i);
        }
    }
    compute_disparity(disparity);
    return 0;
}

void qx_hbp::init_message(int scale_index)
{
    const int h = m_h_pyramid[scale_index];
    const int w = m_w_pyramid[scale_index];
    qx_array_4d<short>::type message = m_message_pyramid[scale_index];

    if (scale_index==(m_nr_scale-1))
    {
        memset(message[0][0][0],0,sizeof(short)*h*w*NrNeighbors*m_nr_plane);
    }
    else
    {
        const int h2 = m_h_pyramid[scale_index+1];
        const int w2 = m_w_pyramid[scale_index+1];
    	qx_array_4d<short>::type message2 = m_message_pyramid[scale_index+1];

        for (int y=0;y<h;y++)
        {
            int yi=min((y>>1), h2-1);
            for (int x=0;x<w;x++)
            {
                int xi=min((x>>1),w2-1);
                memcpy(message[y][x][0],
                       message2[yi][xi][0],
                       sizeof(short)*NrNeighbors*m_nr_plane);
            }
        }
    }
    return;
}
void qx_hbp::compute_data_cost_pyramid(qx_array_3d<short>::const_type corr_vol)
{
    qx_array_3d<short>::type finer;
    qx_array_3d<short>::type coarser;

    int i,j,k,d,ah,aw,bh,bw;

    /* 
     * Copy correlation volume to lowest level
     */
    memcpy(m_data_cost_pyramid[0][0][0], corr_vol[0][0], m_h*m_w*m_nr_plane*sizeof(corr_vol[0][0][0]));

    /* 
     * Threshold data cost by m_cost_max_data_term
     */

    for (i=0; i<m_h; ++i)
    {
        for(j=0; j<m_w; ++j)
        {
            for(d=0; d<m_nr_plane; ++d)
            {
                m_data_cost_pyramid[0][i][j][d]=min(m_data_cost_pyramid[0][i][j][d], m_cost_max_data_term);
            }
        }
    }

    /* 
     * Start with twice the height and width, as they will be halved
     * immediately inside the loop.
     */
    ah=m_h<<1;
    aw=m_w<<1;

    /*
     * Loop over all pixels in the pyramid, i.e. over all scales first, starting from
     * the base scale (image resolution)
     */
    for (i=0;i<m_nr_scale-1;i++)
    {
        /* 
         * ah, aw are the dimensions of the previously computed
         * scale (initially the base image costs)
         * bh, bw are the dimensions of the scale to compute, i.e.
         * bh=ah/2 and bw=aw/2
         */
        ah=ah>>1;
        aw=aw>>1;
        bh=ah>>1;
        bw=aw>>1;

        /*
         * Let a point to the costs for the previously computed
         * scale (of size ah x aw) and b point to the costs of
         * the scale to compute
         */
        finer = m_data_cost_pyramid[i];
        coarser = m_data_cost_pyramid[i+1];

        /*
         * Initialize coarser level to all zeros
         */
        memset(coarser[0][0], 0, bh*bw*m_nr_plane*sizeof(short));

        /*
         * Iterate over all pixels and disparities of a
         * Sum up the costs for all base pixels
         */
        for (j=0;j<ah;j++)
        {
            const int j2 = min((j>>1), bh-1);
            for (k=0;k<aw;k++)
            {
                const int k2 = min((k>>1), bw-1);
                for (d=0;d<m_nr_plane;d++)
                {
                    coarser[j2][k2][d] += finer[j][k][d];
                }
            }
        }
    }
}

void qx_hbp::compute_message_per_pixel_per_neighbor(qx_array_1d<short>::type comp_func_sub, short minimum)
{
    short prev;
    for (int d=1;d<m_nr_plane;d++)
    {
        prev=comp_func_sub[d-1]+m_cost_discontinuity_single_jump;
        if (prev<comp_func_sub[d])
            comp_func_sub[d]=prev;
    }
    for (int d=m_nr_plane-2;d>=0;d--)
    {
        prev=comp_func_sub[d+1]+m_cost_discontinuity_single_jump;
        if (prev<comp_func_sub[d])
            comp_func_sub[d]=prev;
    }
    minimum+=m_cost_max_discontinuity;
    for (int d=0;d<m_nr_plane;d++)
    {
        if (minimum<comp_func_sub[d])
        {
            comp_func_sub[d]=minimum;
        }
    }

    bpstereo_normalize(comp_func_sub,m_nr_plane);

    return;
}

void qx_hbp::compute_message_per_pixel(qx_array_4d<short>::type message, qx_array_3d<short>::const_type cost, int y, int x, int h, int w)
{
    const int yy=h-1;
    const int xx=w-1;

    const qx_array_1d<short>::const_type p0 = cost[y][x];
    const qx_array_1d<short>::const_type p1 = 
            (y>0) 
            ? message[y-1][x][NbSouth]
            : m_zero_messages;
    const qx_array_1d<short>::const_type p2 =
            (x>0)
            ? message[y][x-1][NbEast]
            : m_zero_messages;
    const qx_array_1d<short>::const_type p3 =
            (y<yy)
            ? message[y+1][x][NbNorth]
            : m_zero_messages;
    const qx_array_1d<short>::const_type p4 = 
            (x<xx)
            ? message[y][x+1][NbWest]
            : m_zero_messages;

    const qx_array_2d<short>::type msg = 
            message[y][x];

    short minimum[4]={30000,30000,30000,30000};
    for (int d=0;d<m_nr_plane;d++)
    {
        msg[0][d] = p0[d]+p2[d]+p3[d]+p4[d];
        msg[1][d] = p0[d]+p1[d]+p3[d]+p4[d];
        msg[2][d] = p0[d]+p1[d]+p2[d]+p4[d];
        msg[3][d] = p0[d]+p1[d]+p2[d]+p3[d];

        if (msg[0][d]<minimum[0]) minimum[0]=msg[0][d];
        if (msg[1][d]<minimum[1]) minimum[1]=msg[1][d];
        if (msg[2][d]<minimum[2]) minimum[2]=msg[2][d];
        if (msg[3][d]<minimum[3]) minimum[3]=msg[3][d];
    }

    compute_message_per_pixel_per_neighbor(msg[0],minimum[0]);
    compute_message_per_pixel_per_neighbor(msg[1],minimum[1]);
    compute_message_per_pixel_per_neighbor(msg[2],minimum[2]);
    compute_message_per_pixel_per_neighbor(msg[3],minimum[3]);

    return;
}

void qx_hbp::compute_message(int scale)
{
    const qx_array_3d<short>::const_type cost = m_data_cost_pyramid[scale];
    const qx_array_4d<short>::type message = m_message_pyramid[scale];
    const int h = m_h_pyramid[scale];
    const int w = m_w_pyramid[scale];

    const int yy=h-1;
    const int xx=w-1;
    
    for (int i=0;i<2;i++)
    {
        for (int y=0;y<=yy;y++) 
        {
            for(int x=xx-1+(y+i)%2;x>=0;x-=2)
            {
                compute_message_per_pixel(message, cost, y, x, h, w);
            }
        }
    }
}
int qx_hbp::compute_disparity(qx_array_2d<short>::type disparity,int scale)
{
    int h=m_h_pyramid[scale];
    int w=m_w_pyramid[scale];
    int yy=h-1;
    int xx=w-1;
    memset(disparity[0],0,sizeof(short)*h*w);
    int nr_planes=m_nr_plane;
    qx_array_3d<short>::const_type cost = m_data_cost_pyramid[scale];
    qx_array_4d<short>::type message = m_message_pyramid[scale];
    int d0;
    qx_array_1d<short>::const_type p0;
    qx_array_1d<short>::const_type p1;
    qx_array_1d<short>::const_type p2;
    qx_array_1d<short>::const_type p3;
    qx_array_1d<short>::const_type p4;

    for (int y=0;y<h;y++)
    {
        for (int x=0;x<w;x++)
        {
            p0=cost[y][x];
            p1=(y>0) ? message[y-1][x][NbSouth] : m_zero_messages;
            p2=(x>0) ? message[y][x-1][NbEast] : m_zero_messages;
            p3=(y<yy) ? message[y+1][x][NbNorth] : m_zero_messages;
            p4=(x<xx) ? message[y][x+1][NbWest] : m_zero_messages;
            for (int d=0;d<nr_planes;d++) m_temp_f[d]=p0[d]+p1[d]+p2[d]+p3[d]+p4[d];
            bpstereo_vec_min(m_temp_f,nr_planes,d0);
            disparity[y][x]=short(d0);
        }
    }
    memcpy(disparity[h-1],disparity[h-2],sizeof(short)*w);
    memcpy(disparity[0],disparity[1],sizeof(short)*w);
    for (int y=0;y<h;y++)
    {
        disparity[y][0]=disparity[y][1];
        disparity[y][w-1]=disparity[y][w-2];
    }
    return(0);
}

