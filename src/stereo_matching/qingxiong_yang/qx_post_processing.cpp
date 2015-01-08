#include "qx_basic.hpp"
#include "qx_post_processing.hpp"

namespace
{
    ///*bilateral filtering*/
    unsigned char euro_dist_rgb_max(const unsigned char *a,const unsigned char *b)
    {
        unsigned char x,y,z;
        x=abs(a[0]-b[0]);
        y=abs(a[1]-b[1]);
        z=abs(a[2]-b[2]);
        return(max(max(x,y),z));
    }

    qx_array_1d<double> get_color_weighted_table(double dist_color,int len)
    {
        qx_array_1d<double> table_color(len);

        for (int y=0;y<len;y++)
        {
            table_color[y] = std::exp(-double(y*y)/(2*dist_color*dist_color));
        }
        
        return table_color;
    }

    qx_array_2d<double> get_space_weighted_filter(int win_size,double dist_space)
    {
        const int half = (win_size >> 1);
        qx_array_2d<double> table_space(half+1, half+1);

        for(int y=0; y<=half; y++)
        {
            for(int x=0; x<=half; x++)
            {
                const double yy = double(y)*double(y);
                const double xx = double(x)*double(x);
                table_space[y][x] = std::exp( -std::sqrt(yy+xx)/dist_space);
            }
        }

        return table_space;
    }

    template<typename T>
    int vec_min_pos(const T *in, int len)
    {
        T min_val=in[0];
        int min_pos=0;
        for (int i=1;i<len;i++) 
        {
            if (in[i]<min_val)
            {
                min_val=in[i];
                min_pos= i;
            }
        }
        return min_pos;
    }

}

qx_disparity_map_bf_refinement::qx_disparity_map_bf_refinement(int h,int w,int nr_plane,int radius)
    : m_h(h),
      m_w(w),
      m_nr_plane(nr_plane),
      m_radius(radius),
      m_table_color(get_color_weighted_table(QX_DEF_BF_FILTER_SIGMA_RANGE,256)),
      m_table_space(get_space_weighted_filter(m_radius*2+1, m_radius+1))
{
}

int qx_disparity_map_bf_refinement::disparity_refinement(short * const *disparity,
                                                         const unsigned char * const * const *image,
                                                         int nr_iter,
                                                         float edge_threshold,
                                                         float max_disc_threshold)
{
    //image_display(disparity,m_h,m_w);
    short edge_disc=max<short>(1,short(m_nr_plane*edge_threshold+0.5));
    short max_disc=short(m_nr_plane*max_disc_threshold+0.5);
    short dp_[5],dp[5];
    int nr_dp;
    for (int ii=0;ii<nr_iter;ii++)
    {
        for (int y=1;y<m_h-1;y++)
        {
            for (int x=1;x<m_w-1;x++)
            {
                const short * const up=&(disparity[y-1][x]);
                const short * const down=&(disparity[y+1][x]);
                const short * const curr=&(disparity[y][x]);
                const short * const left=&(disparity[y][x-1]);
                const short * const right=&(disparity[y][x+1]);

                if (abs(*up-*curr)>=edge_disc||abs(*left-*curr)>=edge_disc||abs(*down-*curr)>=edge_disc||abs(*right-*curr)>=edge_disc)
                {
                    dp_[0]=*curr;
                    dp_[1]=*up;
                    dp_[2]=*left;
                    dp_[3]=*down;
                    dp_[4]=*right;

                    nr_dp=1;
                    dp[0]=dp_[0];
                    for (int i=1;i<5;i++)
                    {
                        for (int j=0;j<nr_dp;j++)
                        {
                            if (dp_[i]!=dp[j])
                            {
                                dp[nr_dp++]=dp_[i];
                                j=nr_dp;
                            }
                        }
                    }
                    disparity_refinement(disparity,image,dp,nr_dp,max_disc,y,x,m_radius,m_h,m_w);
                }
            }
        }
        for (int y=m_h-2;y>0;y--)
        {
            for (int x=m_w-2;x>0;x--)
            {
                const short * const up=&(disparity[y-1][x]);
                const short * const down=&(disparity[y+1][x]);
                const short * const curr=&(disparity[y][x]);
                const short * const left=&(disparity[y][x-1]);
                const short * const right=&(disparity[y][x+1]);

                if (abs(*up-*curr)>=edge_disc||abs(*left-*curr)>=edge_disc||abs(*down-*curr)>=edge_disc||abs(*right-*curr)>=edge_disc)
                {
                    dp_[0]=*curr;
                    dp_[1]=*up;
                    dp_[2]=*left;
                    dp_[3]=*down;
                    dp_[4]=*right;

                    nr_dp=1;
                    dp[0]=dp_[0];
                    for (int i=1;i<5;i++)
                    {
                        for (int j=0;j<nr_dp;j++)
                        {
                            if (dp_[i]!=dp[j])
                            {
                                dp[nr_dp++]=dp_[i];
                                j=nr_dp;
                            }
                        }
                    }
                    disparity_refinement(disparity,image,dp,nr_dp,max_disc,y,x,m_radius,m_h,m_w);
                }
            }
        }
    }
    return(0);
}

void qx_disparity_map_bf_refinement::disparity_refinement(short * const *disparity, const unsigned char * const * const * image, const short *dp,int nr_dp,short max_disc,int y,int x,int radius,int h,int w)
{
    const int ymin=max(0,y-radius);
    const int xmin=max(0,x-radius);
    const int ymax=min(h-1,y+radius);
    const int xmax=min(w-1,x+radius);

    double weight_sum = 0;
    double cost[5]={0,0,0,0,0};

    const unsigned char *ic = image[y][x];

    for (int yi=ymin;yi<=ymax;yi++)
    {
        short *disp_y=disparity[yi];
        for (int xi=xmin;xi<=xmax;xi++)
        {
            const char dist_rgb = euro_dist_rgb_max(image[yi][xi],ic);
            const double weight=m_table_color[dist_rgb]*m_table_space[abs(y-yi)][abs(x-xi)];

            weight_sum+=weight;

            for (int i=0;i<nr_dp;i++)
            {
                cost[i]+=min<short>(max_disc,abs(disp_y[xi]-dp[i]))*weight;
            }
        }
    }
    for (int i=0;i<nr_dp;i++) cost[i]/=weight_sum;
    const int id = vec_min_pos(cost,nr_dp);

    disparity[y][x]=dp[id];
}
