#include "qx_basic.hpp"
#include "qx_stereo_corr_vol_cpu.hpp"
#include <algorithm>
#include <cmath>

namespace
{
    short rgb_distance(const unsigned char *p1, const unsigned char *p2)
    {
        const float tr=0.299f, tg=0.587f, tb=0.114f;
        return static_cast<unsigned char>(std::min(tr*abs(p1[0]-p2[0])+tg*abs(p1[1]-p2[1])+tb*abs(p1[2]-p2[2])+0.5f, 255.0f));
    }
}

qx_stereo_corr_vol_cpu::qx_stereo_corr_vol_cpu(const int h, const int w, const int nr_planes, pixel_matching_type method)
    : m_h(h),
      m_w(w),
      m_nr_planes(nr_planes),
      m_method(method)
{
    // nothing to do here
    return;
}

qx_stereo_corr_vol_cpu::~qx_stereo_corr_vol_cpu()
{
    return;
}

void qx_stereo_corr_vol_cpu::corr_vol_zncc(corr_vol_t corr_vol, const_image_t image_left, const_image_t image_right, int winradius) const
{
    /* Straight-forward implementation, not necessarily optimized */
    for (int y=0;y<m_h;y++)
    {
        for (int x=0;x<m_w;x++)
        {
            for (int d=0;d<m_nr_planes;d++)
            {
                if(x >= d)
                {
                    float mean_left[3] = {0,0,0};
                    float mean_right[3] = {0,0,0};

                    unsigned valid = 0;

                    for(int i=-winradius; i<=winradius; ++i)
                    {
                        for(int j=-winradius; j<=winradius; ++j)
                        {
                            if(y >= -i && y < m_h-i && x >= d-j && x < m_w-j)
                            {
                                mean_left[0] += image_left[y+i][x+j][0];
                                mean_left[1] += image_left[y+i][x+j][1];
                                mean_left[2] += image_left[y+i][x+j][2];
                                mean_right[0] += image_left[y+i][x-d+j][0];
                                mean_right[1] += image_left[y+i][x-d+j][1];
                                mean_right[2] += image_left[y+i][x-d+j][2];
                                ++valid;
                            }
                        }
                    }

                    mean_left[0] /= valid;
                    mean_left[1] /= valid;
                    mean_left[2] /= valid;
                    mean_right[0] /= valid;
                    mean_right[1] /= valid;
                    mean_right[2] /= valid;

                    float norm_left[3] = {};
                    float norm_right[3] = {};
                    float correlation[3] = {};

                    for(int i=-winradius; i<=winradius; ++i)
                    {
                        for(int j=-winradius; j<=winradius; ++j)
                        {
                            if(y >= -i && y < m_h-i && x >= d-j && x < m_w-j)
                            {
                                const float tmp_left[3] = { 
                                    image_left[y][x][0] - mean_left[0],
                                    image_left[y][x][1] - mean_left[1],
                                    image_left[y][x][2] - mean_left[2]
                                };

                                const float tmp_right[3] = {
                                    image_right[y][x][0] - mean_right[0],
                                    image_right[y][x][1] - mean_right[1],
                                    image_right[y][x][2] - mean_right[2]
                                };

                                norm_left[0] += tmp_left[0] * tmp_left[0];
                                norm_left[1] += tmp_left[1] * tmp_left[1];
                                norm_left[2] += tmp_left[2] * tmp_left[2];
                                norm_right[0] += tmp_right[0] * tmp_right[0];
                                norm_right[1] += tmp_right[1] * tmp_right[1];
                                norm_right[2] += tmp_right[2] * tmp_right[2];

                                correlation[0] += tmp_left[0] * tmp_right[0];
                                correlation[1] += tmp_left[1] * tmp_right[1];
                                correlation[2] += tmp_left[2] * tmp_right[2];
                            }
                        }
                    }

                    correlation[0] /= std::sqrt(norm_left[0] * norm_right[0]);
                    correlation[1] /= std::sqrt(norm_left[1] * norm_right[1]);
                    correlation[2] /= std::sqrt(norm_left[2] * norm_right[2]);

                    /* Do color weighted sum */
                    {
                        const float tr=0.299f, tg=0.587f, tb=0.114f;
                        const float correlation_sum = tr * correlation[0] + tg * correlation[1] + tb*correlation[2];
                        corr_vol[y][x][d] = short(std::min(255.0f * (0.5f - 0.5f*correlation_sum), 255.0f));
                    }
                }
                else
                {
                    corr_vol[y][x][d] = 255;
                }
            }
        }
    }

}

void qx_stereo_corr_vol_cpu::aggregate_color_weighted(corr_vol_t cost_out, const_corr_vol_t cost_in, const_image_t image_left, const_image_t image_right, int winradius) const
{
    const float beta = 4.0f;
    const float gamma = 2.0f;

    /* Precompute exponentials */
    qx_array_1d<float> color_exp_lut(256);
    qx_array_1d<float> position_exp_lut(2*winradius+1);

    for(int i=0; i<256; ++i)
        color_exp_lut[i] = std::exp(-i/beta);

    for(int i=-winradius; i<=winradius; ++i)
        position_exp_lut[i+winradius] = std::exp(-std::sqrt(1.0f*i*i)/gamma);

    /* Straight-forward implementation, not necessarily optimized */
    for (int y=0;y<m_h;y++)
    {
        for (int x=0;x<m_w;x++)
        {
            for (int d=0;d<m_nr_planes;d++)
            {
                if(x >= d)
                {
                    float num = 0;
                    float den = 0;
                    for(int i=-winradius; i<=winradius; ++i)
                    {
                        // In Qingxiong Yang's paper as well as in Kweon and Yoon's paper it looks like
                        // we should look at all pairs of pixels in the left and right sub-window.
                        // A closer look reveals that this is probably not true.
                        for(int j=-winradius; j<=winradius; ++j)
                        {
                            if(y >= -i && y < m_h-i && x >= d-j && x < m_w-j)
                            {
                                const short cost = cost_in[y+i][x+j][d];
                                const float w_col_left = color_exp_lut[ rgb_distance(image_left[y][x], image_left[y+i][x+j]) ];
                                const float w_col_right = color_exp_lut[ rgb_distance(image_right[y][x], image_right[y+i][x-d+j]) ];
                                const float w_pos_left = position_exp_lut[i+winradius] * position_exp_lut[j+winradius];
                                const float w_pos_right = position_exp_lut[i+winradius] * position_exp_lut[j+winradius]; // yes, this is the same as above

                                num += w_col_left*w_col_right*w_pos_left*w_pos_right*cost;
                                den += w_col_left*w_col_right*w_pos_left*w_pos_right;
                            }
                        }
                    }
                    cost_out[y][x][d] = short(std::min(num/den+0.5f, 255.0f));
                }
                else
                {
                    cost_out[y][x][d] = 255;
                }
            }
        }
    }
    return;
}

void qx_stereo_corr_vol_cpu::corr_vol(corr_vol_t corr_vol,
                                     const_image_t image_left,
                                     const_image_t image_right,
                                     const bool is_left_reference) const
{
    memset(corr_vol[0][0],0,sizeof(short)*m_h*m_w*m_nr_planes);
    if (m_method==QX_DEF_STEREO_CORR_VOL_METHOD_SMOOTH_BIRCHFIELD)
    {
        if (is_left_reference)
        {
            birchfield_absolute_difference_rgb(corr_vol, image_left, image_right);
        }
        else
        {
            birchfield_absolute_difference_rgb_right_reference(corr_vol, image_left, image_right);
        }
    }
    else if (m_method==QX_DEF_STEREO_CORR_VOL_METHOD_ABSOLUTE_DIFFERENCE)
    {
        absolute_difference_rgb(corr_vol, image_left, image_right);
    }
    else
    {
        printf("qx_stereo_corr_vol_cpu::corr_vol is not yet able to perform this method!!\n");
        printf("using  birchfield_absolute_difference_rgb\n");
        if (is_left_reference)
        {
            birchfield_absolute_difference_rgb(corr_vol, image_left, image_right);
        }
        else
        {
            birchfield_absolute_difference_rgb_right_reference(corr_vol, image_left, image_right);
        }
    }

    return ;
}

void qx_stereo_corr_vol_cpu::aggregate_smooth(corr_vol_t corr_vol, int winradius) const
{
    /* Smooth iteratively with a binomial 3x3 kernel (1,2,1) x (1,2,1)^T
     * Two iterations correspond to (1,2,1) * (1,2,1) = (1,4,6,4,1)
     * Three correspond to (1,2,1) * (1,4,6,4,1) = (1,6,15,20,15,6,1)
     * and so on, so every iteration corresponds to an increase of the
     * radius by exactly one.
     * (Up to numerical rounding issues)
     */
    qx_array_1d<short> buf(m_nr_planes);

    for(int iter=0; iter<winradius; ++iter)
    {
        /* Horizontal smoothing */
        for (int y=1;y<m_h-1;y+=1)
        {
            for (int d=0;d<m_nr_planes;d+=1)
            {
                buf[d] = corr_vol[y][0][d];
            }
            for (int x=1;x<m_w-1;x+=1)
            {
                for (int d=0;d<m_nr_planes;d+=1)
                {
                    short tmp = corr_vol[y][x][d];
                    corr_vol[y][x][d] = buf[d] + 2*corr_vol[y][x][d] + corr_vol[y][x+1][d];
                    buf[d] = tmp;
                } // end of 'for each disparity'
            } // end of 'for each column'
        } // end of 'for each row'
        /* Vertical smoothing */
        for (int x=1;x<m_w-1;x+=1)
        {
            for (int d=0;d<m_nr_planes;d+=1)
            {
                buf[d] = corr_vol[0][x][d];
            }
            for (int y=1;y<m_h-1;y+=1)
            {
                for (int d=0;d<m_nr_planes;d+=1)
                {
                    short tmp = corr_vol[y][x][d];
                    corr_vol[y][x][d] = (buf[d] + 2*corr_vol[y][x][d] + corr_vol[y+1][x][d]) / 16;
                    buf[d] = tmp;
                } // end of 'for each disparity'
            } // end of 'for each column'
        } // end of 'for each row'
    }
}

void qx_stereo_corr_vol_cpu::absolute_difference_rgb(corr_vol_t corr_vol, const_image_t image_left, const_image_t image_right) const
{
    int y,x,d;
    short * const *corr_vol_y,*corr_vol_x;
    unsigned char const * const *image1_y;
    unsigned char const * image1_x;
    unsigned char const * const *image2_y;
    unsigned char const *image2_x;

    for (y=0;y<m_h;y+=1)
    {
        corr_vol_y=corr_vol[y];
        image1_y=image_left[y];
        image2_y=image_right[y];
        for (x=0;x<m_w;x+=1)
        {
            corr_vol_x=corr_vol_y[x];
            image1_x=image1_y[x];
            for (d=0;d<m_nr_planes;d+=1)
            {
                if (x-d>=0)
                {
                    image2_x=image2_y[x-d];
                    corr_vol_x[d]=
                            rgb_distance(image1_x, image2_x);
                }
                else
                {
                    corr_vol_x[d]=255;
                }
            } // end of 'for each disparity'
        } // end of 'for each column'
    } // end of 'for each row'

    return;
}

void qx_stereo_corr_vol_cpu::birchfield_absolute_difference_rgb(corr_vol_t corr_vol, const_image_t image_left, const_image_t image_right) const
{

    int y,x,d;
    for (y=0;y<m_h;y++)
    {
        for (x=0;x<m_w;x++)
        {
            for (d=0;d<m_nr_planes;d++)
            {
                if(x-d>=0)
                    corr_vol[y][x][d] = birchfield_absolute_difference_rgb_per_pixel(image_left, image_right, y, x, d);
                else
                    corr_vol[y][x][d] = 255;
            }
        }
    }
}

void qx_stereo_corr_vol_cpu::birchfield_absolute_difference_rgb_right_reference(corr_vol_t corr_vol, const_image_t image_left, const_image_t image_right) const
{
    int y,x,d;
    for (y=0;y<m_h;y++)
    {
        for (x=0;x<m_w;x++)
        {
            for (d=0;d<m_nr_planes;d++)
            {
                if(x+d<m_w)
                    corr_vol[y][x][d] = birchfield_absolute_difference_rgb_per_pixel_right_reference(image_left, image_right, y, x, d);
                else
                    corr_vol[y][x][d] = 255;
            }
        }
    }
}

short qx_stereo_corr_vol_cpu::birchfield_absolute_difference_rgb_per_pixel(const_image_t image_left, const_image_t image_right, int y, int x, int d) const
{
    const unsigned char *left[3] = {};
    const unsigned char *right[3] = {};

    left[0] = (x>0) ? image_left[y][x-1] : image_left[y][0];
    left[1] = image_left[y][x];
    left[2] = (x<m_w-1) ? image_left[y][x+1] : image_left[y][m_w-1];

    right[0] = (x>d) ? image_right[y][x-d-1] : image_right[y][0];
    right[1] = image_right[y][x-d];
    right[2] = (x<d+m_w-1) ? image_right[y][x-d+1] : image_right[y][m_w-1];

    return birchfield_absolute_difference_rgb_per_pixel(left[0], left[1], left[2], right[0], right[1], right[2]);
}

short qx_stereo_corr_vol_cpu::birchfield_absolute_difference_rgb_per_pixel_right_reference(const_image_t image_left, const_image_t image_right, int y, int x, int d) const
{
    const unsigned char *left[3] = {};
    const unsigned char *right[3] = {};

    left[0] = (x+d>0) ? image_left[y][x+d-1] : image_left[y][0];
    left[1] = image_left[y][x+d];
    left[2] = (x+d<m_w-1) ? image_left[y][x+d+1] : image_left[y][m_w-1];

    right[0] = (x>0) ? image_right[y][x-1] : image_right[y][0];
    right[1] = image_right[y][x];
    right[2] = (x<m_w-1) ? image_right[y][x+1] : image_right[y][m_w-1];

    return birchfield_absolute_difference_rgb_per_pixel(left[0], left[1], left[2], right[0], right[1], right[2]);
}

short qx_stereo_corr_vol_cpu::birchfield_absolute_difference_rgb_per_pixel(const unsigned char *image10, const unsigned char *image11, const unsigned char *image12,
        const unsigned char *image20, const unsigned char *image21, const unsigned char *image22)
{
    float rgb[3]={0.299f,0.587f,0.114f};
    short cost[5];
    /* The weighted sum of absolute differences of the center pixels image11 and image21 */
    cost[0]=short(abs(image11[0]-image21[0])*rgb[0]+abs(image11[1]-image21[1])*rgb[1]+abs(image11[2]-image21[2])*rgb[2]+0.5);

    /* Weighted sum of absolute differences of the center pixel in the first image and the average of the left and center pixel in the second image */
    cost[1]=short(abs(image11[0]-0.5f*(image21[0]+image20[0]))*rgb[0]+abs(image11[1]-0.5f*(image21[1]+image20[1]))*rgb[1]
                  +abs(image11[2]-0.5f*(image21[2]+image20[2]))*rgb[2]+0.5);

    /* Weighted sum of absolute differences of the center pixel in the first image and the average of the right and center pixel in the second image */
    cost[2]=short(abs(image11[0]-0.5f*(image21[0]+image22[0]))*rgb[0]+abs(image11[1]-0.5f*(image21[1]+image22[1]))*rgb[1]+
                  abs(image11[2]-0.5f*(image21[2]+image22[2]))*rgb[2]+0.5);

    /* Weighted sum of absolute differences of the average of the left and center pixel in the first image and the center pixel in the second image */
    cost[3]=short(abs(image21[0]-0.5f*(image11[0]+image10[0]))*rgb[0]+abs(image21[1]-0.5f*(image11[1]+image10[1]))*rgb[1]+
                  abs(image21[2]-0.5f*(image11[2]+image10[2]))*rgb[2]+0.5);
    
    /* Weighted sum of absolute differences of the average of the right and center pixel in the first image and the center pixel in the second image */
    cost[4]=short(abs(image21[0]-0.5f*(image11[0]+image12[0]))*rgb[0]+abs(image21[1]-0.5f*(image11[1]+image12[1]))*rgb[1]+
                  abs(image21[2]-0.5f*(image11[2]+image12[2]))*rgb[2]+0.5);

    return(min(
            min(
                    min(cost[0],cost[1]),
                    min(cost[2],cost[3])),
            cost[4]));
}

void qx_stereo_corr_vol_cpu::corr_vol_reverse(corr_vol_t corr_vol_right, const_corr_vol_t corr_vol) const
{
    int y,x,d;
    for (y=0;y<m_h;y++)
    {
        for (x=0;x<m_w-m_nr_planes;x++)
            for (d=0;d<m_nr_planes;d++)
                corr_vol_right[y][x][d]=corr_vol[y][x+d][d];
        for (x=m_w-m_nr_planes;x<m_w;x++)
            //for(x=m_w-m_nr_planes;x<int(m_w-m_nr_planes*0.1);x++)
        {
            for (d=0;d<m_nr_planes;d++)
            {
                if ((x+d)<m_w)
                {
                    corr_vol_right[y][x][d]=corr_vol[y][x+d][d];
                }
                else
                {
                    corr_vol_right[y][x][d]=corr_vol_right[y][x][d-1];
                }
            }
        }
    }
    return;
}
