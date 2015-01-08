/********************************************************************************************************
\Author:	Qingxiong Yang (http://vision.ai.uiuc.edu/~qyang6/)
\Function:	Disparity map refinement using joint bilateral filtering (CPU) given a single color image.
\reference: Qingxiong Yang, Liang Wang and Narendra Ahuja, A Constant-Space Belief Propagation Algorithm
			for Stereo Matching, IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2010.
*********************************************************************************************************/
#ifndef QX_DISPARITY_MAP_BF_REFINEMENT_H
#define QX_DISPARITY_MAP_BF_REFINEMENT_H
#define QX_DEF_BF_FILTER_RADIUS						3		/*filter radius*/
#define QX_DEF_BF_FILTER_SIGMA_RANGE				10		/*filter range sigma*/
#define QX_DEF_MAX_DISC_THRESHOLD					0.2f	/*Truncation of disparity continuity*/
#define QX_DEF_EDGE_THRESHOLD						0.1f	/*Truncation of disparity continuity*/

#include "qx_basic.hpp"

class qx_disparity_map_bf_refinement
{
public:
    qx_disparity_map_bf_refinement(int h,int w,int nr_plane,int radius=QX_DEF_BF_FILTER_RADIUS);
    ~qx_disparity_map_bf_refinement() {}

    int disparity_refinement(short * const *disparity, const unsigned char * const * const *image, int nr_iter=1,
                             float edge_threshold=QX_DEF_EDGE_THRESHOLD,
                             float max_disc_threshold=QX_DEF_MAX_DISC_THRESHOLD);

private:
    void disparity_refinement(short * const *disparity, const unsigned char * const * const * image, const short *dp,int nr_dp, short max_disc, int y, int x, int radius, int h, int w);

    int m_h,m_w,m_nr_plane,m_radius; /*Post processing: joint bilateral filtering*/
    qx_array_1d<double> m_table_color;
    qx_array_2d<double> m_table_space;
};
#endif
