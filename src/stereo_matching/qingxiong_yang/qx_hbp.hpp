/********************************************************************************************************
\Author:	Qingxiong Yang (http://vision.ai.uiuc.edu/~qyang6/)
\Function:	Hierarchical BP on CPU given the correlation volume and the color image.
\reference: Pedro Felzenszwalb and Daniel Huttenlocher, Efficient Belief Propagation for Early Vision,
			IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2004.
*********************************************************************************************************/
#ifndef QX_BP_HIERARCHICAL_H
#define QX_BP_HIERARCHICAL_H

#define QX_DEF_BP_NR_SCALES							5

#define QX_DEF_BP_NR_ITER0							5
#define QX_DEF_BP_NR_ITER1							5
#define QX_DEF_BP_NR_ITER2							5
#define QX_DEF_BP_NR_ITER3							5
#define QX_DEF_BP_NR_ITER4							5
#define QX_DEF_BP_NR_ITER5							5

#define QX_DEF_BP_COST_MAX_DATA_TERM				15		/*Truncation of data cost*/
#define QX_DEF_BP_COST_DISCONTINUITY_SINGLE_JUMP	10		/*discontinuity cost*/
#define QX_DEF_BP_MAX_NR_JUMP						2		/*Truncation of discontinuity cost*/


class qx_hbp
{
public:
	struct options 
    {
        int nr_planes;
        short cost_discontinuity_single_jump;
        short cost_max_data_term;
        int max_nr_jump;
        int nr_scales;
        options(int nr_planes,
                short cost_discontinuity_single_jump,
                short cost_max_data_term,
                int max_nr_jump,
                int nr_scales)
            : nr_planes(nr_planes),
              cost_discontinuity_single_jump(cost_discontinuity_single_jump),
              cost_max_data_term(cost_max_data_term),
              max_nr_jump(max_nr_jump),
              nr_scales(nr_scales)
        {}
    };

    static options get_default_options()
    {
        return options(Default_Nr_Planes,
            Default_Cost_Discontinuity_Single_Jump,
            Default_Cost_Max_Data_Term,
            Default_Max_Nr_Jump,
            Default_Nr_Scales);
    }

    enum { 
        /* Default number of iterations if iterations array is not passed */
        Default_Nr_Iter0 = 5,
        Default_Nr_Iter1 = 5,
        Default_Nr_Iter2 = 5,
        Default_Nr_Iter3 = 5,
        Default_Nr_Iter4 = 5,
        Default_Nr_Iter5 = 5, // and all higher levels

        /* Default options for get_default_options */
        Default_Nr_Scales = 5,                        // Number of scales
        Default_Nr_Planes = 64,                       // Number of disparities to initially consider for the coarsest scale
        Default_Cost_Max_Data_Term = 15,              // Truncation of the data cost
        Default_Cost_Discontinuity_Single_Jump = 10,  // Cost per unit difference between neighboring disparities
        Default_Max_Nr_Jump = 2                       // Truncate the difference between neighboring disparities
    };
   
    qx_hbp(int h,int w,const options& opts=get_default_options(), int *iterations=NULL);
    
    ~qx_hbp();

    int disparity(qx_array_2d<short>::type disparity, qx_array_3d<short>::const_type corr_vol);

private:
    qx_hbp(const qx_hbp&);
    qx_hbp& operator= (const qx_hbp&);

    void clean();
    enum Neighbor { NbNorth=0, NbWest=1, NbSouth=2, NbEast=3 };
    enum { NrNeighbors = 4 };

    void compute_data_cost_pyramid(qx_array_3d<short>::const_type corr_vol);
    void init_message(int scales_idx);

    void compute_message(int scale);
    void compute_message_per_pixel(qx_array_4d<short>::type message, qx_array_3d<short>::const_type cost, int y, int x, int h, int w);
    void compute_message_per_pixel_per_neighbor(qx_array_1d<short>::type m_comp_func_sub, short minimum);
    int compute_disparity(qx_array_2d<short>::type disparity,int scale=0);

    int	m_h;
    int m_w;
    int m_nr_plane;
    int m_nr_scale;
    qx_array_1d<int> m_iteration;
    short m_cost_discontinuity_single_jump;
    short m_cost_max_discontinuity;
    short m_cost_max_data_term;

    qx_array_1d< qx_array_3d<short> > m_data_cost_pyramid;
    
    qx_array_1d<short> m_temp_f;
    qx_array_1d<short> m_temp_s;
    qx_array_1d<short> m_temp_s2;
    qx_array_1d<short> m_zero_messages;

    qx_array_1d< qx_array_4d<short> > m_message_pyramid;
    qx_array_1d<int> m_h_pyramid;
    qx_array_1d<int> m_w_pyramid;
    qx_array_1d< qx_array_2d<short> > m_disparity_map_pyramid;
};
#endif
