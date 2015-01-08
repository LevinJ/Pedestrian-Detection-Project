/********************************************************************************************************
\Author:	Qingxiong Yang (http://vision.ai.uiuc.edu/~qyang6/)
\Function:	Constant space BP (CPU) given two color images.
\reference: Qingxiong Yang, Liang Wang and Narendra Ahuja, A Constant-Space Belief Propagation Algorithm
			for Stereo Matching, IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2010.
*********************************************************************************************************/
#ifndef qx_csbp_H
#define qx_csbp_H

#include <vector>

class qx_csbp_base
{

public:
    /*
     * These are the options that have reasonable default values
     * (unlike width and height)
     */
    struct options 
    {
        int nr_planes;
        int nr_planes_base_level;
        short cost_discontinuity_single_jump;
        short cost_max_data_term;
        int max_nr_jump;
        int nr_scales;
        bool use_local_minima;
        options(int nr_planes,
                int nr_planes_base_level,
                short cost_discontinuity_single_jump,
                short cost_max_data_term,
                int max_nr_jump,
                int nr_scales,
                bool use_local_minima)
            : nr_planes(nr_planes),
              nr_planes_base_level(nr_planes_base_level),
              cost_discontinuity_single_jump(cost_discontinuity_single_jump),
              cost_max_data_term(cost_max_data_term),
              max_nr_jump(max_nr_jump),
              nr_scales(nr_scales),
              use_local_minima(use_local_minima)
        {}
    };

    static options get_default_options()
    {
        return options(Default_Nr_Planes,
            Default_Nr_Planes_Base_Level,
            Default_Cost_Discontinuity_Single_Jump,
            Default_Cost_Max_Data_Term,
            Default_Max_Nr_Jump,
            Default_Nr_Scales,
            Default_Use_Local_Minima);
    }

    qx_csbp_base(int h, int w, const options& opts=get_default_options(), int *iterations=NULL);
    virtual ~qx_csbp_base() {}

    enum { 
        /* Default number of iterations if iterations array is not passed */
        Default_Nr_Iter0 = 5,
        Default_Nr_Iter1 = 5,
        Default_Nr_Iter2 = 5,
        Default_Nr_Iter3 = 5,
        Default_Nr_Iter4 = 5,
        Default_Nr_Iter5 = 5, // and all higher levels

        /* Default options for get_default_options */
        Default_Nr_Planes_Base_Level = 2,             // Number of disparities to keep at base scale
        Default_Nr_Scales = 5,                        // Number of scales
        Default_Nr_Planes = 64,                       // Number of disparities to initially consider for the coarsest scale
        Default_Cost_Max_Data_Term = 30,              // Truncation of the data cost
        Default_Cost_Discontinuity_Single_Jump = 10,  // Cost per unit difference between neighboring disparities
        Default_Max_Nr_Jump = 2,                      // Truncate the difference between neighboring disparities
        Default_Use_Local_Minima = true,              // Keep all local minima when reducing disparities (1 or 0)
    };

    /* 
     * The "main" function.
     * Computes the optimal disparity image.
     * Disparity for pixel (x,y) will be at index y*w + x.
     * This relies on subclasses to implement build_integral_data_cost_volume(),
     * which will fill in the data dependent part.
     *
     * The return value points into memory managed by qx_csbp_base.
     * The underlying array will be overwritten on the next call to compute_disparity.
     */
    short *compute_disparity();

private:
    /* No copies allowed */
    qx_csbp_base(const qx_csbp_base&);
    qx_csbp_base& operator=(const qx_csbp_base&);

protected:
    /* 
     * Integral image cost volume.
     * The integral image is organized in blocks of 
     * m_nr_planes entries for each pixel (x,y), where 
     * the pixels are traversed in a row-major fashion.
     * The volume has the size (h+1)x(w+1)x(m_nr_planes)
     * The entry (x+1,y+1,d) describes the integral of
     * the data cost of all pixels in the range from (0,0)
     * through (x,y) for the given disparity d.
     * 
     * The type of the integral image is integral_image_element_type.
     *
     */
    typedef int64_t integral_image_element_type;
    std::vector<integral_image_element_type> m_integral_data_cost_volume;
    std::vector<integral_image_element_type*> m_integral_data_cost_volume_row_ptr;

    /* The computation of the integral image is left to subclasses */
    virtual void build_integral_data_cost_volume() = 0;
    integral_image_element_type get_integral_data_cost(int x, int y, int disp) const { return get_integral_data_costs(x,y)[disp]; }
    integral_image_element_type * get_integral_data_costs(int x, int y);
    const integral_image_element_type * get_integral_data_costs(int x, int y) const;
    short get_data_cost_for_pixel(int scale, int x, int y, int disp) const;

    int get_base_width() const { return m_w; }
    int get_base_height() const { return m_h; }
    int get_nr_planes() const { return m_nr_planes; }

    short get_cost_max_data_term() const { return m_cost_max_data_term; }

private:
    /*
     * Arrays to keep the data costs at a given iteration/pyramid level
     */
    int m_nr_scale;
    qx_array_1d<int> m_iteration;   // number of iterations to run message passing on each scale
    std::vector<short> m_data_cost;          // data costs inferred from previous scale (?)
    std::vector<short> m_data_cost_selected; // data costs after reduction of labels (? or is it the other way around...)
    std::vector<short> m_selected_disparity;
    std::vector<int> m_max_nr_plane_pyramid; // maximum number of disparities considered on each scale
    std::vector<int> m_w_pyramid;            // width of each pyramid level
    std::vector<int> m_h_pyramid;            // height of each pyramid level

    int get_height_at_scale(int scale) const { return m_h_pyramid[scale]; }
    int get_width_at_scale(int scale) const { return m_w_pyramid[scale]; }
    int get_nr_plane_at_scale(int scale) const { return m_max_nr_plane_pyramid[scale]; }
    short * get_selected_disparities(int scale, int x, int y);
    short * get_selected_data_cost(int scale, int x, int y);
    short * get_data_cost(int scale, int x, int y);

    /*
     * Keep the base level dimensions easily accessible
     * Also, the total number of planes considered (i.e. the planes in the cost volume)
     * usually differs from the number of planes on the coarsest level 
     * (which is 2^i * nr_planes_at_base_level)
     */
    int m_w;
    int m_h; 
    int m_nr_planes;

    /* 
     * Additional options
     */
    short m_cost_discontinuity_single_jump;
    short m_cost_max_discontinuity;
    short m_cost_max_data_term;

    /* 
     * Some temporary storage
     */
    std::vector<short> m_temp;
    std::vector<short> m_temp_2;
    std::vector<short> m_temp_3;

    /*
     * Message computation
     */
    std::vector<short> m_message;

    enum Neighbor { NbNorth=0, NbWest=1, NbSouth=2, NbEast=3 };
    enum { NrNeighbors = 4 };
    short * get_messages(int scale, int x, int y, Neighbor nb);

    void init_message(int scale);
    void compute_message(int scale);
    void compute_message_per_pixel(short*c0,short *p0,short *p1,short *p2,short *p3,short *p4,
                                   short*d0,short*d1,short*d2,short*d3,short*d4,
                                   int y,int x,int nr_planes,int scale,int &count);
    void compute_message_per_pixel_per_neighbor(short *m_comp_func_sub,short minimum,
            short *disp_left,short *disp_right,int nr_planes,int scale);
    
    /*
     * Compute the disparity after the last round of message passing.
     * This is simply the minimum cost disparity for every node.
     */
    int compute_disparity(short*disparity,int scale);

    void compute_data_cost_init(int scale);
    void compute_data_cost(int scale);
};

/* 
 * Subclass that builds the integral image and only
 * needs to obtain pixel-wise costs from 
 *
 * TODO naming is horrible.
 * This suffers from the problems you get
 * when subclassing defines functionality.
 * Mix-ins might be a better solution here, but they'd be
 * ugly too (virtual inheritance...)
 *
 * One could cleanly and efficiently provide the necessary
 * information to qx_csbp_base through templates, but this
 * would make the whole implementation visible and would 
 * suffer from the usual compile-time penalty.
 *
 * Choose your poison, really.
 */
class qx_csbp_pixelwise : public qx_csbp_base
{
    public:
        qx_csbp_pixelwise(int h, int w, const options& opts = get_default_options(), int *iterations=NULL)
            : qx_csbp_base(h, w, opts, iterations)
        {}
    
    protected:
        void build_integral_data_cost_volume();
        virtual void set_left_pixel(int x, int y) = 0;
        virtual void set_disparity(int d) = 0;
        virtual short cost_for_pixel_pair() = 0;
};

/*
 * Implementation for RGB8 images using absolute differences (AD) of pixels (with RGB weighting)
 *
 * Note: it is now trivial enough to add your own sub-class for other pixel/image types, as well as
 *       for cost volume preprocessing.
 */
template<typename PxT, unsigned NCh>
class qx_csbp : public qx_csbp_pixelwise
{
    public:
        qx_csbp(int h, int w, const options& opts = get_default_options(), int *iterations=NULL)
            : qx_csbp_pixelwise(h,w,opts,iterations),
              m_left(NULL),
              m_right(NULL),
              m_px_left(NULL),
              m_px_right_base(NULL)
        {}

        void set_images(const PxT *left, const PxT *right)
        {
            m_px_left = m_left = left;
            m_px_right_base = m_right = right;
        }

        short *disparity(const PxT *left, const PxT *right)
        {
            set_images(left, right);
            return compute_disparity();
        }

    protected:
        void set_left_pixel(int x, int y)
        {
            const int idx = (y*get_base_width()+x)*NCh;
            m_px_left = m_left + idx;
            m_px_right_base = m_right + idx;
        }
        void set_disparity(int disparity)
        {
            m_disparity = disparity;
        }

        short cost_for_pixel_pair()
        {
            const PxT *px_right = m_px_right_base - m_disparity * NCh;
            int sum = NCh >> 1;
            for(unsigned i=0; i!=NCh; ++i)
                sum += std::abs(m_px_left[i] - px_right[i]);
            return sum / NCh;
        }
   
    private:
        const PxT *m_left;
        const PxT *m_right;
        const PxT *m_px_left;
        const PxT *m_px_right_base;
        int m_disparity;
};

template<typename PxT>
class qx_csbp_rgb : public qx_csbp_pixelwise
{
    public:
        qx_csbp_rgb(int h, int w, const options& opts = get_default_options(), int *iterations=NULL)
            : qx_csbp_pixelwise(h,w,opts,iterations),
              m_left(NULL),
              m_right(NULL),
              m_px_left(NULL),
              m_px_right_base(NULL)
        {}

        void set_images(const PxT *left, const PxT *right)
        {
            m_px_left = m_left = left;
            m_px_right_base = m_right = right;
        }

        short *disparity(const PxT *left, const PxT *right)
        {
            set_images(left, right);
            return compute_disparity();
        }

    protected:
        void set_left_pixel(int x, int y)
        {
            const int idx = (y*get_base_width()+x)*3;
            m_px_left = m_left + idx;
            m_px_right_base = m_right + idx;
        }
        void set_disparity(int disparity)
        {
            m_disparity = disparity;
        }

        short cost_for_pixel_pair()
        {
            const PxT *px_right = m_px_right_base - m_disparity * 3;
            return (299 * abs(m_px_left[0]-px_right[0])
                 + 587 * abs(m_px_left[1]-px_right[1])
                 + 114 * abs(m_px_left[2]-px_right[2])
                 + 500) / 1000; // 500 is for rounding
        }
   
    private:
        const PxT *m_left;
        const PxT *m_right;
        const PxT *m_px_left;
        const PxT *m_px_right_base;
        int m_disparity;
};

class qx_csbp_from_costvolume : public qx_csbp_pixelwise
{
    public:
        qx_csbp_from_costvolume(int h, int w, const options& opts = get_default_options(), int *iterations=NULL)
            : qx_csbp_pixelwise(h,w,opts,iterations),
	      m_cost_volume(NULL),
	      m_x(0),
	      m_y(0),
	      m_disparity(0)
        {}

	void set_cost_volume(short ***costvol)
	{
		m_cost_volume = costvol;
        }

        short *disparity(short ***costvol)
        {
            set_cost_volume(costvol);
            return compute_disparity();
        }

    protected:
        void set_left_pixel(int x, int y)
        {
		m_x = x;
		m_y = y;
        }

        void set_disparity(int disparity)
        {
		m_disparity = disparity;
        }

        short cost_for_pixel_pair()
        {
		return m_cost_volume[m_y][m_x][m_disparity];
        }
   
    private:
	short ***m_cost_volume;
	int m_x;
	int m_y;
	int m_disparity;
};


#endif
