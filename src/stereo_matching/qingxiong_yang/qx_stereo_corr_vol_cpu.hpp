/***********************************************************************************
\Author:
\Function:	Correlation volume calculated with difference approaches (CPU version).
************************************************************************************/
#ifndef QX_STEREO_CORR_VOL_H
#define QX_STEREO_CORR_VOL_H
/*Correlation Volume (Birchfield Dissimilarity)*/
#define QX_DEF_STEREO_CORR_VOL_METHOD					0
#define QX_DEF_STEREO_CORR_VOL_METHOD_SMOOTH_BIRCHFIELD			0
#define QX_DEF_STEREO_CORR_VOL_METHOD_ABSOLUTE_DIFFERENCE		1

#include "qx_basic.hpp"

class qx_stereo_corr_vol_cpu
{
public:
    enum pixel_matching_type { PixelMatchingBirchfieldTomasi, PixelMatchingZNCC, PixelMatchingSAD };

	typedef qx_array_3d<short>::type corr_vol_t;
	typedef qx_array_3d<short>::const_type const_corr_vol_t;
	typedef qx_array_3d<unsigned char>::type image_t;
	typedef qx_array_3d<unsigned char>::const_type const_image_t;

    qx_stereo_corr_vol_cpu(const int h, const int w, const int nr_planes, pixel_matching_type method=PixelMatchingBirchfieldTomasi);
    ~qx_stereo_corr_vol_cpu();

    /* Cost volume computation */
    void corr_vol(
			corr_vol_t corr_vol,
			const_image_t image_left,
			const_image_t image_right,
			const bool is_left_reference=true)
		const;

    void corr_vol_reverse(
			corr_vol_t corr_vol_right,
			const_corr_vol_t corr_vol)
		const;

    void corr_vol_zncc(
			corr_vol_t corr_vol,
			const_image_t image_left,
			const_image_t image_right,
			int radius)
		const;
    
    /* Aggregation */
    void aggregate_smooth(
			corr_vol_t corr_vol,
			int radius) 
		const;

    void aggregate_color_weighted(
			corr_vol_t corr_vol_out,
			const_corr_vol_t corr_vol_in,
			const_image_t image_left,
			const_image_t image_right,
			int radius)
		const;

private:
    int	m_h,m_w,m_nr_planes,m_method;

    void absolute_difference_rgb(
			corr_vol_t corr_vol,
			const_image_t image_left,
			const_image_t image_right)
		const;

    void birchfield_absolute_difference_rgb(
			corr_vol_t corr_vol,
			const_image_t image_left,
			const_image_t image_right)
		const;
			
    void birchfield_absolute_difference_rgb_right_reference(
			corr_vol_t corr_vol,
			const_image_t image_left,
			const_image_t image_right)
		const;
    
	short birchfield_absolute_difference_rgb_per_pixel(
			const_image_t image_left,
			const_image_t image_right,
			int y,
			int x,
			int d)
		const;

    short birchfield_absolute_difference_rgb_per_pixel_right_reference(
			const_image_t image_left,
			const_image_t image_right,
			int y,
			int x,
			int d)
		const;

    static short birchfield_absolute_difference_rgb_per_pixel(const unsigned char *image10, const unsigned char *image11, const unsigned char *image12,
                                                       const unsigned char *image20, const unsigned char *image21, const unsigned char *image22);

};
#endif
