#ifndef FASTREVERSEMAPPER_HPP
#define FASTREVERSEMAPPER_HPP

#include "ReverseMapper.hpp"

#include <opencv2/core/core.hpp>

namespace doppia {

class FastReverseMapper: public ReverseMapper
{
public:
    FastReverseMapper(const dimensions_t &dimensions, const shared_ptr<const CameraCalibration> &calibration_p);
    ~FastReverseMapper();

    virtual void add_undistortion();

    /// if H_is_centered is true H expects the center pixel of the image to be (0,0),
    /// if H_is_centered is false H expects the center pixel of the image to be (center_x, center_y)
    virtual void add_homography_projection(const HomographyMatrix &H, const bool H_is_centered);

    virtual void warp(const input_image_view_t &input, const output_image_view_t &output) const;

    virtual const point2<float> &warp(const point2<int> &point) const;

protected:

    cv::Mat slow_lookup_table_x, slow_lookup_table_y;
    cv::Mat fast_lookup_table_map1, fast_lookup_table_map2;

    void update_fast_lookup_table();

};

} // end of namespace doppia

#endif // FASTREVERSEMAPPER_HPP
