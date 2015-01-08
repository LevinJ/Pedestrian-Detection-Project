#ifndef REVERSEMAPPER_HPP
#define REVERSEMAPPER_HPP

#include <Eigen/Core>

#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/gil/utilities.hpp>

#include "video_input/calibration/CameraCalibration.hpp"
#include "video_input/AbstractVideoInput.hpp"
#include "video_input/preprocessing/AbstractPreprocessor.hpp"

namespace doppia {

using boost::shared_ptr;
using boost::scoped_ptr;
using boost::gil::point2;

typedef Eigen::Matrix3f HomographyMatrix;

/// Utility class used to warp images
/// @see http://en.wikipedia.org/wiki/Image_warping
/// @note Should always use FastReverseMapper instead of this base implementation
class ReverseMapper
{
public:

    typedef AbstractPreprocessor::input_image_view_t input_image_view_t;
    typedef AbstractPreprocessor::output_image_view_t output_image_view_t;
    typedef AbstractVideoInput::input_image_view_t::point_t dimensions_t;

    ReverseMapper(const dimensions_t &dimensions, const shared_ptr<const CameraCalibration> &calibration_p);

    /// Default destructor
    virtual ~ReverseMapper();

    virtual void add_undistortion();

    /// if H_is_centered is true H expects the center pixel of the image to be (0,0),
    /// if H_is_centered is false H expects the center pixel of the image to be (center_x, center_y)
    virtual void add_homography_projection(const HomographyMatrix &H, const bool H_is_centered);

    virtual void warp(const input_image_view_t &input, const output_image_view_t &output) const;

    virtual const point2<float> &warp(const point2<int> &point) const;

protected:

    bool added_undistortion, added_homography;

    const dimensions_t input_dimensions;

    /// temporary image buffer
    AbstractVideoInput::input_image_t t_img;
    typedef AbstractVideoInput::input_image_t::view_t t_img_view_t;
    t_img_view_t t_img_view;

    shared_ptr<const CameraCalibration> calibration_p;

    /// look-up table that keeps an (x,y) -> (x,y) mapping
    typedef boost::multi_array<point2<float>, 2> lookup_table_t;
    scoped_ptr<lookup_table_t> lookup_table_p;

    const point2<float> from_undistorted_to_source(const point2<float> &destination) const;

};

} // end of namespace doppia

#endif // REVERSEMAPPER_HPP
