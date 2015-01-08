#ifndef CPUPREPROCESSOR_HPP
#define CPUPREPROCESSOR_HPP


#include "AbstractPreprocessor.hpp"

#include "ReverseMapper.hpp"

#include <Eigen/Core>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia
{

    typedef Eigen::Matrix3f HomographyMatrix;

    using boost::shared_ptr;
    using boost::scoped_ptr;

/**
 * Image preprocessing class:
 *  - Unbayering
 *  - Undistortion
 *  - Rectification
 *  - Smoothing
 *  - Corner detection
 *
     * Based on code from Andreas Ess
 **/
class CpuPreprocessor : public AbstractPreprocessor
{

public:

    static boost::program_options::options_description get_args_options();

    CpuPreprocessor(const dimensions_t &dimensions,
                    const StereoCameraCalibration &stereo_calibration,
                    const boost::program_options::variables_map &options);

    ~CpuPreprocessor();



    /// @param camera_index indicates which camera is being given as input. The undistortion and rectification steps are camera dependent.
    void run(const input_image_view_t& input, const int camera_index,
             const output_image_view_t &output);

    const point2<float> run(const point2<int> &point, const int camera_index) const;

    const HomographyMatrix& get_right_rectification_homography() const { return right_rectification_homography; }
    const HomographyMatrix& get_left_rectification_homography() const { return left_rectification_homography; }


    /// @returns the stereo calibration corresponding to the post-processed images
    const StereoCameraCalibration& get_post_processing_calibration() const;

protected:

    StereoCameraCalibration post_processing_stereo_calibration;

    void set_post_processing_stereo_calibration();


    bool should_compute_residual, should_remove_specular_reflection;

    /// temporary image buffer
    AbstractVideoInput::input_image_t t_img;

    void compute_rectification_homographies(const dimensions_t &dimensions, const StereoCameraCalibration &stereo_calibration);
    void compute_rectification(const input_image_view_t &src, const int camera_index, const output_image_view_t &dst);
    void compute_rectification(const input_image_view_t &src, const HomographyMatrix &H, const output_image_view_t &dst);

    const point2<float> compute_rectification(const point2<float> &point, const int camera_index) const;
    const point2<float> compute_rectification(const point2<float> &point, const HomographyMatrix &H) const;

    ///R everse mappers are used for undistortion and rectification
    shared_ptr<const CameraCalibration> left_camera_calibration_p;
    shared_ptr<const CameraCalibration> right_camera_calibration_p;

    scoped_ptr<ReverseMapper> left_reverse_mapper_p;
    scoped_ptr<ReverseMapper> right_reverse_mapper_p;

    void compute_warping(const input_image_view_t &src, const int camera_index, const output_image_view_t &dst);

public:
    const point2<float> compute_warping(const point2<int> &point, const int camera_index) const;

protected:

    /**
    Smooth given image with a Gaussian

     @param im image to smooth
     @param sigma sigma of Gaussian
     @param fsize size of filter mask
    */
    void compute_smoothing(const input_image_view_t &src, const output_image_view_t &dst);

public:
    // FIXME is this really a good idea ?
    void compute_residual(const input_image_view_t &src, const output_image_view_t &dst);

protected:
    void compute_specular_removal(const input_image_view_t &src, const output_image_view_t &dst);

    /// Rectification Homography transform for cameras
    HomographyMatrix left_rectification_homography, right_rectification_homography;
    HomographyMatrix left_rectification_inverse_homography, right_rectification_inverse_homography;

    /// set while caling compute_rectification_homographies
    InternalCalibrationMatrix new_left_internal_calibration, new_right_internal_calibration;


};

} // end of namespace doppia

#endif // CPUPREPROCESSOR_HPP
