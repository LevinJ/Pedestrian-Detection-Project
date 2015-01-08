#ifndef OPENCVSTEREO_HPP
#define OPENCVSTEREO_HPP

#include "AbstractStereoBlockMatcher.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/legacy/legacy.hpp> // opencv 2.4, for CvStereoGCState

#include <boost/scoped_ptr.hpp>

namespace cv
{

template<> inline void Ptr<CvStereoGCState>::delete_obj()
{
    cvReleaseStereoGCState(&obj);
}


template<> inline void Ptr<CvStereoBMState>::delete_obj()
{
    cvReleaseStereoBMState(&obj);
}


/// Default value for maxIters is defined from the cvCreateStereoGCState documentation
class  StereoGC
{
public:

    StereoGC();
    StereoGC(int numberOfDisparities, int maxIters=2);
    void init(int numberOfDisparities, int maxIters=2);
    void operator()( const Mat& left, const Mat& right, Mat& disparity );

    Ptr<CvStereoGCState> state;
};

namespace gpu {
 class StereoConstantSpaceBP; // forward declaration
 class StereoBeliefPropagation; // forward declaration
}

} // end of namespace cv

namespace doppia
{

using namespace boost;
using namespace std;

class OpenCvStereo: public AbstractStereoBlockMatcher
{

public:

    static boost::program_options::options_description get_args_options();

    OpenCvStereo(const boost::program_options::variables_map &options);
    ~OpenCvStereo();

    void set_rectified_images_pair( gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right);

    void compute_disparity_map(gil::gray8c_view_t &left, gil::gray8c_view_t &right, bool left_right_are_inverted);

    void compute_disparity_map(gil::rgb8c_view_t  &left, gil::rgb8c_view_t &right, bool left_right_are_inverted);


protected:

    cv::StereoBM stereo_bm;
    cv::StereoGC stereo_gc;
#if defined(USE_GPU)
    boost::scoped_ptr<cv::gpu::StereoConstantSpaceBP> stereo_gpu_csbp_p;
    boost::scoped_ptr<cv::gpu::StereoBeliefPropagation> stereo_gpu_bp_p;
#endif

    enum StereoAlgorithmType { BlockMatchingAlgorithm, GraphCutAlgorithm, CsbpAlgorithm, BeliefPropagationAlgorithm };
    StereoAlgorithmType stereo_algorithm;
    int gc_max_iterations;

    template<typename ImgView> void compute_disparity_map_impl(ImgView &left, ImgView &right, bool left_right_are_inverted);

};

} // end of namespace doppia

#endif // OPENCVSTEREO_HPP
