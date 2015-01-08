
// Python interface to opencv gpu resize

#include <cstdio>
#include <stdexcept>

#include <boost/numpy.hpp>
#include <boost/cstdint.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace opencv_cpp
{

namespace bp = boost::python;
namespace bn = boost::numpy;

typedef float float32_t;
typedef double float64_t;
typedef boost::uint8_t uint8_t;

/// Based on http://wiki.python.org/moin/boost.python/HowTo#Multithreading_Support_for_my_function
class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
        m_thread_state = PyEval_SaveThread();
        return;
    }

    inline ~ScopedGILRelease()
    {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
        return;
    }

private:
    PyThreadState * m_thread_state;
};


cv::Mat rgb8_ndarray_to_mat(const bn::ndarray &a)
{
    if(a.get_nd() != 3)
    {
        printf("ERROR: Input ndarray has dimension %i != 3\n", a.get_nd());
        throw std::invalid_argument("rgb8_ndarray_to_mat expects an ndarray of dimension 3");
    }

    if(a.get_dtype() != bn::dtype::get_builtin<uint8_t>())
    {
        throw std::invalid_argument("rgb8_ndarray_to_mat expects an ndarray with dtype == uint8");
    }

    return cv::Mat(a.get_shape()[0], a.get_shape()[1], CV_8UC3, a.get_data());

} // end of rgb8_ndarray_to_mat


/// Will upload the image to the gpu, resize it, and call the result back
void gpu_resize(const bn::ndarray &input_ndarray,
                bn::ndarray &output_ndarray)
{
    cv::Mat output_ndarray_as_mat = rgb8_ndarray_to_mat(output_ndarray);

    cv::Mat cpu_input, cpu_output;
    cv::gpu::GpuMat gpu_input, gpu_output;

    cpu_input = rgb8_ndarray_to_mat(input_ndarray);
    gpu_input.upload(cpu_input);

    cv::gpu::resize(gpu_input, gpu_output, output_ndarray_as_mat.size());

    gpu_output.download(cpu_output);

    cpu_output.copyTo(output_ndarray_as_mat);

    return;
} // end of matrix_convolve_2d_fast_v1


} // end of opencv_cpp namespace


BOOST_PYTHON_MODULE(opencv_gpu_resize) {

    boost::numpy::initialize();
    boost::python::def("gpu_resize",
                       opencv_cpp::gpu_resize,
                       "Expects to receive two numpy arrays of 3 dimensions and dtype uint8\n"
                       "The input array defines the image content,\n"
                       "the output array defines the output size (should be pre-allocated)");

} // end of python module convolve_cpp


// end of file
