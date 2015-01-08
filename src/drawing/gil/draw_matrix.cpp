#include "draw_matrix.hpp"

#include <boost/gil/image_view.hpp>

#include <stdexcept>

namespace doppia {


void draw_matrix(const Eigen::MatrixXf &matrix, boost::gil::rgb8_view_t &view)
{

    if(matrix.cols() != view.width() or
            matrix.rows() != view.height() )
    {
        throw std::invalid_argument("draw_matrix expects to receive a view and a matrix of identical dimensions");
    }

    if(matrix.size() == 0)
    {
        // empty matrix (non initialized) matrix
        // nothing to be done
        return;
    }

    Eigen::MatrixXf matrix_copy = matrix;

    const float max_value = matrix_copy.maxCoeff();
    const float min_value = matrix_copy.minCoeff();

    if(min_value != 0)
    {
        matrix_copy -= Eigen::MatrixXf::Constant(matrix.rows(), matrix.cols(), min_value);
    }

    if( (max_value - min_value) != 255.0f)
    {
        matrix_copy *= 255.0f / (max_value - min_value);
    }

    for(int y=0; y < view.height(); y+=1)
    {
        for(int x=0; x < view.width(); x+=1)
        {
            const boost::gil::bits8 t_value = matrix_copy(y,x);
            boost::gil::rgb8_pixel_t t_pixel(t_value, t_value, t_value);
            *(view.at(x,y)) = t_pixel;

        } // end of "for each x"
    } // end of "for each y"

    return;
}


} // end of namespace doppia
