#ifndef DRAW_MATRIX_HPP
#define DRAW_MATRIX_HPP

#include <Eigen/Core>
#include <boost/gil/typedefs.hpp>

namespace doppia {

void draw_matrix(const Eigen::MatrixXf &matrix, boost::gil::rgb8_view_t &view);

} // end of namespace doppia

#endif // DRAW_MATRIX_HPP
