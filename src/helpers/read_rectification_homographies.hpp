#ifndef READ_RECTIFICATION_HOMOGRAPHIES_HPP
#define READ_RECTIFICATION_HOMOGRAPHIES_HPP

#include <Eigen/Core>
#include <iosfwd>
#include <cstdio>

namespace doppia
{
	void read_rectification_homographies(std::istream& is, Eigen::Matrix3f& left, Eigen::Matrix3f& right);
	void read_rectification_homographies(FILE *in, Eigen::Matrix3f& left, Eigen::Matrix3f& right);
}

#endif

