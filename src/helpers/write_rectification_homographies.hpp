#ifndef WRITE_RECTIFICATION_HOMOGRAPHIES_HPP
#define WRITE_RECTIFICATION_HOMOGRAPHIES_HPP

#include <Eigen/Core>
#include <iosfwd>
#include <cstdio>

namespace doppia
{
	void write_rectification_homographies(std::ostream& os, const Eigen::Matrix3f& left, const Eigen::Matrix3f& right);
	void write_rectification_homographies(FILE *fout, const Eigen::Matrix3f& left, const Eigen::Matrix3f& right);
}

#endif

