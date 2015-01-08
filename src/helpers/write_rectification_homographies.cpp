#include "write_rectification_homographies.hpp"
#include "stereo_rectification_homographies.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <ostream>
#include <cassert>

namespace
{
	void copy_homography(doppia_protobuf::Homography3d& dst, const Eigen::Matrix3f& src)
	{
		dst.set_h11(src(0,0));
		dst.set_h12(src(0,1));
		dst.set_h13(src(0,2));
		dst.set_h21(src(1,0));
		dst.set_h22(src(1,1));
		dst.set_h23(src(1,2));
		dst.set_h31(src(2,0));
		dst.set_h32(src(2,1));
		dst.set_h33(src(2,2));
	}

	void write(google::protobuf::io::ZeroCopyOutputStream& stream, const Eigen::Matrix3f& left, const Eigen::Matrix3f& right)
	{
		doppia_protobuf::StereoRectificationHomographies stereo_homographies;
		copy_homography(*stereo_homographies.mutable_left(), left);
		copy_homography(*stereo_homographies.mutable_right(), right);
		google::protobuf::TextFormat::Print(stereo_homographies, &stream);
	}
}

namespace doppia
{
	void write_rectification_homographies(std::ostream& os, const Eigen::Matrix3f& left, const Eigen::Matrix3f& right)
	{
		google::protobuf::io::OstreamOutputStream stream(&os);
		write(stream, left, right);
	}

	void write_rectification_homographies(FILE *fout, const Eigen::Matrix3f& left, const Eigen::Matrix3f& right)
	{
		assert(fout);
		google::protobuf::io::FileOutputStream stream(fileno(fout));
		write(stream, left, right);
	}
}
