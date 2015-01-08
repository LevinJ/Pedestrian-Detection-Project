#include "read_rectification_homographies.hpp"
#include "stereo_rectification_homographies.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <istream>
#include <cassert>

namespace
{
	void copy_homography(Eigen::Matrix3f& dst, const doppia_protobuf::Homography3d& src)
	{
		dst(0,0) = src.h11();
		dst(0,1) = src.h12();
		dst(0,2) = src.h13();
		dst(1,0) = src.h21();
		dst(1,1) = src.h22();
		dst(1,2) = src.h23();
		dst(2,0) = src.h31();
		dst(2,1) = src.h32();
		dst(2,2) = src.h33();
	}

	void read(google::protobuf::io::ZeroCopyInputStream& stream, Eigen::Matrix3f& left, Eigen::Matrix3f& right)
	{
		doppia_protobuf::StereoRectificationHomographies stereo_homographies;
		google::protobuf::TextFormat::Parse(&stream, &stereo_homographies);
		copy_homography(left, stereo_homographies.left());
		copy_homography(right, stereo_homographies.right());
	}
}

namespace doppia
{
	void read_rectification_homographies(std::istream& is, Eigen::Matrix3f& left, Eigen::Matrix3f& right)
	{
		google::protobuf::io::IstreamInputStream stream(&is);
		read(stream, left, right);
	}

	void read_rectification_homographies(FILE *fin, Eigen::Matrix3f& left, Eigen::Matrix3f& right)
	{
		assert(fin);
		google::protobuf::io::FileInputStream stream(fileno(fin));
		read(stream, left, right);
	}
}

