


#include "SumOfAbsoluteDifference.hpp"

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include "helpers/for_each.hpp"

#include <climits>

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

namespace doppia
{



using namespace std;
using namespace boost;
using namespace boost::gil;


program_options::options_description SumOfAbsoluteDifference::get_args_options(void)
{
    program_options::options_description desc("SumOfAbsoluteDifference options");

    desc.add_options();

    return desc;
}

SumOfAbsoluteDifference::SumOfAbsoluteDifference(const program_options::variables_map &options) :
        AbstractStereoBlockMatcher(options)
{



    return;
}


SumOfAbsoluteDifference::~SumOfAbsoluteDifference()
{

    // nothing to do here
    return;
}

void SumOfAbsoluteDifference::set_rectified_images_pair( gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right)
{

    this->AbstractStereoMatcher::set_rectified_images_pair(left, right);


    return;
}



/**
  Based on IVT - Integrating Vision Toolkit class CStereoVision
  */
void SumOfAbsoluteDifference::compute_disparity_map(gray8c_view_t &left, gray8c_view_t &right, bool left_right_are_inverted)
{
    //bool CStereoVision::Process(const CByteImage *pLeftImage, const CByteImage *pRightImage, CByteImage *pDepthImage,
    //       int nWindowSize, int d1, int d2, int d_step, int threshold)

    if (left_right_are_inverted)
    {
        throw std::runtime_error("SumOfAbsoluteDifference does not implement right to left matching yet");
    }

    const int disparity_step = 1;
    assert(disparity_step != 0);

    assert(window_width == window_height);
    const int window_size = window_width;
    const int half_window_size = (window_size - 1) / 2;

    const int threshold = 1000;
    const int maximum_allowed_error = window_size * window_size * threshold;

    const int disparity1 = 0, disparity2 = 50;
    assert((disparity_step > 0 && disparity1 < disparity2) || (disparity_step < 0 && disparity1 > disparity2));

    const int width = left.width();
    const int height = left.height();

    const int max_row = height - window_size;
    const int max_column = width - window_size;
    //  int offset = 0;

#pragma omp parallel for
    for (int row = 0; row < max_row; ++row)
    {
        for (int column = 0; column < max_column; ++column)
        {
            // const int diff = width - window_size;

            int best_error = INT_MAX;
            int best_disparity = 0;

            const int max_disparity = disparity2 < column ? disparity2 : column;

            // find correlation
            for (int t_disparity = disparity1; t_disparity <= max_disparity; t_disparity += disparity_step)
            {
                // int offset2 = offset;
                int error = 0;

                for (int y = 0; y < window_size; y++)
                {
                    for (int x = 0; x < window_size; x++)
                    {

                        const int left_row = row + y, right_row = left_row;
                        const int left_column = column + x, right_column = left_column - t_disparity;

                        // const int v = pLeftImageData[offset2] - pRightImageData[offset2 - d];
                        const int v = (*left.at(left_column, left_row))[0] - (*right.at(right_column, right_row))[0];
                        error += v * v;
                        // offset2++;
                    }

                    // offset2 += diff;
                }

                if (error < best_error)
                {
                    best_error = error;
                    best_disparity = t_disparity;
                }
            } // end of "for each disparity in the disparity search range"

            // pDepthImageData[(i + nWindowSize / 2) * width + j + nWindowSize / 2] = best_error < nWindowSize * nWindowSize * threshold ? best_d : 0;
            (*disparity_map_view.at(column + half_window_size, row + half_window_size)) = best_error < maximum_allowed_error ? best_disparity : 0;

            // offset++;
        } // end of "for each column in the image"

        // offset += nWindowSize;
    } // end of "for each row in the image"

    return;
}



void SumOfAbsoluteDifference::compute_disparity_map(rgb8c_view_t  &/*left*/, rgb8c_view_t &/*right*/, bool /*left_right_are_inverted*/)
{

    throw std::runtime_error("SumOfAbsoluteDifference::compute_disparity_map does not support color images yet");
    return;
}

} // end of namespace doppia


