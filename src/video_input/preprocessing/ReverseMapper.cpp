#include "ReverseMapper.hpp"

#include "helpers/fill_multi_array.hpp"

#include <boost/gil/extension/numeric/sampler.hpp>

#include <cstdio>

namespace doppia
{


    ReverseMapper::ReverseMapper(const dimensions_t &dimensions,
                                 const shared_ptr<const CameraCalibration> &calibration_p)
                                     : input_dimensions(dimensions)
    {

        added_undistortion = false;
        added_homography = false;

        this->calibration_p = calibration_p;

        // t_img setup ---
        {
            t_img.recreate(input_dimensions); // lazy allocation
            t_img_view = boost::gil::view(t_img);
        }

        // lookup_table_p setup --
        {
            lookup_table_p.reset(new lookup_table_t(boost::extents[input_dimensions.y][input_dimensions.x]));

            { // fill the lookup table with default values

                lookup_table_t &lookup_table = *lookup_table_p;

                for (int y = 0; y <  input_dimensions.y; y+=1)
                {
                    for (int x = 0; x < input_dimensions.x; x+=1)
                    {
                        // Clipping
                        point2<float> &the_point = lookup_table[y][x];
                        the_point.x = x;
                        the_point.y = y;

                    } // end of for each x
                } // end of for each y

            }
        } // end of lookup_table_p setup

        return;
    }

    ReverseMapper::~ReverseMapper()
    {
        // nothing to do here
        return;
    }




    /**
 * Create undistortion object given internal calibration and
 * image dimensions.
 *
 * @param fx focal length x
 * @param fy focal length y
 * @param cx camera center x
 * @param cy camera center y
 * @param k1 radial distortion coeff. 1
 * @param k2 radial distortion coeff. 2
 * @param k3 radial distortion coeff. 3
 * @param w image width
 * @param h image height
 **/
    void ReverseMapper::add_undistortion()
    {
        if(added_homography)
        {
            throw std::runtime_error("ReverseMapper undistortion should be set before rectification, not afterwards");
        }

        if(added_undistortion)
        {
            throw std::runtime_error("ReverseMapper::add_undistortion should not be called twice");
        }


        lookup_table_t &lookup_table = *lookup_table_p;

        for (int y = 0; y <  input_dimensions.y; y+=1)
        {
            for (int x = 0; x < input_dimensions.x; x+=1)
            {

                const point2<float> xy(x,y);

                point2<float> &the_point = lookup_table[y][x];
                // where we overwrite the previous value on the lookup table since we are supposed to be called first
                the_point = from_undistorted_to_source(xy);

            } // end of for each x
        } // end of for each y


        added_undistortion = true;

        return;
    }


    const point2<float> ReverseMapper::from_undistorted_to_source(const point2<float> &destination) const
    {
        const float
                focal_length_x = calibration_p->get_focal_length_x(),
                focal_length_y = calibration_p->get_focal_length_x(),
                center_x = calibration_p->get_image_center_x(),
                center_y = calibration_p->get_image_center_y();

        const RadialDistortionParametersVector &distortion = calibration_p->get_radial_distortion_parameters();


        const float &x = destination.x, &y = destination.y;

        // Convert pixel to camera coordinates
        float xf = (x - center_x) / focal_length_x;
        float yf = (y - center_y) / focal_length_y;

        // Apply distortion
        const float x2 = xf * xf;
        const float y2 = yf * yf;
        const float r2 = x2 + y2;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;

        const float val = 1 + r2 * distortion[0] + r4 * distortion[1] + r6 *  distortion[2];

        xf *= val;
        yf *= val;

        // Back to pixel coordinates
        xf = (xf * focal_length_x) + center_x;
        yf = (yf * focal_length_y) + center_y;

        // Clipping
        point2<float> source_point;
        source_point.x = std::max(0.0f, std::min<float>(xf, input_dimensions.x - 1));
        source_point.y = std::max(0.0f, std::min<float>(yf, input_dimensions.y - 1));

        return source_point;
    }

    void ReverseMapper::add_homography_projection(const HomographyMatrix &H, const bool H_is_centered)
    {
        if(added_homography)
        {
            throw std::runtime_error("ReverseMapper::add_homography_projection should not be called twice");
        }

        lookup_table_t &lookup_table = *lookup_table_p;

        const dimensions_t &output_dimensions = input_dimensions;

        const float
                center_x = calibration_p->get_image_center_x(),
                center_y = calibration_p->get_image_center_y();


        point2<float> t_point;

        for (int y = 0; y <  output_dimensions.y; y+=1)
        {
            for (int x = 0; x < output_dimensions.x; x+=1)
            {
                point2<float> &undistorted_point = t_point;

                float u = x, v = y;
                if(H_is_centered)
                {
                    u -= center_x;
                    v -= center_y;
                }

                undistorted_point.x = H(0,0) * u + H(0,1) * v + H(0,2);
                undistorted_point.y = H(1,0) * u + H(1,1) * v + H(1,2);
                const float w = H(2,0) * u + H(2,1) * v + H(2,2);
                undistorted_point.x /= w;
                undistorted_point.y /= w;

                if(H_is_centered)
                {
                    undistorted_point.x += center_x;
                    undistorted_point.y += center_y;
                }


                point2<float> &the_point = lookup_table[y][x];
                // where we overwrite the previous value since we are taking into account previous calls

                if(added_undistortion)
                {
                    // we recompute the undistortion
                    the_point = from_undistorted_to_source(undistorted_point);
                }
                else
                {
                    // we disregard the undistortion since it is not ment to be used
                    the_point = undistorted_point;
                }

            } // end of for each x
        } // end of for each y


        added_homography = true;
        return;
    }


    const point2<float> & ReverseMapper::warp(const point2<int> &point) const
    {

        if(lookup_table_p.get() == NULL)
        {
            throw std::runtime_error("ReverseMapper::undistort lookup_table_p not yet initialized");
        }

        if(point.y < 0 or point.y >= static_cast<int>(lookup_table_p->shape()[0])  )
        {
            throw std::runtime_error("ReverseMapper::undistort input point.y is out of range");
        }

        if(point.x < 0 or point.x >= static_cast<int>(lookup_table_p->shape()[1])  )
        {
            throw std::runtime_error("ReverseMapper::undistort input point.x is out of range");
        }

        return (*lookup_table_p)[point.y][point.x];
    }


    void ReverseMapper::warp(const input_image_view_t &input, const output_image_view_t &output) const
    {
        assert(input.dimensions() == output.dimensions());
        assert(input.dimensions() == t_img.dimensions());

        using namespace boost::gil;

        if((not (added_homography or added_undistortion)) or (lookup_table_p.get() == NULL))
        {
            throw std::runtime_error("Called ReverseMapper::warp without having set the undistortion and/or homography lookup table");
        }
        else
        {
            if(   lookup_table_p->shape()[1] != (size_t) input.width()
                  or lookup_table_p->shape()[0] != (size_t) input.height())
            {
                throw std::runtime_error("ReverseMapper::undistort input size does not match lookup table size");
            }
        }


        typedef output_image_view_t::value_type output_value_type_t;
        const output_value_type_t default_pixel_value(0,0,0);

        typedef lookup_table_t::subarray<1>::type row_view_t;

#pragma omp parallel for
        for (int y = 0; y < output.height(); y+=1)
        {

            t_img_view_t::x_iterator row_iterator = t_img_view.row_begin(y);

            const row_view_t row_lookup_table_view = (*lookup_table_p)[y];
            row_view_t::const_iterator row_lookup_table_it = row_lookup_table_view.begin();

            for (int x = 0; x < output.width(); x+=1, ++row_iterator, ++row_lookup_table_it)
            {
                //const point2<float> &t_point = lookup_table[y][x];
                const point2<float> &t_point = *row_lookup_table_it;

                //output_value_type_t &output_pixel = (*t_img_view.xy_at(x,y));
                output_value_type_t &output_pixel = *row_iterator;

                const bool result_was_set = sample(bilinear_sampler(), input, t_point, output_pixel);

                if (result_was_set == false)
                { // t_point is out of the src range
                    output_pixel = default_pixel_value;
                }
            } // for each x value
        } // for each y value

        copy_pixels(boost::gil::const_view(t_img), output);

        return;
    }

} // end of namespace doppia

