
// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include "AbstractStereoMatcher.hpp"

#include "helpers/get_option_value.hpp"

#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <cassert>



namespace doppia
{

using std::printf;
using namespace boost;
using namespace boost::program_options;
using namespace boost::gil;

program_options::options_description AbstractStereoMatcher::get_args_options()
{
    program_options::options_description desc("AbstractStereoMatcher options");


    desc.add_options()

            ("rgb2gray,g", program_options::value<bool>()->default_value(false),
             "convert the input color images to gray images")

            ("max_disparity,d",
             program_options::value<int>()->default_value(64), "matching disparities are expected to be in the range [0, max_disparity]")
            ;


    return desc;
}


AbstractStereoMatcher::AbstractStereoMatcher(const program_options::variables_map &options)
{

    convert_from_rgb_to_gray = get_option_value<bool>(options, "rgb2gray");
    max_disparity = get_option_value<int>(options, "max_disparity");

    first_disparity_map_computation = true;
    return;
}

AbstractStereoMatcher::~AbstractStereoMatcher()
{
    // nothing to do here
    return;
}



void AbstractStereoMatcher::set_rectified_images_pair( any_image<input_images_t>::const_view_t &left,  any_image<input_images_t>::const_view_t &right )
{

    assert( left.dimensions() == right.dimensions() );

    if(false)
    {
        // z For integer types, causes printf to expect a size_t sized integer argument.
        printf("AbstractStereoMatcher::set_rectified_images_pair left input image dimensions == (%zi, %zi)\n",
               left.width(), left.height());
    }


    // convert and copy --
    if (convert_from_rgb_to_gray and (left.current_type_is<rgb8c_view_t>() == true) )
    {

        // cast to color view
        rgb8c_view_t
                left_view(left._dynamic_cast<rgb8c_view_t>()),
                right_view(right._dynamic_cast<rgb8c_view_t>());

        // resize destination images
        gray_left_image.recreate(left_view.dimensions());
        gray_right_image.recreate(right_view.dimensions());

        // copy
        copy_and_convert_pixels(left_view, view(gray_left_image));
        copy_and_convert_pixels(right_view, view(gray_right_image));

        // cast to any_image_view
        any_image<AbstractStereoMatcher::input_images_t>::const_view_t gray_left_view(const_view(gray_left_image)), gray_right_view(const_view(gray_right_image));

        // set the gray views
        this->left = gray_left_view;
        this->right = gray_right_view;
    }
    else
    {
        // copy the image references
        this->left = left;
        this->right = right;
    }

    assert( this->left.dimensions() == this->right.dimensions() );
    //printf("this->left.dimensions() == (%zi, %zi)\n", this->left.width(), this->left.height());
    //printf("this->right.dimensions() == (%zi, %zi)\n", this->right.width(), this->right.height());

    // resize disparity map --
    disparity_map_p.reset( new disparity_map_t(left.dimensions() ));
    disparity_map_view = view(*disparity_map_p);
    fill_pixels(disparity_map_view, 0);

    return;
}



void AbstractStereoMatcher::compute_disparity_map()
{

    // if the disparity map was set, then we can guess that the left and right views are set too
    assert(disparity_map_p.get() != NULL);

    if ( left.current_type_is<gray8c_view_t>() )
    {
        const bool record_gray_input_images = false;
        if (record_gray_input_images)
        {
            printf("Creating input_left.png and input_right.png images\n");
            boost::gil::png_write_view("input_left.png", left);
            boost::gil::png_write_view("input_right.png", right);
        }

        compute_disparity_map(left._dynamic_cast<gray8c_view_t>(), right._dynamic_cast<gray8c_view_t>());
    }
    else if ( left.current_type_is<rgb8c_view_t>() )
    {
        compute_disparity_map(left._dynamic_cast<rgb8c_view_t>(), right._dynamic_cast<rgb8c_view_t>());
    }
    else
    {
        throw std::runtime_error("Calling AbstractStereoMatcher::compute_disparity_map with unsuported images type");
    }

    first_disparity_map_computation = false;
    return;
}

AbstractStereoMatcher::disparity_map_t::const_view_t AbstractStereoMatcher::get_disparity_map()
{
    return const_view(*disparity_map_p);
}


} // end of namespace doppia
