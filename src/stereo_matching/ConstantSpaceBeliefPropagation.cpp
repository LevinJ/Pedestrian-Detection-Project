#include "ConstantSpaceBeliefPropagation.hpp"

#include "qingxiong_yang/qx_basic.hpp"

#include "qingxiong_yang/qx_csbp.hpp"
#include "qingxiong_yang/qx_post_processing.hpp"

#include "helpers/get_option_value.hpp"

#include <omp.h>

namespace doppia
{

    using namespace std;
    using namespace boost;
    using namespace boost::gil;


    program_options::options_description ConstantSpaceBeliefPropagation::get_args_options()
    {
        program_options::options_description desc("ConstantSpaceBeliefPropagation options");

        desc.add_options()

            /*        ("pixels_matching",
                      program_options::value<string>()->default_value("sad"),
                      "pixels matching method: birchfield or sad")
                      */
            ("csbp.local",
             program_options::value<bool>()->default_value(true),
             "use local minima instead of global minima during belief propagation")

            ("csbp.scales",
             program_options::value<int>()->default_value(qx_csbp_base::Default_Nr_Scales),
             "number of multi-resolution scales used. 5 >= scales >= 1")

            ("csbp.iterations_per_scale",
             program_options::value<int>()->default_value(5),
             "number of iterations on every scale")

            ("csbp.k0",
             program_options::value<int>()->default_value(qx_csbp_base::Default_Nr_Planes_Base_Level),
             "number of possible disparity values at the finest resolution")

            ("csbp.cost_single_jump",
             program_options::value<short>()->default_value(qx_csbp_base::Default_Cost_Discontinuity_Single_Jump),
             "maximum cost between two pixels inside the belief propagation")

            ("csbp.post_filtering_radius",
             program_options::value<int>()->default_value(9),
             "post-process the disparity map to improve the object edges using a bilateral filter. Radius < 3 will disable the filtering.")

            ("csbp.post_filtering_iterations",
             program_options::value<int>()->default_value(1),
             "number of passes of the post-processing bilateral filter. Iterations < 1 will disable the filtering.")

            ;

        return desc;
    }


    ConstantSpaceBeliefPropagation::ConstantSpaceBeliefPropagation(const program_options::variables_map &options) :
        AbstractStereoBlockMatcher(options)
    {

        use_local_minima = get_option_value<bool>(options, "csbp.local");

        num_scales = get_option_value<int>(options, "csbp.scales");
        iterations_per_scale = get_option_value<int>(options, "csbp.iterations_per_scale");
        max_nr_plane = get_option_value<int>(options, "csbp.k0");
        discontinuity_cost_single_jump = get_option_value<short>(options, "csbp.cost_single_jump");

        bilateral_filter_radius = get_option_value<int>(options, "csbp.post_filtering_radius");
        bilateral_filter_iterations = get_option_value<int>(options, "csbp.post_filtering_iterations");

        do_post_processing  = (bilateral_filter_radius >= 3) and (bilateral_filter_iterations >= 1);

        if(num_scales < 1 or num_scales > 5)
        {
            throw std::runtime_error("csbp.scales is expected to be in the range [1,5]");
        }

        /*
           const string pixels_matching_string = get_option_value<string>(options, "pixels_matching");

           if (pixels_matching_string.empty() or (pixels_matching_string.compare("birchfield") == 0))
           {
           pixels_matching_method = QX_DEF_STEREO_CORR_VOL_METHOD_SMOOTH_BIRCHFIELD;
           }
           else if (pixels_matching_string.compare("sad") == 0)
           {
           pixels_matching_method = QX_DEF_STEREO_CORR_VOL_METHOD_ABSOLUTE_DIFFERENCE;
           }
           else
           {
           throw std::runtime_error("Unknown 'pixels_matching' value");
           }
           */


        return;
    }


    ConstantSpaceBeliefPropagation::~ConstantSpaceBeliefPropagation()
    {

        // nothing to do here
        // the destructor of qx_local_csbp or qx_global_csbp takes care of removing all objects
        return;
    }

    void ConstantSpaceBeliefPropagation::set_rectified_images_pair(
            gil::any_image<input_images_t>::const_view_t &left,
            gil::any_image<input_images_t>::const_view_t &right)
    {

        this->AbstractStereoMatcher::set_rectified_images_pair(left, right);

        if(csbp_p.get() == NULL)
        {

            const int width = disparity_map_p->dimensions().x;
            const int height = disparity_map_p->dimensions().y;

            qx_csbp_base::options opts = qx_csbp_base::get_default_options();

            opts.use_local_minima = use_local_minima;
            opts.nr_planes_base_level = max_nr_plane;
            opts.nr_planes = max_disparity;
            opts.cost_discontinuity_single_jump = discontinuity_cost_single_jump;
            opts.nr_scales = num_scales;

            std::vector<int> iterations(num_scales, iterations_per_scale);

            csbp_p.reset(new qx_csbp_rgb<unsigned char>(height, width, opts, &iterations[0]));

            const int num_colors = 3; // TODO this is easy enough to generalize in 

            qx_image_left_mem.reset(new unsigned char[width*height*num_colors]);
            qx_image_right_mem.reset(new unsigned char[width*height*num_colors]);
            qx_image_left = qx_image_left_mem.get();
            qx_image_right = qx_image_right_mem.get();

            if(do_post_processing)
            {  
                disparity_refinement_p.reset(new qx_disparity_map_bf_refinement(height, width, opts.nr_planes, bilateral_filter_radius));
            }
        }
        else
        {
            // already initalized, nothing to do here

        }

        return;
    }


    void ConstantSpaceBeliefPropagation::compute_disparity_map(gray8c_view_t &/*left*/, gray8c_view_t &/*right*/, const bool left_right_are_inverted)
    {

        if (left_right_are_inverted)
        {
            throw std::runtime_error("ConstantSpaceBeliefPropagation does not implement right to left matching yet");
        }

        throw std::runtime_error("ConstantSpaceBeliefPropagation::compute_disparity_map does not support gray images yet");

        return;
    }


    void ConstantSpaceBeliefPropagation::compute_disparity_map(rgb8c_view_t  &left, rgb8c_view_t &right, const bool /*left_right_are_inverted*/)
    {

        static int num_iterations = 0;
        static double cumulated_time = 0;
        const int num_iterations_for_timing = 50;

        // copy the images
        {

            const int num_rows = left.height();
            const int num_cols = left.width();
            const int num_colors = 3;

            for (int row=0; row < num_rows; row+=1)
            {
                rgb8c_view_t::x_iterator left_row_it = left.row_begin(row);
                rgb8c_view_t::x_iterator right_row_it = right.row_begin(row);
                for (int col=0; col < num_cols; col+=1, ++left_row_it, ++right_row_it)
                {
                    for (int color=0;color < num_colors; color+=1)
                    {
                        const int pixel_index = row*num_cols*num_colors + num_colors*col + color;
                        qx_image_left[pixel_index] = (*left_row_it)[color];
                        qx_image_right[pixel_index] = (*right_row_it)[color];
                    }
                }
            }

        }

        const double start_wall_time = omp_get_wtime();

        // compute the disparity
        short *qx_disparity = NULL;
        if(csbp_p.get() != NULL)
        {
            qx_disparity = csbp_p->disparity(qx_image_left, qx_image_right);
        }
        else
        {
            throw std::runtime_error("Failed to properly initialize ConstantSpaceBeliefPropagation");
        }

        assert(qx_disparity != NULL);
        cumulated_time += omp_get_wtime() - start_wall_time;
        // we do not count the refinement time (just to be generous)

        if(disparity_refinement_p.get() != NULL)
        {
            // from uchar* to uchar** and uchar***

            const int w = disparity_map_p->dimensions().x;
            const int h = disparity_map_p->dimensions().y;

            qx_array_2d<short> disparity(h, w);
            qx_array_3d<unsigned char> image_left(h, w, 3);

            for (int row=0; row < left.height(); row+=1)
            {
                rgb8c_view_t::x_iterator left_row_it = left.x_at(0, row);
                for (int col=0; col < left.width(); col+=1, ++left_row_it)
                {
                    disparity[row][col] = qx_disparity[row*left.width() + col];

                    for (unsigned int color=0;color < 3; color+=1)
                    {
                        image_left[row][col][color] = (*left_row_it)[color];
                    }
                }
            }

            // apply bilateral filter at edges
            disparity_refinement_p->disparity_refinement(disparity, image_left,
                    bilateral_filter_iterations);

            // copy the resulting disparity
            for (int row=0; row < left.height(); row+=1)
            {
                disparity_map_t::view_t::x_iterator disparity_row_it = disparity_map_view.row_begin(row);
                for (int col=0; col < left.width(); col+=1, ++disparity_row_it)
                {
                    (*disparity_row_it)[0] = disparity[row][col];
                }
            }
        }
        else
        {
            // copy the resulting disparity
            for (int row=0; row < left.height(); row+=1)
            {
                disparity_map_t::view_t::x_iterator disparity_row_it = disparity_map_view.row_begin(row);
                for (int col=0; col < left.width(); col+=1, ++disparity_row_it)
                {
                    (*disparity_row_it)[0] = qx_disparity[row*left.width() + col];
                }
            }

        }

        num_iterations += 1;
        if((num_iterations % num_iterations_for_timing) == 0)
        {
            printf("Average ConstantSpaceBeliefPropagation::compute_disparity_map speed %.2lf [Hz] (in the last %i iterations) (non multithreaded)\n",
                    num_iterations_for_timing / cumulated_time, num_iterations_for_timing );
            cumulated_time = 0;
        }

        return;
    }

} // end of namespace doppia



