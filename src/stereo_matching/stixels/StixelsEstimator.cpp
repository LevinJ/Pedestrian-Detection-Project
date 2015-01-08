#include "StixelsEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"

// only for do_horizontal_averaging
#include "stereo_matching/stixels/StixelsEstimatorWithHeightEstimation.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <stdexcept>
#include <cstdio>
#include <limits>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelsEstimator");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelsEstimator");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "StixelsEstimator");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelsEstimator");
}

} // end of anonymous namespace



namespace doppia {

/**
  TODO:
  - Fix the weights on the left side of the cost image (clearly darker)
  - Review the weight between object_cost and ground_cost. What is the proper mixage strategy ?
  - Implement the "multiresolution thing", which makes an average of the bottom pixels when computing the u-disparity cost
  - Speed speed speed !
  */


boost::program_options::options_description StixelsEstimator::get_args_options()
{
    boost::program_options::options_description desc("StixelsEstimator options");

    desc.add_options()

            ("stixel_world.use_ground_cost_mirroring",
             boost::program_options::value<bool>()->default_value(false),
             "use the presence of vertical object to penalize the empty areas")

            ("stixel_world.ground_cost_weight",
             boost::program_options::value<float>()->default_value(1.0),
             "relative weight between the ground cost and the objects cost")


            ("stixel_world.ground_cost_threshold",
             boost::program_options::value<float>()->default_value(-1),
             "threshold the ground cost based on a percent of the highest value. "
             "ground_cost_threshold == 0.3 indicates that only values higher than 0.3*highest_ground_cost_value will be allowed. "
             "ground_cost_threshold <= 0 indicates that no thresholding should be applied")


            ("stixel_world.u_disparity_boundary_diagonal_weight",
             boost::program_options::value<float>()->default_value(0),
             "when solving the dynamic program that estimates the bottom of the stixels, "
             "the diagonal weight penalizes using diagonals (occlusions) "
             "values in the range [0, 20] are reasonable")

            ;

    return desc;
}


StixelsEstimator::StixelsEstimator()
    : BaseStixelsEstimator(*reinterpret_cast<MetricStereoCamera *>(NULL),
                           2.0,  // expected object height
                           30, // minimum object height in pixels
                           0), // stixels width
      u_disparity_boundary_diagonal_weight(20),
      use_ground_cost_mirroring(false),
      ground_cost_weight(1.0),
      ground_cost_threshold(-1)
{
    // nothing to do here
    // this constructor should only be used for unit testing
    return;
}

int check_stixel_width(const int stixel_width)
{
    if(stixel_width != 1)
    {
        log_warning() << "Current implementation of StixelsEstimator ignores stixels_width (and always uses width == 1)" << std::endl;
    }

    return 1;
}


StixelsEstimator::StixelsEstimator(
        const boost::program_options::variables_map &options,
        const MetricStereoCamera &camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        const int stixel_width)
    : BaseStixelsEstimator(camera, expected_object_height, minimum_object_height_in_pixels,
                           check_stixel_width(stixel_width))
{

    use_ground_cost_mirroring = get_option_value<bool>(options, "stixel_world.use_ground_cost_mirroring");
    ground_cost_weight = get_option_value<float>(options, "stixel_world.ground_cost_weight");
    ground_cost_threshold = get_option_value<float>(options, "stixel_world.ground_cost_threshold");

    if(ground_cost_threshold >= 1)
    {
        throw std::invalid_argument("stixel_world.ground_cost_threshold should be < 1");
    }

    u_disparity_boundary_diagonal_weight = get_option_value<float>(options, "stixel_world.u_disparity_boundary_diagonal_weight");

    max_cost_value = 0;

    return;
}

StixelsEstimator::~StixelsEstimator()
{
    // nothing to do here
    return;
}


void StixelsEstimator::set_disparity_cost_volume(const boost::shared_ptr<DisparityCostVolume> &cost_volume_p, const float max_cost_value_)
{
    pixels_cost_volume_p = cost_volume_p;
    max_cost_value = max_cost_value_;
    return;
}


void StixelsEstimator::set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right)
{
    // by default we do nothing, child classes may do something specific with these images
    return;
}

/// Provide the best estimate available for the ground plane
void StixelsEstimator::set_ground_plane_estimate(const GroundPlane &ground_plane,
                                                 const GroundPlaneEstimator::line_t &v_disparity_ground_line)
{
    if(not pixels_cost_volume_p)
    {
        throw std::runtime_error("Sorry, you need to call StixelsEstimator::set_ground_disparity_cost_volume before StixelsEstimator::set_ground_plane_estimate");
    }

    the_ground_plane = ground_plane;
    the_v_disparity_ground_line = v_disparity_ground_line;

    const int num_rows = pixels_cost_volume_p->rows();
    const int num_disparities = pixels_cost_volume_p->disparities();
    set_v_disparity_line_bidirectional_maps(num_rows, num_disparities);
    set_v_given_disparity(num_rows, num_disparities);

    return;
}

void StixelsEstimator::compute()
{
    // create the disparity space image --
    // (using estimated ground plane)
    compute_disparity_space_cost();

    // find the optimal ground-obstacle boundary --
    // (using dynamic programming)
    compute_ground_obstacle_boundary();

    return;
}

const StixelsEstimator::u_disparity_cost_t &StixelsEstimator::get_u_disparity_cost() const
{
    return u_disparity_cost;
}

const StixelsEstimator::u_disparity_cost_t &StixelsEstimator::get_object_u_disparity_cost() const
{
    return object_u_disparity_cost;
}

const StixelsEstimator::u_disparity_cost_t &StixelsEstimator::get_ground_u_disparity_cost() const
{
    return ground_u_disparity_cost;
}

const StixelsEstimator::u_disparity_cost_t &StixelsEstimator::get_M_cost() const
{
    return M_cost;
}

const std::vector<int> &StixelsEstimator::get_u_disparity_ground_obstacle_boundary() const
{
    return u_disparity_ground_obstacle_boundary;
}

const std::vector<int> &StixelsEstimator::get_u_v_ground_obstacle_boundary() const
{
    return u_v_ground_obstacle_boundary;
}


void column_wise_cost_normalization(Eigen::MatrixXf &cost)
{

    for(int c=0; c < cost.cols(); c+=1)
    {
        const float col_max_value = cost.col(c).maxCoeff();
        const float col_min_value = cost.col(c).minCoeff();
        cost.col(c).array() -= col_min_value;

        if(col_max_value > col_min_value)
        {
            const float scaling = 1.0f / (col_max_value - col_min_value);
            cost.col(c) *= scaling;
        }
    } // end of "for each column"

    return;
}

void row_wise_cost_normalization(Eigen::MatrixXf &cost)
{

    for(int r=0; r < cost.rows(); r+=1)
    {
        const float row_max_value = cost.row(r).maxCoeff();
        const float row_min_value = cost.row(r).minCoeff();
        cost.row(r).array() -= row_min_value;

        if(row_max_value > row_min_value)
        {
            const float scaling = 1.0f / (row_max_value - row_min_value);
            cost.row(r) *= scaling;
        }
    } // end of "for each column"

    return;
}

// helper object for coefficient wise operations over Eigen matrices
template<typename Scalar>
struct CwiseThreshold {

    Scalar inferior;

    CwiseThreshold(const Scalar& inferior_)
        : inferior(inferior_)
    {
        // nothing to do here
        return;
    }
    const Scalar operator()(const Scalar& x) const
    {
        return x < inferior ? 0 : x;
    }

};


void StixelsEstimator::compute_disparity_space_cost()
{
    // Here (u,v) refers to the 2d image plane, just like (x,y) or (cols, rows)

    const int num_rows = pixels_cost_volume_p->rows();
    const size_t num_columns = pixels_cost_volume_p->columns();
    const size_t num_disparities = pixels_cost_volume_p->disparities();

    static bool first_call = true;

    if(  v_given_disparity.size() != num_disparities or
         disparity_given_v.size() != pixels_cost_volume_p->rows())
    {
        throw std::runtime_error("StixelsEstimator::compute_disparity_space_cost "
                                 "called before StixelsEstimator::set_v_disparity_line_bidirectional_maps");
    }

    const bool do_averaging_test = false;
    //const bool do_averaging_test = true;
    if (do_averaging_test)
    {
        if(first_call)
        {
            log_warning() << "StixelsEstimator::compute_disparity_space_cost is using averaging_test" << std::endl;
        }
        const float max_cost_value = this->max_cost_value; // cost_volume_estimator_p->get_maximum_cost_per_pixel();
        //const int horizontal_kernel_size = 11; // [pixels]
        const int horizontal_kernel_size = 3; // [pixels]
        boost::shared_ptr<DisparityCostVolume> filtered_pixels_cost_volume_p(new DisparityCostVolume());

        filtered_pixels_cost_volume_p->resize(*pixels_cost_volume_p);

        do_horizontal_averaging(horizontal_kernel_size, *pixels_cost_volume_p, max_cost_value, *filtered_pixels_cost_volume_p);

        pixels_cost_volume_p = filtered_pixels_cost_volume_p; // replace the old pixel volume with the filtered one
    }


    // reset and resize the object_cost and ground_cost
    // Eigen::MatrixXf::Zero(rows, cols)
    object_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);
    ground_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);

    typedef DisparityCostVolume::range_t range_t;
    typedef DisparityCostVolume::const_data_2d_view_t const_data_2d_view_t;
    typedef DisparityCostVolume::const_data_1d_view_t const_data_1d_view_t;
    typedef DisparityCostVolume::const_data_2d_subarray_t const_data_2d_subarray_t;
    typedef DisparityCostVolume::const_data_1d_subarray_t const_data_1d_subarray_t;

    const bool use_fast_access = false;
    const bool search_nearby_disparities = false;

    if(use_fast_access)
    {
        //#pragma omp parallel
        for(int v=0; v < num_rows; v += 1)
        { // iterate over the rows

            const_data_2d_subarray_t column_disparity_slice =
                    pixels_cost_volume_p->columns_disparities_slice(v);

            const int d_at_v = disparity_given_v[v];
            const size_t d_min_one = std::max(d_at_v - 1, 0);
            const size_t d_plus_one = std::min(d_at_v + 1, static_cast<int>(num_disparities) - 1);

            for(size_t u = 0; u < num_columns; u += 1)
            { // iterate over the columns

                const_data_1d_subarray_t disparity_slice = column_disparity_slice[u];

                //#pragma omp for
                for(size_t d = 0; d < num_disparities; d += 1)
                {
                    // for each (u, disparity) value accumulate over the vertical axis --
                    const int minimum_v_for_disparity = top_v_for_stixel_estimation_given_disparity[d];
                    const int ground_obstacle_v_boundary = v_given_disparity[d];

                    //const int ground_obstacle_v_boundary_plus = ground_obstacle_v_boundary + 2;
                    //const int ground_obstacle_v_boundary_minus = ground_obstacle_v_boundary - 2;

                    if(v < minimum_v_for_disparity)
                    {
                        // we skip the upper pixels in the image
                        // which are above the expected objects
                        continue;
                    }
                    else if(v < ground_obstacle_v_boundary)
                        //if(v < ground_obstacle_v_boundary_minus)
                    { // if above the ground-object boundary
                        // we accumulate into the object cost
                        float &object_cost = object_u_disparity_cost(d, u);
                        object_cost += disparity_slice[d];
                    }
                    else
                        //else if(v > ground_obstacle_v_boundary_plus)
                    { // else (below the ground-object boundary)
                        // we accumulate into the ground cost
                        float &ground_cost = ground_u_disparity_cost(d, u);

                        if(search_nearby_disparities)
                        {
                            // for increased robustness we search around too
                            ground_cost += std::min(
                                               std::min(
                                                   disparity_slice[d_at_v],
                                                   disparity_slice[d_min_one]),
                                               disparity_slice[d_plus_one]);
                        }
                        else
                        {
                            ground_cost += disparity_slice[d_at_v];
                        }
                    }
                    //else
                    {
                        // we do nothing, we skip this "ambiguous" pixels
                    }

                } // end of "for each disparity"
            } // end of "for each column"
        } // end of "for each row"
    }
    else
    { // use_fast_access == false

        // note, this is one for the slowest ways to access the cost volume data
        // but is also the one which offers the best parallel access to the data
#pragma omp parallel for
        for(size_t u = 0; u < num_columns; u += 1)
        { // iterate over the columns

            const_data_2d_view_t rows_disparities_slice =
                    pixels_cost_volume_p->rows_disparities_slice(u);

            for(size_t d = 0; d < num_disparities; d += 1)
            {
                const_data_1d_view_t rows_slice =
                        rows_disparities_slice[ boost::indices[range_t()][d] ];

                // for each (u, disparity) value accumulate over the vertical axis --
                const int minimum_v_for_disparity = top_v_for_stixel_estimation_given_disparity[d];
                const size_t ground_obstacle_v_boundary = v_given_disparity[d];
                // precomputed_v_disparity_line already checked for >=0 and < num_rows

                // from tentative ground upwards, over the object -
                float &object_cost = object_u_disparity_cost(d, u);
                for(size_t v=minimum_v_for_disparity; v < ground_obstacle_v_boundary; v+=1)
                {
                    object_cost += rows_slice[v];
                }

                // normalize the object cost -
                object_cost /= (ground_obstacle_v_boundary - minimum_v_for_disparity);
                assert(object_cost >= 0);

                // from tentative ground downards, over the ground -
                float &ground_cost = ground_u_disparity_cost(d, u);
                for(std::size_t v=ground_obstacle_v_boundary; v < rows_slice.size(); v+=1)
                {
                    const int d_at_v = disparity_given_v[v];

                    if(search_nearby_disparities)
                    {
                        // for increased robustness we search around too
                        const size_t d_min_one = std::max(d_at_v - 1, 0);
                        const size_t d_plus_one = std::min(d_at_v + 1, static_cast<int>(num_disparities) - 1);

                        ground_cost += std::min( std::min(
                                                     rows_disparities_slice[v][d_at_v],
                                                     rows_disparities_slice[v][d_min_one]),
                                                 rows_disparities_slice[v][d_plus_one]);
                    }
                    else
                    {
                        ground_cost += rows_disparities_slice[v][d_at_v];
                    }
                } // end of "for each v value below the ground_obstacle_v_boundary"

                // normalize the ground cost -
                if (ground_obstacle_v_boundary < rows_slice.size())
                {
                    ground_cost /= (rows_slice.size() - ground_obstacle_v_boundary);
                }

                assert(ground_cost >= 0);

            } // end of "for each disparity"
        } // end of "for each u"

    } // end of "if use_fast_access"


    // post filtering steps --
    {
        post_process_object_u_disparity_cost(object_u_disparity_cost);
        post_process_ground_u_disparity_cost(ground_u_disparity_cost, num_rows);
    }

    // set the final cost --
    u_disparity_cost = object_u_disparity_cost + ground_u_disparity_cost;

    // mini fix to the "left area initialization issue"
    fix_u_disparity_cost();

    first_call = false;
    return;
}

/// mini trick to fix the "left area initialization issue"
/// as it is the discarted left area of the image creates an homogeneous 0 cost area
/// in this area the algorithms just "goes down" while it should "stay up at disparity 0"
/// then we modify the diagonal line between the zero area and the costs area
/// so that the zero area is not the best choice anymore
void StixelsEstimator::fix_u_disparity_cost()
{
    const float max_u_disparity_cost = u_disparity_cost.maxCoeff();

    // for every pixel from left to right where cost == 0
    // set cost = -0.1
    const float small_negative_value = -1E-5;
    size_t u = 0;
    for(; u < static_cast<size_t>(u_disparity_cost.cols()); u+=1)
    {
        float &t_cost = u_disparity_cost(0, u);
        //if(t_cost == 0)
        if(object_u_disparity_cost(0, u) == 0)
        {
            t_cost = small_negative_value;
        }
        else
        {
            // as soon as we find the first non-zero point, we stop
            break;
        }
    }

    // we fill the lower left triangle
    const size_t start_u = std::max<int>(u - 1, 0);
    for(size_t d = 0; d < static_cast<size_t>(u_disparity_cost.rows()); d+=1)
    {
        const size_t max_u = std::min<size_t>(u_disparity_cost.cols(), start_u + d + 1);
        for(u=start_u; u < max_u; u+=1)
        {
            u_disparity_cost(d,u) = max_u_disparity_cost;
        }
    }
    return;
}

void StixelsEstimator::post_process_object_u_disparity_cost(u_disparity_cost_t &object_cost) const
{
    const bool high_frequency_band_pass_filter = false; // FIXME just for debugging
    if(high_frequency_band_pass_filter)
    {
        high_pass_vertical_cost_filter(object_cost);
    }


    const bool do_horizontal_smoothing = false;
    if(do_horizontal_smoothing)
    {
        low_pass_horizontal_cost_filter(object_cost);
        //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
        //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
        //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
    }

    return;
}

void StixelsEstimator::post_process_ground_u_disparity_cost(u_disparity_cost_t &ground_cost, const int num_rows) const
{

    const int num_columns = ground_cost.cols();
    const int num_disparities = ground_cost.rows();

    const bool threshold_ground_cost = ground_cost_threshold > 0;
    if(threshold_ground_cost)
    {
        const float ground_cost_threshold_fraction = ground_cost_threshold;
        const float ground_cost_threshold = ground_cost.maxCoeff() * ground_cost_threshold_fraction;
        ground_cost = ground_cost.unaryExpr(CwiseThreshold<float>(ground_cost_threshold));
    }


    const bool fix_out_of_image_disparities = (threshold_ground_cost == false);
    if(fix_out_of_image_disparities)
    {
        // 25 is hardcoded in void DisparityCostVolumeEstimator::compute_costs_impl
        // we add 5 pixels for "safety margin"
        const int extra_margin = 5;
        const int last_stereo_matching_row = std::max(0, (num_rows-1) - 25 - extra_margin);
        const int max_disparity_in_image = disparity_given_v[last_stereo_matching_row];

        //printf("max_disparity_in_image == %i\n", max_disparity_in_image);
        //printf("num_disparities == %i\n", num_disparities);

        const int disparity_margin = 10; // FIXME rather arbriratry value
        const int last_disparity = std::max(0, max_disparity_in_image - disparity_margin);
        const int segment_size = num_disparities - last_disparity;

        // for each column
#pragma omp parallel for
        for(int col = 0; col < num_columns; col += 1 )
        {
            const float t_cost = ground_cost(last_disparity, col);
            //const float t_cost = 255;
            ground_cost.col(col).segment(last_disparity, segment_size).setConstant(t_cost);
        } // end of "for each column"

    } // end of "fix_out_of_image_disparities"

    if(use_ground_cost_mirroring)
    {
        // for each column
#pragma omp parallel for
        for(int col = 0; col < num_columns; col += 1 )
        {
            // FIXME hardcoded value
            //const int disparity_jump = 4;
            const int disparity_jump = 10;

            // find the first minima
            int local_minima_disparity_index = 0;


            {
                Eigen::VectorXf delta_cost = Eigen::VectorXf::Zero(num_disparities);

                for(int d=disparity_jump; d < num_disparities; d+=1 )
                {
                    delta_cost(d) = ground_cost(d - disparity_jump, col) - ground_cost(d, col);
                }

                delta_cost.maxCoeff(&local_minima_disparity_index);
            }

            // fill in the lower part of the minima ---
            const float ground_cost_at_minima = ground_cost(local_minima_disparity_index, col);
            const float max_ground_cost = ground_cost.col(col).maxCoeff();

            assert(max_ground_cost >= ground_cost_at_minima);

            //const int ramp_length = disparity_jump;
            const int ramp_length = 50; // FIXME hardcoded value
            const int end_ramp_disparity = std::min(local_minima_disparity_index + ramp_length, num_disparities);
            const float cost_diff = max_ground_cost - ground_cost_at_minima;

            // linea ramp between ground_cost_at_minima and max_ground_cost
            for(int d=local_minima_disparity_index; d < end_ramp_disparity; d+=1 )
            {
                ground_cost(d, col) = ground_cost_at_minima +
                                      cost_diff * static_cast<float>(d - local_minima_disparity_index) / ramp_length;
            }

            for(int d=end_ramp_disparity; d < num_disparities; d+=1 )
            {
                ground_cost(d, col) = max_ground_cost;
            }

        } // end of "for each column"

    } // end of "use_ground_cost_mirroring"


    if(ground_cost_weight != 1.0)
    {
        ground_cost *= ground_cost_weight;
    }

    return;
}

void StixelsEstimator::high_pass_vertical_cost_filter(u_disparity_cost_t &cost) const
{

    const int num_columns = cost.cols();
    const int num_rows = cost.rows();

    Eigen::MatrixXf
            filtered_cost = cost,
            t_cost = filtered_cost;

    const int num_vertical_filter_passes = 10; // 5, 20

    for(int col=0; col < num_columns; col += 1)
    {
        for(int i=0; i < num_vertical_filter_passes; i+=1)
        {
            // vertical boxed 3x1 filter

            // first pixels -
            {
                const int row = 0;
                const float current_row_value = filtered_cost(row + 0, col);
                const float previous_row_value = current_row_value;
                const float next_row_value = filtered_cost(row + 1, col);

                t_cost(row, col) =
                        (current_row_value + previous_row_value + next_row_value) / 3;
            }

            // most pixels -
            for(int row = 1; row < (num_rows - 1); row +=1)
            {
                const float current_row_value = filtered_cost(row + 0, col);
                const float previous_row_value = filtered_cost(row - 1, col);
                const float next_row_value = filtered_cost(row + 1, col);

                t_cost(row, col) =
                        (current_row_value + previous_row_value + next_row_value) / 3;
            } // end of "for each row in column"

            // last pixel -
            {
                const int row = (num_rows - 1);
                const float current_row_value = filtered_cost(row + 0, col);
                const float previous_row_value = filtered_cost(row - 1, col);
                const float next_row_value = current_row_value;

                t_cost(row, col) =
                        (current_row_value + previous_row_value + next_row_value) / 3;
            }

            // copy data -
            filtered_cost = t_cost;
        } // end of "num filter passes"
    } // end of "for each column"

    cost = cost - filtered_cost;

    return;
}

void StixelsEstimator::low_pass_horizontal_cost_filter(u_disparity_cost_t &cost) const
{
    // since our costs have "borders"
    // computing correctly the first and last pixels of each row is not important

    const int num_columns = cost.cols();
    const int num_rows = cost.rows();

    Eigen::MatrixXf
            filtered_cost = cost,
            t_cost = filtered_cost;


    filtered_cost = cost;
    t_cost = cost;
    const int num_horizontal_filter_passes = 3; // 5, 20

    for(int row=0; row < num_rows; row += 1)
    {
        for(int i=0; i < num_horizontal_filter_passes; i+=1)
        {
            // horizontal boxed 1x3 filter
            for(int col = 1; col < (num_columns - 1); col +=1)
            {
                const float current_col_value = filtered_cost(row, col + 0);
                const float previous_col_value = filtered_cost(row, col - 1);
                const float next_col_value = filtered_cost(row, col + 1);

                t_cost(row, col) =
                        (current_col_value + previous_col_value + next_col_value) / 3;
            } // end of "for each row in column"

            filtered_cost = t_cost;

        } // end of "num filter passes"
    } // end of "for each column"

    cost = filtered_cost;
    return;
}

void StixelsEstimator::compute_ground_obstacle_boundary()
{
    // run dynamic programming over the disparity space cost image --

    // v1 is significantly faster than v0, but they compute exactly the same values
    //compute_ground_obstacle_boundary_v0();
    compute_ground_obstacle_boundary_v1();

    // v2 should be faster than v1 (on paper), but it is not (in practice)
    // v1 is one instance where "computing more is faster than computing less"
    //compute_ground_obstacle_boundary_v2();

    return;
} // end of StixelsEstimator::compute_ground_obstacle_boundary


inline void StixelsEstimator::compute_ground_obstacle_boundary_v0()
{
    // run dynamic programming over the disparity space cost image --

    // see section III.C of Kubota et al. 2007 paper
    // (see StixelsEstimator class documentation)

    const int num_columns = u_disparity_cost.cols();
    const int num_disparities = u_disparity_cost.rows();

    if(pixels_cost_volume_p)
    {
        if(pixels_cost_volume_p->disparities() != static_cast<size_t>(num_disparities) or
           pixels_cost_volume_p->columns() != static_cast<size_t>(num_columns))
        {
            throw std::runtime_error("StixelsEstimator::compute_ground_obstacle_boundary_v0"
                                     "u_disparity_cost does not match the expected dimensions");
        }
    }

    const float diagonal_weight = u_disparity_boundary_diagonal_weight;

    // right to left pass --

    // Kubota et al. 2007 penalizes using object cost
    // this is completelly arbritrary, we use here
    // object_cost + ground_cost

    //const u_disparity_cost_t &c_i_cost = ground_u_disparity_cost;
    const u_disparity_cost_t &c_i_cost = object_u_disparity_cost;
    //const u_disparity_cost_t &c_i_cost = u_disparity_cost;

    {
        //M_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);

        // we first copy all m_i(d_i) values
        M_cost = u_disparity_cost;

        for(int column = num_columns - 2; column >=0; column -= 1)
        {
            // equation 3 with d_{i-1} replaced by e
            // we do min instead of max because we are using correlation cost
            // M_i(d_i) = m_i(d_i) + min_e[ M_{i-1}(e) - c_i(d_i, e) ]

#pragma omp parallel for
            for(int d=0; d < num_disparities; d+=1)
            {
                float min_M_minus_c = std::numeric_limits<float>::max();

                for(int e=0; e < num_disparities; e+=1)
                {
                    // implementing the definition of c_i(d_i,e) at equation 5
                    // c is c_i(d, e);
                    float c = 0;
                    const int e_minus_one = e-1;

                    if(d > e_minus_one)
                    {
                        c = 0;
                    }
                    else if (d == e_minus_one)
                    {
                        c = -diagonal_weight - c_i_cost(d, column);
                        //c = -diagonal_weight;
                    }
                    else
                    { // d < e_minus_one
                        // this is not a candidate for min[...]
                        // c = -infinity
                        // (numeric_limits::max should not be used, to avoid float overflow)
                        //c = -10E4;
                        continue;
                    }

                    const float t_cost = M_cost(e, column + 1) - c;

                    // we do min instead of max because we are using correlation cost
                    min_M_minus_c = std::min(t_cost, min_M_minus_c);
                } // end of "for each disparity e"

                //printf("right to left pass M_cost(%i, %i) += %.3f\n",
                //       d, column, min_M_minus_c);
                M_cost(d, column) += min_M_minus_c;
            } // end of "for each disparity d"
        } // end of "for each column", i.e. "for each u value"
    } // end of right to left pass

    // left to right pass --
    {
        const u_disparity_cost_t &const_M_cost = M_cost;

        u_disparity_ground_obstacle_boundary.resize(num_columns);

        // we set the first value directly
        {
            int &d_star = u_disparity_ground_obstacle_boundary[0];
            // minCoeff takes the "lowest index", but
            // we search for the maximum index with minimum value
            float t_cost = std::numeric_limits<float>::max();
            for(int d=0; d < num_disparities; d+=1)
            {
                const float &m_cost = const_M_cost(d, 0);
                if(m_cost <= t_cost)
                {
                    d_star = d;
                    t_cost = m_cost;
                }
                //printf("M(d, 0) == %.3f\n", const_M_cost(d, 0));
            }
        }

        // the rest are recursively
        for(int column = 1; column < num_columns; column += 1)
        {
            const int previous_d_star = u_disparity_ground_obstacle_boundary[column - 1];

            int &d_star = u_disparity_ground_obstacle_boundary[column];
            float min_M_minus_c = std::numeric_limits<float>::max();

            for(int e=0; e < num_disparities; e+=1)
            {
                // implementing the definition of c_i(d_i,e) at equation 5
                // now applied to equation 4
                // c is c_{i+1}(d_star_{i+1}, e)
                float c = 0;

                const int e_minus_one = e-1;
                if(previous_d_star > e_minus_one)
                {
                    c = 0;
                }
                else if (previous_d_star == e_minus_one)
                {
                    c = -diagonal_weight - c_i_cost(previous_d_star, column - 1);
                    // on the left to right pass there is no need to re-penalize the diagonal jumps
                    //c = -ground_u_disparity_cost(previous_d_star, column - 1);
                    //c = -diagonal_weight;
                }
                else
                { // previous_d_star < e_minus_one
                    // this is not a candidate for min[...]
                    // c = -infinity,
                    // (numeric_limits::max should not be used, to avoid float overflow)
                    //c = -10E4;
                    continue;
                }

                const float t_cost = const_M_cost(e, column) - c;


                // we do min instead of max because we are using correlation cost
                if(t_cost <= min_M_minus_c)
                {
                    d_star = e;
                    min_M_minus_c = t_cost;
                }
            }  // end of "for each disparity e"

            // d_start is now set

        } // end of "for each column", i.e. "for each u value"

    } // end of left to right pass

    // at this point u_disparity_ground_obstacle_boundary is now set

    // set u_v_ground_obstacle_boundary and the stixels outputs --
    u_disparity_boundary_to_stixels();

    return;
} // end of StixelsEstimator::compute_ground_obstacle_boundary_v0


inline void StixelsEstimator::compute_ground_obstacle_boundary_v1()
{
    // run dynamic programming over the disparity space cost image --

    // see section III.C of Kubota et al. 2007 paper
    // (see StixelsEstimator class documentation)

    const int
            num_columns = u_disparity_cost.cols(),
            num_disparities = u_disparity_cost.rows();

    if(pixels_cost_volume_p)
    {
        if(pixels_cost_volume_p->disparities() != static_cast<size_t>(num_disparities) or
           pixels_cost_volume_p->columns() != static_cast<size_t>(num_columns))
        {
            throw std::runtime_error("StixelsEstimator::compute_ground_obstacle_boundary_v1 "
                                     "u_disparity_cost does not match the expected dimensions");
        }
    }

    const float diagonal_weight = u_disparity_boundary_diagonal_weight;

    // right to left pass --

    // Kubota et al. 2007 penalizes using object cost
    // this is completelly arbritrary, we use here
    // object_cost + ground_cost

    //const u_disparity_cost_t &c_i_cost = ground_u_disparity_cost;
    const u_disparity_cost_t &c_i_cost = object_u_disparity_cost;
    //const u_disparity_cost_t &c_i_cost = u_disparity_cost;

    {
        //M_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);

        // we first copy all m_i(d_i) values
        M_cost = u_disparity_cost;

#pragma omp parallel
        for(int column = num_columns - 2; column >=0; column -= 1)
        {
            // equation 3 with d_{i-1} replaced by e
            // we do min instead of max because we are using correlation cost
            // M_i(d_i) = m_i(d_i) + min_e[ M_{i-1}(e) - c_i(d_i, e) ]

            const int next_column = column + 1;

            // with or without static scheduling we got roughly the same performance
            // however, in this case, "default" seems to be the best choice
#pragma omp for //schedule(static)
            for(int d=0; d < num_disparities; d+=1)
            {
                float min_M_minus_c = std::numeric_limits<float>::max();

                // e_minus_one < d
                {
                    const int e_end = std::min(d+1, num_disparities);

                    for(int e=0; e < e_end ; e+=1)
                    {
                        // implementing the definition of c_i(d_i,e) at equation 5
                        // c is c_i(d, e);
                        // c = 0
                        //const float t_cost = M_cost(e, next_column) - c; // and c = 0;
                        const float t_cost = M_cost(e, next_column);

                        // we do min instead of max because we are using correlation cost
                        min_M_minus_c = std::min(t_cost, min_M_minus_c);
                    } // end of "for each disparity e"
                }

                // d == e_minus_one
                {
                    const int e = d+1;
                    if(e < num_disparities)
                    {
                        // implementing the definition of c_i(d_i,e) at equation 5
                        // c is c_i(d, e);
                        const float c = -diagonal_weight - c_i_cost(d, column);
                        //const float c = -diagonal_weight;

                        const float t_cost = M_cost(e, next_column) - c;

                        // we do min instead of max because we are using correlation cost
                        min_M_minus_c = std::min(t_cost, min_M_minus_c);
                    }
                }

                //// e_minus_one > d
                //for(int e_minus_one = -1; e_minus_one < d; e+=1)
                //{
                //    // this is not a candidate for min[...]
                //    // c = -infinity
                //    // (numeric_limits::max should not be used, to avoid float overflow)
                //    //c = -10E4;
                //    continue;
                //}

                //printf("right to left pass M_cost(%i, %i) += %.3f\n",
                //       d, column, min_M_minus_c);
                M_cost(d, column) += min_M_minus_c;
            } // end of "for each disparity d"
        } // end of "for each column", i.e. "for each u value"
    } // end of right to left pass

    // left to right pass --
    {
        const u_disparity_cost_t &const_M_cost = M_cost;

        u_disparity_ground_obstacle_boundary.resize(num_columns);

        // we set the first value directly
        {
            int &d_star = u_disparity_ground_obstacle_boundary[0];
            // minCoeff takes the "lowest index", but
            // we search for the maximum index with minimum value
            const_M_cost.col(0).minCoeff(&d_star);
        }

        // the rest are recursively
        for(int column = 1; column < num_columns; column += 1)
        {
            const int previous_d_star = u_disparity_ground_obstacle_boundary[column - 1];

            int &d_star = u_disparity_ground_obstacle_boundary[column];
            float min_M_minus_c = std::numeric_limits<float>::max();

            // e_minus_one < previous_d_star
            {
                const int e_end = std::min(previous_d_star+1, num_disparities);
                for(int e=0; e < e_end; e+=1)
                {
                    //const float c = 0;
                    //const float t_cost = const_M_cost(e, column) - c; // and c = 0
                    const float t_cost = const_M_cost(e, column);

                    // we do min instead of max because we are using correlation cost
                    if(t_cost <= min_M_minus_c)
                    {
                        d_star = e;
                        min_M_minus_c = t_cost;
                    }
                }
            }

            // e_minus_one == previous_d_star
            {
                const int e = previous_d_star+1;
                if(e < num_disparities)
                {
                    const float c = -diagonal_weight - c_i_cost(previous_d_star, column - 1);
                    // on the left to right pass there is no need to re-penalize the diagonal jumps
                    //const float c = -ground_u_disparity_cost(previous_d_star, column - 1);
                    //const float c = -diagonal_weight;

                    const float t_cost = const_M_cost(e, column) - c;

                    // we do min instead of max because we are using correlation cost
                    if(t_cost <= min_M_minus_c)
                    {
                        d_star = e;
                        min_M_minus_c = t_cost;
                    }
                }
            }

            // e_minus_one > previous_d_star
            //{
            // this is not a candidate for min[...]
            // c = -infinity,
            // (numeric_limits::max should not be used, to avoid float overflow)
            //c = -10E4;
            //continue;
            //}

            // end of "for each disparity e" d_start is now set

        } // end of "for each column", i.e. "for each u value"

    } // end of left to right pass

    // at this point u_disparity_ground_obstacle_boundary is now set

    // set u_v_ground_obstacle_boundary and the stixels outputs --
    u_disparity_boundary_to_stixels();

    return;
} // end of StixelsEstimator::compute_ground_obstacle_boundary_v1


inline void StixelsEstimator::compute_ground_obstacle_boundary_v2()
{
    // run dynamic programming over the disparity space cost image --

    // see section III.C of Kubota et al. 2007 paper
    // (see StixelsEstimator class documentation)

    // this version keeps track of the argmax in the right to left pass,
    // so that the left to right pass is faster

    const int
            num_columns = u_disparity_cost.cols(),
            num_disparities = u_disparity_cost.rows();

    if(pixels_cost_volume_p)
    {
        if(pixels_cost_volume_p->disparities() != static_cast<size_t>(num_disparities) or
           pixels_cost_volume_p->columns() != static_cast<size_t>(num_columns))
        {
            throw std::runtime_error("StixelsEstimator::compute_ground_obstacle_boundary_v2 "
                                     "u_disparity_cost does not match the expected dimensions");
        }
    }

    const float diagonal_weight = u_disparity_boundary_diagonal_weight;

    // right to left pass --

    // Kubota et al. 2007 penalizes using object cost
    // this is completelly arbritrary, we use here
    // object_cost + ground_cost

    //const u_disparity_cost_t &c_i_cost = ground_u_disparity_cost;
    const u_disparity_cost_t &c_i_cost = object_u_disparity_cost;
    //const u_disparity_cost_t &c_i_cost = u_disparity_cost;

    {
        // we first copy all m_i(d_i) values
        M_cost = u_disparity_cost;
        const u_disparity_cost_t &const_M_cost = M_cost;

        if((min_M_minus_c_indices.shape()[1] != static_cast<size_t>(num_disparities))
           or (min_M_minus_c_indices.shape()[0] != static_cast<size_t>(num_columns)))
        {
            min_M_minus_c_indices.resize(boost::extents[num_columns][num_disparities]);
            // all accessed values are set, so there is no need to initialize
        }


#pragma omp parallel
        for(int column = num_columns - 2; column >=0; column -= 1)
        {
            // equation 3 with d_{i-1} replaced by e
            // we do min instead of max because we are using correlation cost
            // M_i(d_i) = m_i(d_i) + min_e[ M_{i-1}(e) - c_i(d_i, e) ]

            min_M_minus_c_indices_t::reference min_M_minus_c_indices_column = min_M_minus_c_indices[column];

            const int next_column = column + 1;

            // guided schedule seeems to provide the best performance (better than default and static)
            // however, in this case, "default" seems to be the best choice
#pragma omp for //schedule(guided)
            for(int d=0; d < num_disparities; d+=1)
            {
                float min_M_minus_c = std::numeric_limits<float>::max();
                int min_M_minus_c_index = 0;

                // e_minus_one < d
                {
                    const int e_end = std::min(d+1, num_disparities);

                    for(int e=0; e < e_end ; e+=1)
                    {
                        // implementing the definition of c_i(d_i,e) at equation 5
                        // c is c_i(d, e);
                        // c = 0
                        //const float t_cost = const_M_cost(e, next_column) - c; // and c = 0;
                        const float t_cost = const_M_cost(e, next_column);

                        // we do min instead of max because we are using correlation cost
                        //min_M_minus_c = std::min(t_cost, min_M_minus_c);
                        if(t_cost < min_M_minus_c)
                        {
                            min_M_minus_c = t_cost;
                            min_M_minus_c_index = e;
                        }
                    } // end of "for each disparity e"
                }

                // d == e_minus_one
                {
                    const int e = d+1;
                    if(e < num_disparities)
                    {
                        // implementing the definition of c_i(d_i,e) at equation 5
                        // c is c_i(d, e);
                        const float c = -diagonal_weight - c_i_cost(d, column);
                        //const float c = -diagonal_weight;

                        const float t_cost = const_M_cost(e, next_column) - c;

                        // we do min instead of max because we are using correlation cost
                        //min_M_minus_c = std::min(t_cost, min_M_minus_c);
                        if(t_cost < min_M_minus_c)
                        {
                            min_M_minus_c = t_cost;
                            min_M_minus_c_index = e;
                        }
                    }
                }

                //// e_minus_one > d
                //for(int e_minus_one = -1; e_minus_one < d; e+=1)
                //{
                //    // this is not a candidate for min[...]
                //    // c = -infinity
                //    // (numeric_limits::max should not be used, to avoid float overflow)
                //    //c = -10E4;
                //    continue;
                //}

                //printf("right to left pass M_cost(%i, %i) += %.3f\n",
                //       d, column, min_M_minus_c);
                M_cost(d, column) += min_M_minus_c;
                min_M_minus_c_indices_column[d] = min_M_minus_c_index;
            } // end of "for each disparity d"
        } // end of "for each column", i.e. "for each u value"
    } // end of right to left pass

    // left to right pass --
    {
        u_disparity_ground_obstacle_boundary.resize(num_columns);

        // we set the first value directly
        {
            const u_disparity_cost_t &const_M_cost = M_cost;
            int &d_star = u_disparity_ground_obstacle_boundary[0];
            // minCoeff takes the "lowest index", but
            // we search for the maximum index with minimum value
            const_M_cost.col(0).minCoeff(&d_star);
        }

        if(false)
        {
            // use to guess the time impact of the jumping index
            for(int column = 1; column < num_columns; column += 1)
            {
                u_disparity_ground_obstacle_boundary[column] = 0;
            }

        }
        else
        {
            int &previous_d_star = u_disparity_ground_obstacle_boundary[0];
            int previous_column = 0;

            // the rest are set using the stored min_M_minus_c_indices
            for(int column = 1; column < num_columns; column += 1, previous_column = column)
            {
                int &d_star = u_disparity_ground_obstacle_boundary[column];
                // (no boundary check for speed reasons, in debug mode Eigen does the checks)
                d_star = min_M_minus_c_indices[previous_column][previous_d_star];
                previous_d_star = d_star;
            } // end of "for each column", i.e. "for each u value"
        }
    } // end of left to right pass

    // at this point u_disparity_ground_obstacle_boundary is now set

    // set u_v_ground_obstacle_boundary and the stixels outputs --
    u_disparity_boundary_to_stixels();

    return;
} // end of StixelsEstimator::compute_ground_obstacle_boundary_v2


void StixelsEstimator::u_disparity_boundary_to_stixels()
{
    // dummy version, one stixel per column
    //const int num_columns = pixels_cost_volume_p->columns();
    const int num_columns = u_disparity_cost.cols();
    the_stixels.resize(num_columns);
    u_v_ground_obstacle_boundary.resize(num_columns);

    int previous_disparity = 0;
    for(int u = 0; u < num_columns; u += 1)
    {
        const int &disparity = u_disparity_ground_obstacle_boundary[u];
        // map from disparity to v based on the ground estimate
        const int &bottom_v = v_given_disparity[disparity];
        const int &top_v = expected_v_given_disparity[disparity];

        u_v_ground_obstacle_boundary[u] = bottom_v;
        Stixel &t_stixel = the_stixels[u];
        t_stixel.width = 1;
        t_stixel.x = u;
        t_stixel.bottom_y = bottom_v;
        t_stixel.top_y = top_v;
        t_stixel.default_height_value = true;
        t_stixel.disparity = disparity;
        t_stixel.type = Stixel::Unknown;

        // check if u-disparity point corresponds to an occlusion -
        const int next_u = std::min(u + 1, num_columns - 1);
        if(next_u < num_columns)
        {
            // pixels inside a diagonal line "going up" on d are occlusions -
            const int next_disparity = u_disparity_ground_obstacle_boundary[next_u];
            const bool is_occluded =
                    ((disparity -previous_disparity) == 1) and
                    ((next_disparity - disparity) == 1);

            if(is_occluded)
            {
                t_stixel.type = Stixel::Occluded;
            }
        } // end of "if next column exists"


        previous_disparity = disparity;

    } // end of "for each column"

    return;
}



} // end of namespace doppia
