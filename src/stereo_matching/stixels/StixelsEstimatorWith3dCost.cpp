#include "StixelsEstimatorWith3dCost.hpp"

#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"

#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"

#include "helpers/Log.hpp"

#include <boost/format.hpp>

#include <stdexcept>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StixelsEstimatorWith3dCost");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StixelsEstimatorWith3dCost");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StixelsEstimatorWith3dCost");
}

} // end of anonymous namespace



namespace doppia {


typedef boost::multi_array_types::index_range range_t;

typedef DisparityCostVolume::const_data_2d_view_t const_data_2d_view_t;
typedef DisparityCostVolume::const_data_1d_view_t const_data_1d_view_t;
typedef DisparityCostVolume::const_data_2d_subarray_t const_data_2d_subarray_t;
typedef DisparityCostVolume::const_data_1d_subarray_t const_data_1d_subarray_t;

typedef StixelsEstimatorWith3dCost::cost_volume_t cost_volume_t;

// 3d --
typedef cost_volume_t::array_view<3>::type cost_volume_view_t;

// 2d --
typedef cost_volume_view_t::reference cost_volume_2d_subarray_t;
typedef cost_volume_t::array_view<2>::type cost_volume_2d_view_t;

// 1d --
typedef cost_volume_2d_view_t::reference cost_volume_1d_subarray_t;
typedef cost_volume_t::array_view<1>::type cost_volume_1d_view_t;


StixelsEstimatorWith3dCost::StixelsEstimatorWith3dCost(
    const boost::program_options::variables_map &options,
    const MetricStereoCamera &camera,
    const float expected_objects_height,
    const int minimum_object_height_in_pixels,
    const int num_height_levels_,
    const int stixel_width)
    : StixelsEstimator(options, camera, expected_objects_height, minimum_object_height_in_pixels, stixel_width),
      num_height_levels(num_height_levels_)
{
    // nothing to do here
    return;
}

StixelsEstimatorWith3dCost::~StixelsEstimatorWith3dCost()
{
    // nothing to do here
    return;
}

/// Provide the best estimate available for the ground plane
void StixelsEstimatorWith3dCost::set_ground_plane_estimate(const GroundPlane &ground_plane,
                                                           const GroundPlaneEstimator::line_t &v_disparity_ground_line)
{
    StixelsEstimator::set_ground_plane_estimate(ground_plane, v_disparity_ground_line);

    set_mininum_v_given_disparity_and_height_level();
    return;
}


void StixelsEstimatorWith3dCost::set_mininum_v_given_disparity_and_height_level()
{
    const int num_height_levels = this->num_height_levels;


    if(num_height_levels < 2)
    {
        throw std::runtime_error("StixelsEstimatorWith3dCost requires at least two height levels");
    }

    if(expected_object_height <= 0)
    {
        throw std::runtime_error("StixelsEstimatorWith3dCost requires positive value for expected_object_height");
    }

    const int num_disparities = pixels_cost_volume_p->disparities();
    const int max_v = pixels_cost_volume_p->rows() - 1;

    const MetricCamera &left_camera = stereo_camera.get_left_camera();


    const float height_per_level = expected_object_height / (num_height_levels - 1);

    minimum_v.resize(boost::extents[num_height_levels][num_disparities]);

    const bool print_details = false;
    for(int height_level=0; height_level < num_height_levels; height_level+=1)
    {
        const float max_height_for_level = (height_level + 1) * height_per_level;
        minimum_v_t::reference minimum_v_given_disparity = minimum_v[height_level];

        for(int d=0; d < num_disparities; d+=1)
        {
            int &minimum_v = minimum_v_given_disparity[d];
            const int max_minimum_v = std::max(0, v_given_disparity[d] - minimum_object_height_in_pixels);

            if(height_level == (num_height_levels - 1))
            {
                // for the last level we always consider the top of the image
                // as the minimum_v
                minimum_v = 0;
            }
            else if(d == 0)
            {
                minimum_v = max_minimum_v;
            }
            else
            { // height_level != (num_height_levels - 1)

                const float depth = stereo_camera.disparity_to_depth(d);
                const Eigen::Vector2f uv_point =
                        left_camera.project_ground_plane_point(the_ground_plane,
                                                               0, depth, max_height_for_level);
                minimum_v = static_cast<int>(uv_point[1]);

                // check up and lower image bounds
                minimum_v = std::min(std::max(0, minimum_v), max_v);

                // check constraint with respect to v_given_disparity (ground line estimate)
                minimum_v = std::min(minimum_v, max_minimum_v);

                // assert that the minimum_v is high
                assert(v_given_disparity[d] > minimum_v);
            }

            if(print_details)
            {
                log_debug() << boost::format("minimum_v[%i][%i] == %i") %
                               height_level % d %  minimum_v <<
                               std::endl;
            }

        } // end of "for each disparity value"
    } // end of "for each height level"

    return;
} // end of StixelsEstimatorWith3dCost::set_mininum_v_given_disparity_and_height_level


void StixelsEstimatorWith3dCost::compute()
{
    // compute the u-height-disparity cost volume --
    // (using estimated ground plane)
    compute_cost_volume();

    // find the optimal stixels --
    // (using dynamic programming)
    compute_stixels();

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

/// compute the u-height-disparity cost volume
void StixelsEstimatorWith3dCost::compute_cost_volume()
{
    //const int num_rows = pixels_cost_volume_p->rows();
    const size_t num_columns = pixels_cost_volume_p->columns();
    const size_t num_disparities = pixels_cost_volume_p->disparities();
    
    object_cost_volume.resize(boost::extents[num_height_levels][num_disparities][num_columns]);

    // reset and resize the object_cost and ground_cost
    // Eigen::MatrixXf::Zero(rows, cols)
    object_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);
    ground_u_disparity_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);

    // simplest implementation that could ever work
    const bool search_nearby_disparities = true;



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
            const size_t ground_obstacle_v_boundary = v_given_disparity[d];
            // precomputed_v_disparity_line already checked for >=0 and < num_rows

            // from tentative ground upwards, over the object -
            for(int height_level = 0; height_level < num_height_levels; height_level +=1)
            {
                const int minimum_v_for_disparity = minimum_v[height_level][d];
                float &object_cost = object_cost_volume[height_level][d][u];
                for(size_t v=minimum_v_for_disparity; v < ground_obstacle_v_boundary; v+=1)
                {
                    object_cost += rows_slice[v];
                }

                // normalize the object cost -
                object_cost /= (ground_obstacle_v_boundary - minimum_v_for_disparity);
            }

            // from tentative ground downwards, over the ground -
            float &ground_cost = ground_u_disparity_cost(d, u);
            for(size_t v=ground_obstacle_v_boundary; v < rows_slice.size(); v+=1)
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
            if (rows_slice.size() > ground_obstacle_v_boundary)
            {
		ground_cost /= (rows_slice.size() - ground_obstacle_v_boundary);	    
            }

        } // end of "for each disparity"
    } // end of "for each u"


    const bool threshold_ground_cost = true;
    if(threshold_ground_cost)
    {
        // FIXME hardcoded value
        const float ground_cost_threshold_fraction = 1/3.0;
        const float ground_cost_threshold = ground_u_disparity_cost.maxCoeff() * ground_cost_threshold_fraction;
        ground_u_disparity_cost = ground_u_disparity_cost.unaryExpr(CwiseThreshold<float>(ground_cost_threshold));
    }

    // post filtering step --
    {
        const bool high_frequency_band_pass_filter = true; // FIXME just for debugging
        if(high_frequency_band_pass_filter)
        {
            high_pass_vertical_cost_filter(object_cost_volume);
        }


        const bool do_horizontal_smoothing = true;
        if(do_horizontal_smoothing)
        {
            low_pass_horizontal_cost_filter(object_cost_volume);
            //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
            //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
            //low_pass_horizontal_cost_filter(ground_u_disparity_cost);
        }

        const bool use_dirty_hack_normalization = true;
        if(use_dirty_hack_normalization)
        {
            ground_u_disparity_cost *= 1.0; // 3, 1
        }
    }


    // height based weighting --
    {
        std::vector<float> height_cost_weight(num_height_levels);
        for(int height_level = 0; height_level < num_height_levels; height_level +=1)
        {
            // FIXME hardcoded value
            //height_cost_weight[height_level] = 1 - pow(0.8, height_level);
            //height_cost_weight[height_level] = 1 + 10*height_level; // kinda works
            height_cost_weight[height_level] = 1 + 0.5*height_level;
            //height_cost_weight[height_level] = 1- std::pow(0.3, height_level);
        }


        for(int h = 0; h < num_height_levels; h +=1)
        {
            for(size_t d = 0; d < num_disparities; d += 1)

            {
                for(size_t u = 0; u < num_columns; u += 1)
                {
                    float &cost = object_cost_volume[h][d][u];
                    if(cost > 0)
                    {
                        object_cost_volume[h][d][u] /= height_cost_weight[h];
                    }
                    else
                    {
                        object_cost_volume[h][d][u] *= height_cost_weight[h];
                    }
                }
            }
        }

    }


    // set the final cost
    // FIXME what should be done here ?
    //u_disparity_cost = object_u_disparity_cost + ground_u_disparity_cost;

    {
        // resize u_disparity_cost_volume --
        std::vector<size_t> ex;
        const size_t* shape = object_cost_volume.shape();
        ex.assign( shape, shape+object_cost_volume.num_dimensions() );
        u_disparity_cost_volume.resize( ex );

        // copy data --
        //u_disparity_cost_volume = object_cost_volume;//ground_u_disparity_cost;

        for(int h = 0; h < num_height_levels; h +=1)
        {
            for(size_t d = 0; d < num_disparities; d += 1)

            {
                for(size_t u = 0; u < num_columns; u += 1)
                {
                    u_disparity_cost_volume[h][d][u] = object_cost_volume[h][d][u] + ground_u_disparity_cost(d, u);
                }
            }
        }

    }



    const bool plot_inside_u_disparity_cost = true;

    if(plot_inside_u_disparity_cost)
    {

    }

    // mini trick to fix the "left area initialization issue"
    // as it is the discarted left area of the image creates an homogeneous 0 cost area
    // in this area the algorithms just "goes down" while it should "stay up at disparity 0"
    // then we modify the diagonal line between the zero area and the costs area
    // so that the zero area is not the best choice anymore
    /*{
        const float max_u_disparity_cost = u_disparity_cost.maxCoeff();

        // for every pixel from left to right where cost == 0
        // set cost = -0.1
        const float small_negative_value = -1E5;
        size_t u = 0;
        for(; u < num_columns; u+=1)
        {
            float &t_cost = u_disparity_cost(0, u);
            if(t_cost == 0)
            {
                t_cost = small_negative_value;
            }
            else
            {
                // as soon as we find the first non-zero point, we stop
                break;
            }
        }

        const size_t start_u = std::max<size_t>(u - 1, 0);
        for(size_t d = 0; d < num_disparities; d+=1)
        {
            const size_t max_u = std::min(num_columns, start_u + d + 1);
            for(u=start_u; u < max_u; u+=1)
            {
                u_disparity_cost(d,u) = max_u_disparity_cost;
            }
        }
    }*/

    return;
}


void copy_height_level_slice(Eigen::MatrixXf &u_d_slice_cost, StixelsEstimatorWith3dCost::cost_volume_t &cost_volume, const int height_level)
{
    const int num_disparities = cost_volume.shape()[1];
    const int num_columns = cost_volume.shape()[2];

    for(int d=0; d < num_disparities; d +=1)
    {
        for(int c=0; c < num_columns; c +=1)
        {
            u_d_slice_cost(d,c) = cost_volume[height_level][d][c];
        }
    }
    return;
}

void copy_height_level_slice( StixelsEstimatorWith3dCost::cost_volume_t &cost_volume, const int height_level, Eigen::MatrixXf &u_d_slice_cost)
{
    const int num_disparities = cost_volume.shape()[1];
    const int num_columns = cost_volume.shape()[2];

    for(int d=0; d < num_disparities; d +=1)
    {
        for(int c=0; c < num_columns; c +=1)
        {
            cost_volume[height_level][d][c] = u_d_slice_cost(d,c);
        }
    }
    return;
}



void StixelsEstimatorWith3dCost::high_pass_vertical_cost_filter(cost_volume_t &cost_volume)
{

    const int num_height_levels = cost_volume.shape()[0];
    const int num_disparities = cost_volume.shape()[1];
    const int num_columns = cost_volume.shape()[2];


    Eigen::MatrixXf u_d_slice_cost(num_disparities, num_columns);

    for(int l=0; l < num_height_levels; l += 1)
    {
        copy_height_level_slice(u_d_slice_cost, cost_volume, l);
        StixelsEstimator::high_pass_vertical_cost_filter(u_d_slice_cost);
        copy_height_level_slice(cost_volume, l, u_d_slice_cost);

    } // end of "for each height level"

    return;
}

void StixelsEstimatorWith3dCost::low_pass_horizontal_cost_filter(cost_volume_t &cost_volume)
{

    const int num_columns = cost_volume.shape()[2];
    const int num_disparities = cost_volume.shape()[1];
    const int num_height_levels = cost_volume.shape()[0];

    Eigen::MatrixXf u_d_slice_cost(num_disparities, num_columns);

    for(int l=0; l < num_height_levels; l += 1)
    {
        copy_height_level_slice(u_d_slice_cost, cost_volume, l);
        StixelsEstimator::low_pass_horizontal_cost_filter(u_d_slice_cost);
        copy_height_level_slice(cost_volume, l, u_d_slice_cost);

    } // end of "for each height level"

    return;
}


/// estimate the stixels using dynamic programming over the cost volume
void StixelsEstimatorWith3dCost::compute_stixels()
{

    assert(u_disparity_cost_volume.num_dimensions() == 3);
    const int num_columns = u_disparity_cost_volume.shape()[2];
    const int num_disparities = u_disparity_cost_volume.shape()[1];
    const int num_height_levels = u_disparity_cost_volume.shape()[0];


    if ( static_cast<int>(pixels_cost_volume_p->columns()) != num_columns or
            static_cast<int>(pixels_cost_volume_p->disparities()) != num_disparities)
    {
        throw std::runtime_error("StixelsEstimatorWith3dCost::compute_stixels object_cost_volume does not match the expected dimensions");
    }

    // FIXME hardcoded value
    const float diagonal_weight = 20;


    // right to left pass --    
    {

        {
            // resize u_disparity_cost_volume --
            std::vector<size_t> ex;
            const size_t* shape = u_disparity_cost_volume.shape();
            ex.assign( shape, shape+u_disparity_cost_volume.num_dimensions() );
            M_cost_volume.resize( ex );

            // we first copy all m_i(d_i) values
            M_cost_volume = u_disparity_cost_volume;
        }

        for(int column = num_columns - 2; column >=0; column -= 1)
        {
            // equation 3 with d_{i-1} replaced by e
            // we do min instead of max because we are using correlation cost
            // M_i(d_i) = m_i(d_i) + min_e[ M_{i-1}(e) - c_i(d_i, e) ]

            for(int d=0; d < num_disparities; d+=1)	      
                for(int h=0; h < num_height_levels; h+=1)
                {

                    float min_M_minus_c = std::numeric_limits<float>::max();

                    for(int e=0; e < num_disparities; e+=1)
                        for (int eh=0; eh < num_height_levels; eh+=1)
                        {
                            // implementing the defition of c_i(d_i,e) at equation 5
                            // c is c_i(d, e);
                            float c = 0;
                            const int e_minus_one = e-1;
                            if(d > e_minus_one)
                            {
                                c = 0;
                            }
                            else if (d == e_minus_one)
                            {
                                const float t_cost = object_cost_volume[eh][d][column];
                                c = -diagonal_weight -t_cost;
                            }
                            else
                            { // d < e_minus_one
                                // this is not a candidate for min[...]
                                // c = -infinity
                                // (numeric_limits::max should not be used, to avoid float overflow)
                                //c = -10E4;
                                continue;
                            }

                            const float t_cost = M_cost_volume[eh][e][column + 1] - c;

                            // we do min instead of max because we are using correlation cost
                            min_M_minus_c = std::min(t_cost, min_M_minus_c);
                        } // end of "for each disparity e"

                    //printf("right to left pass M_cost(%i, %i) += %.3f\n",
                    //       d, column, min_M_minus_c);

                    M_cost_volume[h][d][column] += min_M_minus_c;
                } // end of "for each disparity d"
        } // end of "for each column", i.e. "for each u value"
    } // end of right to left pass


    // left to right pass --
    {
        u_disparity_ground_obstacle_boundary.resize(num_columns);
	u_height_ground_obstacle_boundary.resize(num_columns);

        // we set the first value directly
        {
            int &d_star = u_disparity_ground_obstacle_boundary[0];
            int &h_star = u_height_ground_obstacle_boundary[0];
	    
	    d_star = 0;
	    h_star = 0;
            float t_cost = M_cost_volume[h_star][d_star][0];
            
            for (int d = 0; d < num_disparities; d+=1)
            {
                for (int h = num_height_levels-1; h >= 0; h-=1)
                {
                    if (M_cost_volume[h][d][0] < t_cost ) //favorized small disparities and big heights when equal costs
                    {
                        t_cost = M_cost_volume[h][d][0];
                        d_star = d;
                        h_star = h;
                    }
                } // end of "for each height level"
            } // end of "for each disparity"
        }

        // the rest are recursively
        for(int column = 1; column < num_columns; column += 1)
        {
            const int previous_d_star = u_disparity_ground_obstacle_boundary[column - 1];
	    const int previous_h_star = u_height_ground_obstacle_boundary[column - 1];

            int &d_star = u_disparity_ground_obstacle_boundary[column];
	    int &h_star = u_height_ground_obstacle_boundary[column];
	    
            d_star = 0;
	    h_star = 0;
	    
            float min_M_minus_c = std::numeric_limits<float>::max();
            
	    for(int e=0; e < num_disparities; e+=1)
            {
                for(int eh=0; eh < num_height_levels; eh+=1)
                {
                    // implementing the defition of c_i(d_i,e) at equation 5
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
                        const float t_cost = object_cost_volume[previous_h_star][previous_d_star][column-1];
                        c = -diagonal_weight - t_cost;
                    }
                    else
                    { // previous_d_star < e_minus_one
                        // this is not a candidate for min[...]
                        // c = -infinity,
                        // (numeric_limits::max should not be used, to avoid float overflow)
                        //c = -10E4;
                        continue;
                    }

                    const float t_cost = M_cost_volume[eh][e][column] - c;

                    // we do min instead of max because we are using correlation cost
                    if(t_cost <= min_M_minus_c)
                    {
                        d_star = e;
                        h_star = eh;
                        min_M_minus_c = t_cost;
                    }
                } // end of "for each height level eh"
            }  // end of "for each disparity e"

        } // end of "for each column", i.e. "for each u value"

    } // end of left to right pass

    // at this point u_disparity_ground_obstacle_boundary is now set


    // set resulting stixels --
    {
        // dummy version, one stixel per column
        the_stixels.resize(num_columns);

        for(int u = 0; u < num_columns; u += 1)
        {
            const int disparity = u_disparity_ground_obstacle_boundary[u];
	    const int height_level = u_height_ground_obstacle_boundary[u];
	    
            // map from disparity to v based on the ground estimate
            const int &bottom_v = v_given_disparity[disparity];
	    const int &top_v = minimum_v[height_level][disparity];

            Stixel &t_stixel = the_stixels[u];
            t_stixel.width = 1;
            t_stixel.x = u;
	    
            t_stixel.bottom_y = bottom_v; 
            t_stixel.top_y = top_v;
            t_stixel.disparity = disparity;
            t_stixel.type = Stixel::Unknown;
        }
    }

    return;
}

} // end of namespace doppia
