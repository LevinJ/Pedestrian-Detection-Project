

/// To be used as
/// cmake ./ && make -j2 && ./test_stixels_estimation; mogrify  -scale 800% *.png && eog u_d_cost_original.png

#define BOOST_TEST_MODULE TestStixelsEstimator
#include <boost/test/unit_test.hpp>

#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "drawing/gil/draw_matrix.hpp"
#include "drawing/gil/colors.hpp"

#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/random.hpp>

#include <Eigen/Core>

#include <string>
#include <cstdio>
#include <iostream>

using namespace doppia;
using namespace std;


boost::mt19937 random_generator;
boost::uniform_real<> uniform_distribution;

boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
noise_generator(random_generator, uniform_distribution);


/// these are helper classes that allows to do the desired tests
class StixelsEstimatorTester;
class EmptyDisparityCostVolume;

class EmptyDisparityCostVolume: public DisparityCostVolume
{
public:
    EmptyDisparityCostVolume(const int max_disparity);
    ~EmptyDisparityCostVolume();
};


EmptyDisparityCostVolume::EmptyDisparityCostVolume(const int max_disparity_)
    : DisparityCostVolume()
{
    //this->max_disparity = max_disparity_;
    return;
}


EmptyDisparityCostVolume::~EmptyDisparityCostVolume()
{
    // nothing to do here
    return;
}

class StixelsEstimatorTester: public StixelsEstimator
{

public:
    StixelsEstimatorTester();
    ~StixelsEstimatorTester();

    void set_costs(const u_disparity_cost_t &ground_cost, const u_disparity_cost_t &object_cost);

protected:
    void compute_disparity_space_cost();
    void compute_ground_obstacle_boundary();

};

StixelsEstimatorTester::StixelsEstimatorTester()
    : StixelsEstimator()
{
    // nothing to do here
    return;
}

StixelsEstimatorTester::~StixelsEstimatorTester()
{
    // nothing to do here
    return;
}

void StixelsEstimatorTester::set_costs(const u_disparity_cost_t &ground_cost, const u_disparity_cost_t &object_cost)
{
    assert(ground_cost.cols() == object_cost.cols());
    assert(ground_cost.rows() == object_cost.rows());
    this->ground_u_disparity_cost = ground_cost;
    this->object_u_disparity_cost = object_cost;

    high_pass_vertical_cost_filter(object_u_disparity_cost);
    low_pass_horizontal_cost_filter(object_u_disparity_cost);
    low_pass_horizontal_cost_filter(ground_u_disparity_cost);

    this->u_disparity_cost = ground_u_disparity_cost + object_u_disparity_cost;


    const int num_disparities = ground_cost.rows();
    this->pixels_cost_volume_p.reset(new EmptyDisparityCostVolume(num_disparities));
    this->pixels_cost_volume_p->resize(ground_cost.rows(), ground_cost.cols(),  num_disparities);

    const int num_rows = pixels_cost_volume_p->rows();
    v_given_disparity.resize(num_disparities);
    disparity_given_v.resize(num_rows);

    //const int minimum_v = 0;
    //minimum_v_given_disparity.resize(num_disparities, minimum_v);

    return;
}

void StixelsEstimatorTester::compute_disparity_space_cost()
{
    // in the tester class this function does nothing
    return;
}

void StixelsEstimatorTester::compute_ground_obstacle_boundary()
{
    StixelsEstimator::compute_ground_obstacle_boundary();
    return;
}

/// create test costs matrices
void create_ideal_noisy_costs(
    const int num_columns,
    const int num_disparities,
    const vector<int> &true_boundary,
    Eigen::MatrixXf &ground_cost, Eigen::MatrixXf &object_cost)
{


    const float cost_value = 50;
    ground_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);
    object_cost = Eigen::MatrixXf::Constant(num_disparities, num_columns, cost_value);

    for(int d=0; d < num_disparities; d+=1)
    {
        for(int u=0; u < num_columns; u +=1)
        {
            object_cost(d, u) += noise_generator()*0.4*cost_value;
        }
    }

    for(int u=0; u < num_columns; u +=1)
    {
        const int boundary_disparity_value = true_boundary[u];

        object_cost(boundary_disparity_value, u) = noise_generator()*0.2*cost_value;

        const int d_minus_one = boundary_disparity_value - 1;
        if(d_minus_one >= 0)
        {
            for(int d=d_minus_one; d >= 0; d -=1)
            {
                ground_cost(d, u) = cost_value;
            }
        }
    }

    return;
} // end of "create_ideal_noisy_costs"




/// create test costs matrices
void create_realistic_noisy_costs(
    const int num_columns,
    const int num_disparities,
    const vector<int> &true_boundary,
    Eigen::MatrixXf &ground_cost, Eigen::MatrixXf &object_cost)
{

    const float max_cost_value = 50;
    ground_cost = Eigen::MatrixXf(num_disparities, num_columns);
    object_cost = Eigen::MatrixXf(num_disparities, num_columns);

    // set object_cost --
    for(int u=0; u < num_columns; u +=1)
    {
        const int boundary_disparity_value = true_boundary[u];

        for(int d=0; d < num_disparities; d+=1)
        {
            const float boundary_scale =
                    std::exp(-5*static_cast<float>(abs(d - boundary_disparity_value)) / num_disparities);

            if(d == boundary_disparity_value)
            {
                object_cost(boundary_disparity_value, u) = noise_generator()*0.5*max_cost_value;
            }
            else
            {
                object_cost(d, u) = boundary_scale*max_cost_value + noise_generator()*0.4*max_cost_value;
            }

        } // end of "for each disparity"
    } // end of "for each column"

    // set ground_cost --
    for(int u=0; u < num_columns; u +=1)
    {
        const int boundary_disparity_value = true_boundary[u];

        const float random_value = noise_generator()*0.8*max_cost_value;

        for(int d=0; d < boundary_disparity_value; d+=1)
        {
                ground_cost(d, u) = random_value;
        } // end of "for each disparity"
    } // end of "for each column"


    boost::uniform_int<> uniform_column_distribution(0, num_columns-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
            random_column_index(random_generator, uniform_column_distribution);

    const int num_random_lines = 150; // 150
    for(int i=0; i < num_random_lines; i+=1)
    {
        // add diagonal line
        {
            const float cost_value = noise_generator()*0.5*max_cost_value;
            int u = random_column_index();
            for(int d=0; d<num_disparities and u < num_columns; d+=1, u+=1)
            {
                object_cost(d, u) += cost_value;
            }
        }

        // add vertical line
        {
            const float cost_value = noise_generator()*0.5*max_cost_value;
            const int u = random_column_index();
            for(int d=0; d<num_disparities and u < num_columns; d+=1)
            {
                object_cost(d, u) += cost_value;
            }
        }

    } // end of "for each random line


    return;
} // end of "create_realistic_noisy_costs"


void create_zero_one_two_costs(
    const int num_columns,
    const int num_disparities,
    const vector<int> &true_boundary,
    Eigen::MatrixXf &ground_cost, Eigen::MatrixXf &object_cost)
{

    ground_cost = Eigen::MatrixXf::Zero(num_disparities, num_columns);
    object_cost = Eigen::MatrixXf::Constant(num_disparities, num_columns, 1);

    for(int d=0; d < num_disparities; d+=1)
    {
        for(int u=0; u < num_columns; u +=1)
        {
            // the upper area has no more cost than the lower area
            //object_cost(d, u) += 0;
        }
    }

    for(int u=0; u < num_columns; u +=1)
    {
        const int boundary_disparity_value = true_boundary[u];
        object_cost(boundary_disparity_value, u) = 0;

        const int d_minus_one = boundary_disparity_value - 1;
        if(d_minus_one >= 0)
        {
            for(int d=d_minus_one; d >= 0; d -=1)
            {
                ground_cost(d, u) = 1;
            }
        }
    }

    return;
} // end of "create_zero_one_two_costs"


void save_images(
    const int num_columns,
    const int num_disparities,
    const vector<int> &true_boundary,
    const vector<int> &expected_boundary,
    const Eigen::MatrixXf &ground_cost, const Eigen::MatrixXf &object_cost,
    const StixelsEstimator &stixels_estimator)
{


    using namespace boost::gil;

    const vector<int> &estimated_boundary =
            stixels_estimator.get_u_disparity_ground_obstacle_boundary();

    rgb8_image_t
            u_d_cost_image(num_columns, num_disparities),
            M_cost_image(num_columns, num_disparities);

    const string
            u_d_cost_original_filename = "u_d_cost_original.png",
            u_d_cost_filtered_filename = "u_d_cost_filtered.png",
            m_cost_filename = "m_cost.png",
            m_cost_normalized_filename = "m_cost_normalized.png",
            m_cost_with_boundary_filename = "m_cost_with_boundary.png";

    rgb8_view_t t_view = view(u_d_cost_image);

    draw_matrix(ground_cost, t_view);
    png_write_view("ground_cost.png", t_view);

    draw_matrix(object_cost, t_view);
    png_write_view("object_cost.png", t_view);


    Eigen::MatrixXf u_d_cost_original = ground_cost + object_cost;
    draw_matrix(u_d_cost_original, t_view);
    png_write_view(u_d_cost_original_filename, t_view);

    draw_matrix(stixels_estimator.get_u_disparity_cost(), t_view);
    png_write_view(u_d_cost_filtered_filename, t_view);

    t_view = view(M_cost_image);
    draw_matrix(stixels_estimator.get_M_cost(), t_view);
    png_write_view(m_cost_filename, t_view);

    // provide a more constrasted visualization
    {
        Eigen::MatrixXf M_cost_normalized = stixels_estimator.get_M_cost();

        // column wise normalization --
        for(int c=0; c < M_cost_normalized.cols(); c+=1)
        {
            const float col_max_value = M_cost_normalized.col(c).maxCoeff();
            const float col_min_value = M_cost_normalized.col(c).minCoeff();
            M_cost_normalized.col(c).array() -= col_min_value;

            if(col_max_value > col_min_value)
            {
                const float scaling =  1.0f / (col_max_value - col_min_value);
                M_cost_normalized.col(c) *= scaling;
            }
        }

        // log scaling --
        M_cost_normalized = (M_cost_normalized.array() + 1).array().log();

        draw_matrix(M_cost_normalized, t_view);
        png_write_view(m_cost_normalized_filename, t_view);
    }

    for(size_t u=0; u < estimated_boundary.size(); u += 1)
    {
        int d = true_boundary[u];
        t_view(u, d) = rgb8_colors::dark_green;

        d = expected_boundary[u];
        t_view(u, d) = rgb8_colors::green;

        d = estimated_boundary[u];
        t_view(u, d) = rgb8_colors::yellow;
    }

    png_write_view(m_cost_with_boundary_filename, t_view);



    printf("Created images %s, %s, %s, %s and %s\n",
           u_d_cost_original_filename.c_str(),
           u_d_cost_filtered_filename.c_str(),
           m_cost_filename.c_str(),
           m_cost_normalized_filename.c_str(),
           m_cost_with_boundary_filename.c_str());

    return;
}

void find_ground_object_boundary_test(
    const int num_columns,
    const int num_disparities,
    const vector<int> &true_boundary,
    const Eigen::MatrixXf &ground_cost, const Eigen::MatrixXf &object_cost,
    const bool create_output_images,
    const bool print_matrices)
{

    assert(ground_cost.cols() == num_columns);
    assert(ground_cost.rows() == num_disparities);

    assert(object_cost.cols() == num_columns);
    assert(object_cost.rows() == num_disparities);

    // set the expected boundary --
    // enforce the 45 [degrees] constraint on the boundary
    vector<int> expected_boundary = true_boundary;
    for(int u=num_columns -2; u >= 0; u-=1)
    {
        const int &previous_boundary = expected_boundary[u + 1];
        int &current_boundary = expected_boundary[u];
        if(current_boundary < previous_boundary)
        {
            current_boundary = previous_boundary - 1;
        }
    }

    // create and execute the estimator --
    StixelsEstimatorTester stixels_estimator;
    stixels_estimator.set_costs(ground_cost, object_cost);
    stixels_estimator.compute();
    const vector<int> &estimated_boundary =
            stixels_estimator.get_u_disparity_ground_obstacle_boundary();


    // save images --
    if(create_output_images)
    {
        save_images(num_columns, num_disparities,
                    true_boundary,
                    expected_boundary,
                    ground_cost, object_cost,
                    stixels_estimator);
    }

    // print matrices --
    if(print_matrices)
    {
        std::cout << std::endl;
        std::cout << "ground_cost" << std::endl <<
                     ground_cost << std::endl << std::endl;
        std::cout << "object_cost" << std::endl <<
                     object_cost << std::endl << std::endl;

        std::cout << "u_disparity_cost" << std::endl <<
                     stixels_estimator.get_u_disparity_cost() <<
                     std::endl << std::endl;

        std::cout << "M_cost" << std::endl <<
                     stixels_estimator.get_M_cost() <<
                     std::endl << std::endl;
    }

    // check the result --
    BOOST_REQUIRE(estimated_boundary.size() == true_boundary.size());

    const int tolerated_difference = 1; // [disparity pixels]
    bool estimation_is_correct = true;
    for(size_t c=0; c < true_boundary.size(); c+=1)
    {
        const bool is_identical = estimated_boundary[c] == expected_boundary[c];

        const bool is_correct =
                (estimated_boundary[c] - expected_boundary[c]) <= tolerated_difference;

        if(not is_identical)
        {
            printf("u == %zi estimated_boundary[c] == %i =?= %i\n",
                   c, estimated_boundary[c], expected_boundary[c]);
        }

        estimation_is_correct &= is_correct;
    }

    BOOST_REQUIRE_MESSAGE(estimation_is_correct, "estimated boundary should be identical to expected_boundary boundary");
    return;
}

BOOST_AUTO_TEST_CASE(FindGroundObjectBoundaryIdealTestCase)
{
    // build a fake M_cost
    // estimate boundary
    // check if estimated boundary matches the expectations

    const int num_columns = 200;
    const int num_disparities = 50;

    // set the true boundary --
    const int boundary_value1 = 12, boundary_value2 = 26;
    assert(boundary_value1 < num_disparities);
    assert(boundary_value2 < num_disparities);

    vector<int> true_boundary(num_columns, boundary_value1);

    for(int u=num_columns/3; u < num_columns*2/3; u+=1)
    {
        true_boundary[u] = boundary_value2;
    }

    const bool create_output_images = true, print_matrices = false;

    Eigen::MatrixXf ground_cost, object_cost;
    create_ideal_noisy_costs(num_columns, num_disparities, true_boundary,
                             ground_cost, object_cost);

    find_ground_object_boundary_test(
                num_columns, num_disparities, true_boundary,
                ground_cost, object_cost,
                create_output_images, print_matrices);
    return;
} // end of "BOOST_AUTO_TEST_CASE"



BOOST_AUTO_TEST_CASE(FindGroundObjectBoundarySmallTestCase)
{
    // build a fake M_cost
    // estimate boundary
    // check if estimated boundary matches the expectations

    const int num_columns = 8;
    const int num_disparities = 4;

    // set the true boundary --
    const int boundary_value1 = 1, boundary_value2 = 3;
    assert(boundary_value1 < num_disparities);
    assert(boundary_value2 < num_disparities);

    vector<int> true_boundary(num_columns, boundary_value1);

    for(int u=num_columns*2/4; u < num_columns*3/4; u+=1)
    {
        true_boundary[u] = boundary_value2;
    }

    const bool create_output_images = false, print_matrices = true;

    Eigen::MatrixXf ground_cost, object_cost;
    create_zero_one_two_costs(num_columns, num_disparities, true_boundary,
                              ground_cost, object_cost);

    find_ground_object_boundary_test(
                num_columns, num_disparities, true_boundary,
                ground_cost, object_cost,
                create_output_images, print_matrices);
    return;
} // end of "BOOST_AUTO_TEST_CASE"



BOOST_AUTO_TEST_CASE(FindGroundObjectBoundaryRealisticTestCase)
{
    // build a fake M_cost
    // estimate boundary
    // check if estimated boundary matches the expectations

    const int num_columns = 200;
    const int num_disparities = 50;

    // set the true boundary --
    const int boundary_value1 = 12, boundary_value2 = 26;
    assert(boundary_value1 < num_disparities);
    assert(boundary_value2 < num_disparities);

    vector<int> true_boundary(num_columns, boundary_value1);

    for(int u=num_columns/3; u < num_columns*2/3; u+=1)
    {
        true_boundary[u] = boundary_value2;
    }

    const bool create_output_images = true, print_matrices = false;

    Eigen::MatrixXf ground_cost, object_cost;
    create_realistic_noisy_costs(num_columns, num_disparities, true_boundary,
                                 ground_cost, object_cost);



    find_ground_object_boundary_test(
                num_columns, num_disparities, true_boundary,
                ground_cost, object_cost,
                create_output_images, print_matrices);
    return;
} // end of "BOOST_AUTO_TEST_CASE"
