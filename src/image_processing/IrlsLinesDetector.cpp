#include "IrlsLinesDetector.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

#include "helpers/get_option_value.hpp"

#include <boost/gil/image_view.hpp>
#include <boost/foreach.hpp>

#include <utility>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace doppia {

using namespace std;
using namespace boost;

typedef IrlsLinesDetector::source_view_t source_view_t;
typedef IrlsLinesDetector::points_t points_t;
typedef AbstractLinesDetector::lines_t lines_t;
typedef AbstractLinesDetector::line_t line_t;

program_options::options_description IrlsLinesDetector::get_args_options()
{
    program_options::options_description desc("IrlsLinesDetector options");

    desc.add_options()

            ("irls.intensity_threshold",
             program_options::value<int>()->default_value(253),
             "intensity threshold used to extract points from the input image")

            ("irls.num_iterations",
             program_options::value<int>()->default_value(15),
             "number reweighted least squares iterations")

            ("irls.max_tukey_c",
             program_options::value<float>()->default_value(50),
             "initial c value for the tukey weight function. This value is in [pixels]")

            ("irls.min_tukey_c",
             program_options::value<float>()->default_value(1),
             "c value for the tukey weight function using in the last iteration. This value is in [pixels]")

            ;


    return desc;

}


IrlsLinesDetector::IrlsLinesDetector(const program_options::variables_map &options)
    : intensity_threshold(get_option_value<int>(options, "irls.intensity_threshold")),
      num_iterations(get_option_value<int>(options, "irls.num_iterations")),
      max_tukey_c(get_option_value<float>(options, "irls.max_tukey_c")),
      min_tukey_c(get_option_value<float>(options, "irls.min_tukey_c")),
      has_previous_line_estimate(false)
{
    assert(max_tukey_c >= min_tukey_c);
    if(intensity_threshold < 0 or intensity_threshold >= 255 )
    {
        throw std::invalid_argument("irls.intensity_threshold should be in the range [0, 255 -1]");
    }
    return;
}


IrlsLinesDetector::IrlsLinesDetector(const int intensity_threshold_,
                                     const int num_iterations_,
                                     const float max_tukey_c_,
                                     const float min_tukey_c_)
    : intensity_threshold(intensity_threshold_),
      num_iterations(num_iterations_),
      max_tukey_c(max_tukey_c_), min_tukey_c(min_tukey_c_),
      has_previous_line_estimate(false)
{
    assert(max_tukey_c >= min_tukey_c);
    return;
}


IrlsLinesDetector::~IrlsLinesDetector()
{
    // nothing to do here
    return;
}

void retrieve_points(const source_view_t &src, const int max_intensity_value,
                     points_t &points,
                     Eigen::VectorXf &prior_points_weights)
{
    Eigen::VectorXf row_weights = Eigen::VectorXf::Ones(src.height());

    int num_points = 0;
    // threshold the input image to obtain point of interests
    for(int row=0; row < src.height(); row+=1)
    {
        int num_points_in_row = 0;
        source_view_t::x_iterator row_it = src.row_begin(row);
        for(int col=0; row_it != src.row_end(row); col+=1, ++row_it)
        {
            const source_view_t::value_type &pixel = *row_it;

            if(pixel > max_intensity_value)
            {
                const int &x = col, &y = row;
                points.push_back(make_pair(x,y));
                num_points += 1;
                num_points_in_row += 1;
            }
            else
            {  // not a point of interest
                continue;
            }
        } // end of "for each column"

        if(num_points_in_row > 0)
        {
            // rows with less points give more confidence
            row_weights(row) = 1.0/num_points_in_row;
        }
    } // end of "for each row"

    if(num_points > 0)
    {
        prior_points_weights.setOnes(num_points);
    }
    else
    {
        // the line estimation will fail in this case, we catch the error later
    }

    int i=0;
    BOOST_FOREACH(points_t::const_reference point, points)
    {
        const int &point_y = point.second;
        prior_points_weights(i) = row_weights(point_y);
        i+=1;
    }

    return;
}

void set_A_and_b(const points_t &points, const int &num_points, Eigen::MatrixXf &A, Eigen::VectorXf &b)
{
    A.setOnes(num_points, 2);
    b.setZero(num_points);

    int i=0;
    BOOST_FOREACH(points_t::const_reference point, points)
    {
        A(i, 0) = point.first; // x coordinate value
        b(i) = point.second; // y coordinate value
        i+=1;
    } // end of "for each point"

    return;
}

/*
void compute_initial_estimate(const points_t &points, const int &num_points)
{

    return;
}
*/

void x_to_line(const Eigen::VectorXf &x, line_t &the_line)
{
    assert(x.size() == 2);
    the_line.direction()(0) = x(0);
    the_line.origin()(0) = x(1);
    return;
}


void line_to_x(const line_t &the_line, Eigen::VectorXf &x)
{
    assert(x.size() == 2);
    x(0) = the_line.direction()(0);
    x(1) = the_line.origin()(0);
    return;
}


/// as defined in
/// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
inline float tukey_weight(const float x, const float c)
{
    assert(c > 0);
    float weight = 0;

    if(std::abs(x) <= c)
    {
        const float x_div_c = x/c;
        const float delta = 1 - x_div_c*x_div_c;
        weight = delta*delta;
    }
    else
    {
        // weight = 0;
    }

    return weight;
}

// tukey_c is in [pixels]
void recompute_weights(const Eigen::MatrixXf &A,
                       const Eigen::VectorXf &b,
                       const Eigen::VectorXf &x,
                       const bool use_horizontal_distance,
                       const float tukey_c,
                       Eigen::VectorXf &w)
{
    // we assume that w is already initialized
    assert(b.rows() == w.rows());

    if(use_horizontal_distance == false)
    {
       throw std::runtime_error("recompute_weights with vertical distance it not yet implemented");
    }

    Eigen::VectorXf &horizontal_error = w;

    horizontal_error = (b - A*x);

    for(int i=0; i < b.rows(); i+=1)
    {
        w(i) = tukey_weight(horizontal_error(i), tukey_c);
    }

    return;
}

/// Provide the best estimate available estimate for the line
void IrlsLinesDetector::set_initial_estimate(const line_t &line_estimate)
{
    previous_line_estimate = line_estimate;
    has_previous_line_estimate = true;
    return;
}


void IrlsLinesDetector::operator()(const source_view_t &src, lines_t &lines)
{

    points_t points;
    Eigen::VectorXf prior_points_weights;

    retrieve_points(src, intensity_threshold, points, prior_points_weights);
    {
        //const int num_points = prior_points_weights.rows();
        //printf("IrlsLinesDetector retrieved %i points\n", num_points);
    }

    this->operator ()(points, prior_points_weights, lines);
    return;
}

/// implementation based on the following tutorial
/// http://graphics.stanford.edu/~jplewis/lscourse/SLIDES.pdf
void IrlsLinesDetector::operator()(const points_t &points,
                                   const Eigen::VectorXf &prior_points_weights,
                                   lines_t &lines)
{
    lines.clear();


    const int num_points = prior_points_weights.rows();
    assert(points.size() == static_cast<size_t>(num_points));

    if(false and points.size() != static_cast<size_t>(num_points))
    {
        printf("num_points count is wrong %zi != %i\n", points.size(), num_points);
        throw std::runtime_error("num_points count is wrong");
    }

    if(num_points < 2)
    {
        // not enough points to compute a line
        // returning zero lines indicates that something went wrong
        return;
    }

    // b=Ax

    set_A_and_b(points, num_points, A, b);
    x.setZero(2); // resize x
    w.setOnes(num_points); // resize w

    Eigen::MatrixXf new_A = A;
    Eigen::VectorXf new_b = b;

    // find initial line estimate --
    if(has_previous_line_estimate)
    {
        line_to_x(previous_line_estimate, x);
    }
    else
    {
        new_A = prior_points_weights.asDiagonal() * A;
        new_b = prior_points_weights.asDiagonal() * b;

        //const bool solved = new_A.svd().solve(new_b, &x); // Eigen2 form
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(new_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        x = svd.solve(new_b);
        const bool solved = true;
        if(solved == false)
        {
            throw std::runtime_error("IrlsLinesDetector::operator() Failed to initial line estimate");
        }
    }
    has_previous_line_estimate = false; // set false for next operator() call

    // iterative reweighted least squares --

    float tukey_c = max_tukey_c; // [pixels]
    const float tukey_c_step = (max_tukey_c - min_tukey_c) / std::max(1, num_iterations - 1); // [pixels]

    //cout << "Initial x" << endl << x << endl;
    for(int i=0; i < num_iterations; i+=1, tukey_c -= tukey_c_step)
    {
        // measure distance between points and line -
        // ( use horizontal or vertical distance )
        const bool use_horizontal_distance = true;
        recompute_weights(A, b, x, use_horizontal_distance, tukey_c, w);
        w = w.cwiseProduct(prior_points_weights);

        // setup the weighted least square estimate -
        new_A = w.asDiagonal() * A;
        new_b = w.asDiagonal() * b;

        // solve weighted least square estimate -        
        //const bool solved = new_A.svd().solve(new_b, &x); // Eigen2 form
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(new_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        x = svd.solve(new_b);
        const bool solved = true;
        if(solved == false)
        {
            printf("IrlsLinesDetector::operator() Failed to solve reweighted line estimate at iteration %i\n",
                   i);
            throw std::runtime_error("IrlsLinesDetector::operator() Failed to solve reweighted line estimate");
        }

        // cout << "x at iteration " << i << endl << x << endl;
        // cout << "tukey_c at iteration " << i << endl << tukey_c << endl;
    } // end of "for each iteration"


    // IrlsLinesDetector estimates only a single line --
    {
        line_t the_line;
        x_to_line(x, the_line);

        lines.push_back(the_line);
    }

    return;
}

float IrlsLinesDetector::compute_l1_residual()
{
    const float residual = ((A*x - b).array() * w.array()).abs().sum();
    return residual;
}

} // namespace doppia
