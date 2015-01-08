#ifndef IRLSLINESDETECTOR_HPP
#define IRLSLINESDETECTOR_HPP

#include "AbstractLinesDetector.hpp"
#include <boost/program_options.hpp>
#include <list>

#include <Eigen/Core>

namespace doppia {

/// Lines detection using Iteratively reweighted least squares
/// this class assume that the input image is binary, and that
/// there is one single dominant line
/// http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
/// Tukey loss
/// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class IrlsLinesDetector : public AbstractLinesDetector
{
public:

    static boost::program_options::options_description get_args_options();

    IrlsLinesDetector(const boost::program_options::variables_map &options);

    IrlsLinesDetector(const int intensity_threshold,
                      const int num_iterations,
                      const float max_tukey_c,
                      const float min_tukey_c);
    ~IrlsLinesDetector();

    /// Find the main line on an image,
    /// assuming that points of interest can be thresholded
    void operator()(const source_view_t &src, lines_t &lines);

    typedef std::pair<int, int>  point_t;
    typedef std::vector< point_t > points_t;

    /// Iterative reweighted least squares estimation of the line
    /// given a set of points (x,y) and initial weights for each point
    /// @param lines contains the single estimated line
    void operator()(const points_t &points,
                    const Eigen::VectorXf &prior_points_weights,
                    lines_t &lines);

    /// Provide the best estimate available estimate for the line
    /// this method should be called right before each call to operator()
    void set_initial_estimate(const line_t &line_estimate);

    float compute_l1_residual();

protected:

    const int intensity_threshold, num_iterations;
    const float max_tukey_c, min_tukey_c;

    bool has_previous_line_estimate;
    line_t previous_line_estimate;

    Eigen::VectorXf b, x, w;
    Eigen::MatrixXf A;
};

} // namespace doppia

#endif // IRLSLINESDETECTOR_HPP
