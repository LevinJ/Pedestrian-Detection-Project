#ifndef BOOTSTRAPPING_LIB_HPP
#define BOOTSTRAPPING_LIB_HPP

#include "IntegralChannelsComputer.hpp"
#include "boosted_learning/ImageData.hpp"

#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/function.hpp>
#include <vector>

namespace bootstrapping
{

typedef boosted_learning::ImageData meta_datum_t;

typedef boost::function<void (const meta_datum_t &, const integral_channels_t &)> append_result_functor_t;


class PushBackFunctor
{
public:
    PushBackFunctor(std::vector<meta_datum_t> &meta_data,
                    std::vector<integral_channels_t> &integral_images);
    ~PushBackFunctor();

    void operator()(const meta_datum_t &meta_datum, const integral_channels_t &integral_image);

protected:

    std::vector<meta_datum_t> &meta_data;
    std::vector<integral_channels_t> &integral_images;
};


/// Given the path to a classifier and set of negative images,
/// and a maximum number of false positive to find, will search for false positives
/// (assuming pedestrians from the INRIA dataset) and fill in the meta_data and integral_images structures.
/// if max_false_positives_per_image is negative, this value is ignored
void bootstrap(const boost::filesystem::path &classifier_model_file,
               const std::vector<std::string> &negative_image_paths_to_explore,
               const size_t max_false_positives,
               const size_t max_false_positives_per_image,
               const float min_scale, const float max_scale, const int num_scales,
               const float min_ratio, const float max_ratio, const int num_ratios,
               const bool use_less_memory,
               std::vector<meta_datum_t> &meta_data,
               std::vector<integral_channels_t> &integral_images,
const boost::program_options::variables_map &options);

/// Depending on the provided function, this version of bootstrap can be much more memory efficient
void bootstrap(const boost::filesystem::path &classifier_model_file,
               const std::vector<std::string> &negative_image_paths_to_explore,
               const size_t max_false_positives,
               const size_t max_false_positives_per_image,
               const float min_scale, const float max_scale, const int num_scales,
               const float min_ratio, const float max_ratio, const int num_ratios,
               const bool use_less_memory,
               append_result_functor_t &functor,
const boost::program_options::variables_map &options);

} // end of namespace bootstrapping

#endif // BOOTSTRAPPING_LIB_HPP
