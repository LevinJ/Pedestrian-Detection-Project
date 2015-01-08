#include "AbstractDisparityCostVolumeEstimator.hpp"

#include "DisparityCostVolume.hpp"

#include "helpers/get_option_value.hpp"

namespace doppia {

using namespace std;
using namespace boost;

program_options::options_description
AbstractDisparityCostVolumeEstimator::get_args_options()
{
    program_options::options_description
            desc("AbstractDisparityCostVolumeEstimator options");

    desc.add_options()

            ("pixels_matching",
             program_options::value<string>()->default_value("sad"),
             "pixels matching method: sad, ssd, census or gradient")


            ("threshold",
             program_options::value<float>()->default_value(0.5),
             "minimum percent of pixels required to declare a match value between [0,1]")


            ("census_window_size",
             program_options::value<int>()->default_value(5),
             "window size used during the census transform")

            ;

    return desc;
}

AbstractDisparityCostVolumeEstimator::AbstractDisparityCostVolumeEstimator()
{
    // this constructor should only be used for unit testing
    return;
}

AbstractDisparityCostVolumeEstimator::AbstractDisparityCostVolumeEstimator(const program_options::variables_map &options)
{

    pixels_matching_method = get_option_value<std::string>(options, "pixels_matching");

    max_disparity = get_option_value<int>(options, "max_disparity");

    maximum_cost_per_pixel = -1; // not yet initialized
    first_computation = true;


/*
    {
        census_window_width = get_option_value<int>(options, "census_window_size");
        census_window_height = census_window_width;

        // window size has to be odd
        assert(census_window_width % 2 == 1);
        assert(census_window_height % 2 == 1);
    }

    {
        threshold_percent = get_option_value<float>(options,"threshold");

        assert(threshold_percent > 0.0f);
        assert(threshold_percent <= 1.0f);
    }
*/

    return;
}

AbstractDisparityCostVolumeEstimator::~AbstractDisparityCostVolumeEstimator()
{
    // nothing to do here
    return;
}



void  AbstractDisparityCostVolumeEstimator::resize_cost_volume(const point_t &input_dimensions,
                                                       DisparityCostVolume &cost_volume) const
{

    DisparityCostVolume::data_3d_view_t data = cost_volume.get_costs_views();

    // lazy initialization
    if(data.empty()
            or data.dimensionality != 3
            or cost_volume.rows() != static_cast<size_t>(input_dimensions.y)
            or cost_volume.columns() != static_cast<size_t>(input_dimensions.x)
            or cost_volume.disparities() != max_disparity)
    {
        cost_volume.resize(input_dimensions.y, input_dimensions.x, max_disparity);
    }

    return;
}

float AbstractDisparityCostVolumeEstimator::get_maximum_cost_per_pixel() const
{
    if(maximum_cost_per_pixel < 0)
    {
        throw std::runtime_error("Called AbstractDisparityCostVolumeEstimator::get_maximum_cost_per_pixel before DisparityCostVolume::compute maximum cost is undefined");
    }

    return maximum_cost_per_pixel;
}



} // end of namespace doppia
