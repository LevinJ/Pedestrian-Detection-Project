#ifndef DOPPIA_VERYFASTDETECTORSCALESTATISTICSAPPLICATION_HPP
#define DOPPIA_VERYFASTDETECTORSCALESTATISTICSAPPLICATION_HPP

#include "applications/BaseApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include <boost/scoped_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include <vector>


namespace doppia {

class ImagesFromDirectory;
class VeryFastIntegralChannelsDetector;

using namespace boost::accumulators;

struct AccumulatorsPerDeltaScale
{
    typedef accumulator_set<double, stats<tag::count, tag::mean, tag::median, tag::min, tag::max > >  accumulator_one_t;
    typedef accumulator_set<double, stats<tag::p_square_quantile > >  quantile_t;

    AccumulatorsPerDeltaScale();
    accumulator_one_t accumulator_one;
    quantile_t very_low_quantile, low_quantile, high_quantile, very_high_quantile;

    void add_sample(const float sample_value);

};

class VeryFastDetectorScaleStatisticsApplication : public BaseApplication
{
protected:
    // processing modules
    scoped_ptr<ImagesFromDirectory> directory_input_p;
    boost::scoped_ptr<VeryFastIntegralChannelsDetector> objects_detector_p;


public:

    VeryFastDetectorScaleStatisticsApplication();
    ~VeryFastDetectorScaleStatisticsApplication();

    /// human readable title used on the console and graphical user interface
    std::string get_application_title();

    /// helper method used by the user interfaces when recording screenshots
    /// this number is expected to change with that same frequency that update_gui is called
    int get_current_frame_number() const;

protected:

    std::vector<AccumulatorsPerDeltaScale>
    statistics_per_delta_scale,
    centered_statistics_per_delta_scale;

    void get_all_options_descriptions(program_options::options_description &desc);

    /// child classes will instanciate here the different processing modules
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    /// core logic of the application
    /// this method should be redefined by all child classes
    /// @note this method needs to repeatedly call update_gui();
    void main_loop();

    void process_frame(const AbstractVideoInput::input_image_view_t &input_view);

    void update_statistics(const boost::multi_array<float, 2> &scores_per_pixel,
                           std::vector<AccumulatorsPerDeltaScale> &statistics_per_delta_scale) const;

    void update_centered_statistics(const boost::multi_array<float, 2> &scores_per_pixel,
                           std::vector<AccumulatorsPerDeltaScale> &centered_statistics_per_delta_scale) const;


    void save_computed_statistics() const;
    void save_computed_centered_statistics() const;

};

} // end of namespace doppia

#endif // DOPPIA_VERYFASTDETECTORSCALESTATISTICSAPPLICATION_HPP
