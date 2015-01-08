#ifndef DETECTORSCOMPARISONTESTAPPLICATION_HPP
#define DETECTORSCOMPARISONTESTAPPLICATION_HPP

#include "applications/BaseApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include <boost/scoped_ptr.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <boost/multi_array.hpp>

#include <vector>

namespace doppia {

class FastestPedestrianDetectorInTheWest;
class IntegralChannelsDetector;
class IntegralChannelsFeature;
class ImagesFromDirectory;
class SoftCascadeOverIntegralChannelsModel;


using namespace boost::accumulators;

class DetectorsComparisonTestApplication : public BaseApplication
{

protected:
    // processing modules
    scoped_ptr<ImagesFromDirectory> directory_input_p;
    boost::scoped_ptr<FastestPedestrianDetectorInTheWest> fpdw_detector_p;
    boost::scoped_ptr<IntegralChannelsDetector> chnftrs_detector_p;

public:

    typedef accumulator_set<double,
    stats<tag::mean, tag::variance > > second_moment_accumulator_t;

    typedef accumulator_set<double,
    stats<tag::mean, tag::variance, tag::min, tag::max > >  channel_statistics_accumulator_t;

    typedef boost::multi_array<channel_statistics_accumulator_t, 2> per_scale_per_channel_statistics_t;

    typedef boost::multi_array<float, 2> feature_values_t;

public:

    DetectorsComparisonTestApplication();
    ~DetectorsComparisonTestApplication();

    /// human readable title used on the console and graphical user interface
    std::string get_application_title();

    /// helper method used by the user interfaces when recording screenshots
    /// this number is expected to change with that same frequency that update_gui is called
    int get_current_frame_number() const;

protected:

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

    void check_computed_statistics();

    void save_channel_statistics();

    std::vector<channel_statistics_accumulator_t> global_channel_statistics;
    per_scale_per_channel_statistics_t per_scale_per_channel_statistics;

    void update_channels_statistics(
        const size_t current_scale_index,
        const IntegralChannelsDetector &chnftrs_detector,
        const FastestPedestrianDetectorInTheWest &fpdw_detector,
        per_scale_per_channel_statistics_t &per_scale_per_channel_statistics,
        vector<channel_statistics_accumulator_t> &global_channel_statistics) const;

    void compute_feature_values(const IntegralChannelsDetector &detector,
                                const IntegralChannelsFeature &feature,
                                feature_values_t &feature_values) const;
};


} // end of namespace doppia

#endif // DETECTORSCOMPARISONTESTAPPLICATION_HPP
