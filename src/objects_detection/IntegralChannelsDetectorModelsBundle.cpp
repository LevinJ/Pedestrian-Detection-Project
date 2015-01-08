#include "IntegralChannelsDetectorModelsBundle.hpp"

#include "MultiScalesIntegralChannelsModel.hpp" // for normalized_bundle helper method

#include "detector_model.pb.h"

#include "helpers/Log.hpp"

#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>
#include <boost/format.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

//#include <boost/math/special_functions/round.hpp>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "IntegralChannelsDetectorModelsBundle");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "IntegralChannelsDetectorModelsBundle");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "IntegralChannelsDetectorModelsBundle");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "IntegralChannelsDetectorModelsBundle");
}

std::vector<float> upscaling_factors;
std::vector<float> downscaling_factors;

} // end of anonymous namespace


// fix for hashing tuples,
// from http://stackoverflow.com/questions/3611951/building-an-unordered-map-with-tuples-as-keys
namespace boost {
namespace tuples {
namespace detail {

template <class Tuple, size_t Index = boost::tuples::length<Tuple>::value - 1>
struct HashValueImpl
{
    static void apply(size_t& seed, Tuple const& tuple)
    {
        HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
        boost::hash_combine(seed, tuple.get<Index>());
    }
};

template <class Tuple>
struct HashValueImpl<Tuple,0>
{
    static void apply(size_t& seed, Tuple const& tuple)
    {
        boost::hash_combine(seed, tuple.get<0>());
    }
};

} // end of namespace detail

template <class Tuple>
size_t hash_value(Tuple const& tuple)
{
    size_t seed = 0;
    detail::HashValueImpl<Tuple>::apply(seed, tuple);
    return seed;
}

} // end of namespace tuples
} // end of namespace boost



namespace doppia {


IntegralChannelsDetectorModelsBundle::IntegralChannelsDetectorModelsBundle()
{
    // nothing to do here
    return;
}


/// this constructor will copy the protobuf data into a more efficient data structure
IntegralChannelsDetectorModelsBundle::IntegralChannelsDetectorModelsBundle(const doppia_protobuf::DetectorModelsBundle &model)
{
    if(model.has_bundle_name())
    {
        log_info() << "Parsing models bundle named \"" << model.bundle_name() << "\"" << std::endl;
    }

    if(model.detectors_size() == 0)
    {
        throw std::invalid_argument("IntegralChannelsDetectorModelsBundle received an empty model");
    }

    doppia_protobuf::DetectorModelsBundle rescaled_model;
    normalized_bundle(model, rescaled_model);

    for(int index=0; index < rescaled_model.detectors_size(); index+=1)
    {
        log_info() << "Model index " << index
                   << " has scale " << rescaled_model.detectors(index).scale() << std::endl;
        detectors.push_back(SoftCascadeOverIntegralChannelsModel(rescaled_model.detectors(index)));
    }

    // validate the received data
    sanity_check();
    return;
}


IntegralChannelsDetectorModelsBundle::~IntegralChannelsDetectorModelsBundle()
{
    // nothing to do here
    return;
}


const IntegralChannelsDetectorModelsBundle::detectors_t& IntegralChannelsDetectorModelsBundle::get_detectors() const
{
    return detectors;
}


bool IntegralChannelsDetectorModelsBundle::has_soft_cascade() const
{
    int num_detectors_with_cascade = 0;
    BOOST_FOREACH(const detector_t &detector, detectors)
    {
        if( detector.has_soft_cascade() )
        {
            num_detectors_with_cascade += 1;
        }
    }

    const bool use_the_detector_model_cascade = num_detectors_with_cascade > (detectors.size()/2.0f);

    log_info() << "Will " << ( (use_the_detector_model_cascade)?"use":"ignore")
               << " the cascade thresholds provided in " << num_detectors_with_cascade
               << " out of " << detectors.size()
               << " detectors in the multiscales model" << std::endl;

    return use_the_detector_model_cascade;
}


void IntegralChannelsDetectorModelsBundle::sanity_check() const
{

    bool found_scale_one = false;
    detector_t::model_window_size_t scale_one_window_size;
    int scale_one_shrinking_factor = 0;

    typedef boost::tuples::tuple<float, detector_t::occlusion_type_t, float> scales_and_occlusions_t;
    typedef boost::tuples::tuple<float, detector_t::occlusion_type_t, float, std::string> scales_and_occlusions_name_t;
    boost::unordered_set<scales_and_occlusions_name_t> scales_and_occlusions_and_name_set;
    BOOST_FOREACH(const detector_t& detector, get_detectors())
    {
        const scales_and_occlusions_name_t detector_scale_and_occlusion_and_semantic_category = boost::tuples::make_tuple(
                    detector.get_scale(),
                    detector.get_occlusion_type(),
                    detector.get_occlusion_level(),
                    detector.get_semantic_category());

        printf("Read model with scale %.3f, occlusion type '%s' and occlusion level %.3f\n",
               detector_scale_and_occlusion_and_semantic_category.get<0>(),
               get_occlusion_type_name(detector_scale_and_occlusion_and_semantic_category.get<1>()).c_str(),
               detector_scale_and_occlusion_and_semantic_category.get<2>());

        if(scales_and_occlusions_and_name_set.find(detector_scale_and_occlusion_and_semantic_category) == scales_and_occlusions_and_name_set.end())
        {
            // not in set
            scales_and_occlusions_and_name_set.insert(detector_scale_and_occlusion_and_semantic_category);

            const float detector_scale = detector_scale_and_occlusion_and_semantic_category.get<0>();
            if(detector_scale == 1.0)
            {
                scale_one_window_size = detector.get_model_window_size();
                scale_one_shrinking_factor = detector.get_shrinking_factor();
                found_scale_one = true;
            }
        }
        else
        {
            const std::string occlusion_type_name = get_occlusion_type_name(detector_scale_and_occlusion_and_semantic_category.get<1>());
            // two detectors with the same scale
            throw std::invalid_argument(
                        boost::str(boost::format(
                                       "IntegralChannelsDetectorModelsBundle received model data "
                                       "with two (conflicting) models at scale %.3f, occlusion type '%s' and occlusion level %.3f, and model name %s")
                                   % detector_scale_and_occlusion_and_semantic_category.get<0>()
                                   % occlusion_type_name
                                   % detector_scale_and_occlusion_and_semantic_category.get<2>()
                                   % detector_scale_and_occlusion_and_semantic_category.get<3>()));
        }

    }

    if(found_scale_one == false)
    {
        throw std::invalid_argument("IntegralChannelsDetectorModelsBundle received model data without scale 1.0");
    }

    // we check the model windows --
    BOOST_FOREACH(const detector_t& detector, get_detectors())
    {
        const float detector_scale = detector.get_scale();
        const detector_t::model_window_size_t &model_window_size = detector.get_model_window_size();

        const float
                delta_x = std::abs(scale_one_window_size.x()*detector_scale - model_window_size.x()),
                delta_y = std::abs(scale_one_window_size.y()*detector_scale - model_window_size.y());

        const float max_delta = 1; // we allow 1 pixel of rounding error
        if((delta_x > max_delta) or (delta_y > max_delta))
        {
            log_error() << "Model for scale " << detector_scale << " has an inconsistent model window size"
                        << std::endl;
            log_error() << boost::str(boost::format(
                                          "Model window size at scale %.3f (width, height) == (%i, %i)")
                                      % 1.0f % scale_one_window_size.x() % scale_one_window_size.y())
                        << std::endl;
            log_error() << boost::str(boost::format(
                                          "Model window size at scale %.3f (width, height) == (%i, %i)")
                                      % detector_scale % model_window_size.x() % model_window_size.y())
                        << std::endl;

            throw  std::invalid_argument("IntegralChannelsDetectorModelsBundle received model data with inconsistent "
                                         "model window size (with respect to their scale)");
        }

        // check the shrinking factors --
        if(scale_one_shrinking_factor != detector.get_shrinking_factor())
        {
            log_error() << boost::str(boost::format(
                                          "detector for scale %.3f has shrinking_factor == %i")
                                      % 1.0f % scale_one_shrinking_factor)
                        << std::endl;

            log_error() << boost::str(boost::format(
                                          "detector for scale %.3f has shrinking_factor == %i")
                                      % detector.get_scale() % detector.get_shrinking_factor())
                        << std::endl;

            throw  std::runtime_error("IntegralChannelsDetectorModelsBundle received model data with inconsistent "
                                      "shrinking factor (amongst the different scales)");
        }

    } // end of "for each detector"

    return;
}

} // end of namespace doppia

