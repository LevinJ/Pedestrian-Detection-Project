#include "MultiScalesIntegralChannelsModel.hpp"

#include "detector_model.pb.h"

#include "helpers/Log.hpp"

#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>
#include <boost/format.hpp>

//#include <boost/math/special_functions/round.hpp>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "MultiScalesIntegralChannelsModel");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "MultiScalesIntegralChannelsModel");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "MultiScalesIntegralChannelsModel");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "MultiScalesIntegralChannelsModel");
}

std::vector<float> upscaling_factors;
std::vector<float> downscaling_factors;

} // end of anonymous namespace



namespace doppia {

/// ProtobufModelDataType shoudl be doppia_protobuf::MultiScalesDetectorModel or DetectorModelsBundle
template<typename ProtobufModelDataType>
void normalized_multiscales_impl(const ProtobufModelDataType &model,
                                 ProtobufModelDataType &rescaled_model)
{
    rescaled_model.CopyFrom(model);

    const float desired_max_score = 1;

    //const bool print_original_multiscales_model = true;
    const bool print_original_multiscales_model = false;
    if(print_original_multiscales_model)
    {
        log_debug() << "Full multiscales/bundle model before normalization" << std::endl;
        log_debug() << model.DebugString() << std::endl;
    }


    log_info() << "Multiscales/bundle model is being normalized to have maximum detection score == "
               << desired_max_score << std::endl;

    for(int detector_index=0; detector_index < rescaled_model.detectors_size(); detector_index+=1)
    {
        doppia_protobuf::SoftCascadeOverIntegralChannelsModel &soft_cascade =
                *(rescaled_model.mutable_detectors(detector_index)->mutable_soft_cascade_model());

        float weights_sum = 0;
        for(int stage_index = 0; stage_index < soft_cascade.stages_size(); stage_index += 1)
        {
            weights_sum += soft_cascade.stages(stage_index).weight();
        } // end for "for each cascade stage"


        log_debug() << "Detector model " << detector_index
                    << ", out of training had a max score of " << weights_sum << std::endl;

        const float weight_scaling_factor = desired_max_score / weights_sum;

        for(int stage_index = 0; stage_index < soft_cascade.stages_size(); stage_index += 1)
        {
            doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage =
                    *(soft_cascade.mutable_stages(stage_index));

            stage.set_weight(stage.weight() * weight_scaling_factor);

            if(stage.cascade_threshold() > -1E5)
            { // rescale the cascade threshold if it has a non-absurd value
                stage.set_cascade_threshold(stage.cascade_threshold() * weight_scaling_factor);
            }
        } // end for "for each cascade stage"

    } // end for "for each model"

    return;
}


void normalized_multiscales(const doppia_protobuf::MultiScalesDetectorModel &model,
                            doppia_protobuf::MultiScalesDetectorModel &rescaled_model)
{
    return normalized_multiscales_impl<>(model, rescaled_model);
}


void normalized_bundle(const doppia_protobuf::DetectorModelsBundle &model,
                       doppia_protobuf::DetectorModelsBundle &rescaled_model)
{
    return normalized_multiscales_impl<>(model, rescaled_model);
}


/// this constructor will copy the protobuf data into a more efficient data structure
MultiScalesIntegralChannelsModel::MultiScalesIntegralChannelsModel(const doppia_protobuf::MultiScalesDetectorModel &model)
    :IntegralChannelsDetectorModelsBundle()
{
    if(model.has_detector_name())
    {
        log_info() << "Parsing multiscales model " << model.detector_name() << std::endl;
    }

    if(model.detectors_size() == 0)
    {
        throw std::invalid_argument("MultiScalesIntegralChannelsModel received an empty model");
    }

    doppia_protobuf::MultiScalesDetectorModel rescaled_model;
    normalized_multiscales(model, rescaled_model);

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


MultiScalesIntegralChannelsModel::~MultiScalesIntegralChannelsModel()
{
    // nothing to do here
    return;
}


void MultiScalesIntegralChannelsModel::sanity_check() const
{

    bool found_scale_one = false;
    detector_t::model_window_size_t scale_one_window_size;
    int scale_one_shrinking_factor = 0;

    boost::unordered_set<float> scales_set;
    BOOST_FOREACH(const detector_t& detector, get_detectors())
    {
        const float detector_scale = detector.get_scale();

        if(scales_set.find(detector_scale) == scales_set.end())
        {
            // not in set
            scales_set.insert(detector_scale);

            if(detector_scale == 1.0)
            {
                scale_one_window_size = detector.get_model_window_size();
                scale_one_shrinking_factor = detector.get_shrinking_factor();
                found_scale_one = true;
            }
        }
        else
        {
            // two detectors with the same scale
            throw std::invalid_argument(
                        boost::str(boost::format(
                                       "MultiScalesIntegralChannelsModel received model data "
                                       "with two (conflicting) models for scale %.3f")
                                   % detector_scale));
        }

    }

    if(found_scale_one == false)
    {
        throw std::invalid_argument("MultiScalesIntegralChannelsModel received model data without scale 1.0");
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

            throw  std::invalid_argument("MultiScalesIntegralChannelsModel received model data with inconsistent "
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

            throw  std::runtime_error("MultiScalesIntegralChannelsModel received model data with inconsistent "
                                      "shrinking factor (amongst the different scales)");
        }

    } // end of "for each detector"

    return;
}

} // end of namespace doppia
