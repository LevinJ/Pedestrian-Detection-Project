#ifndef STIXELWORLDESTIMATORFACTORY_HPP
#define STIXELWORLDESTIMATORFACTORY_HPP

#include "AbstractStixelWorldEstimator.hpp"

#include "video_input/AbstractVideoInput.hpp"

namespace doppia {

// forward declaration
class VideoFromFiles;

class StixelWorldEstimatorFactory
{
public:
    static boost::program_options::options_description get_args_options();

    /// commonly used new_instance method
    static AbstractStixelWorldEstimator* new_instance(const boost::program_options::variables_map &options,
                                                      AbstractVideoInput &video_input);


    /// variant method used inside objects_detection_lib
    static AbstractStixelWorldEstimator* new_instance(const boost::program_options::variables_map &options,
                                                      const AbstractVideoInput::dimensions_t &input_dimensions,
                                                      const MetricStereoCamera &camera,
                                                      const float ground_plane_prior_pitch,
                                                      const float ground_plane_prior_roll,
                                                      const float ground_plane_prior_height,
                                                      VideoFromFiles *video_input_p = NULL);

};

} // end of namespace doppia

#endif // STIXELWORLDESTIMATORFACTORY_HPP
