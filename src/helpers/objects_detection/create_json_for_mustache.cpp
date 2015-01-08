#include "create_json_for_mustache.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/lexical_cast.hpp>

#include <string>

namespace doppia {

using namespace std;
using namespace boost;

typedef SoftCascadeOverIntegralChannelsModel::fast_stages_t cascade_stages_t;
typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t stage_t;

/// Helper function used to create the JSON needed for the code generation attempt
void create_json_for_mustache(std::vector<cascade_stages_t> &detection_cascade_per_scale)
{
    printf("Creating the json data file for mustache code generation (see tools/objects_detection/code_generation).\n"
           "This operation may take a minute or two... thanks for your patience.");
    property_tree::ptree data;

    data.add("num_scales", detection_cascade_per_scale.size());

    property_tree::ptree detector_scales_data;
    for(size_t scale_index=0; scale_index < detection_cascade_per_scale.size(); scale_index +=1)
    {
        property_tree::ptree detector_data;

        // detector_scales
        detector_data.add("scale_index", lexical_cast<string>(scale_index));

        const cascade_stages_t &detection_cascade = detection_cascade_per_scale[scale_index];

        property_tree::ptree stages_data;
        for(size_t stage_index=0; stage_index < detection_cascade.size(); stage_index +=1 )
        {
            //const cascade_stages_t::value_type
            const stage_t &stage = detection_cascade[stage_index];

            property_tree::ptree stage_data;
            stage_data.add("index", stage_index);
            stage_data.add("cascade_threshold", stage.cascade_threshold);

            {
                const stage_t::weak_classifier_t::feature_t &feature = stage.weak_classifier.level1_node.feature;
                stage_data.add("level1_node.feature.channel_index", feature.channel_index);
                stage_data.add("level1_node.feature.box.min_corner.x", feature.box.min_corner().x());
                stage_data.add("level1_node.feature.box.min_corner.y", feature.box.min_corner().y());
                stage_data.add("level1_node.feature.box.max_corner.x", feature.box.max_corner().x());
                stage_data.add("level1_node.feature.box.max_corner.y", feature.box.max_corner().y());

                stage_data.add("level1_node.feature_threshold", stage.weak_classifier.level1_node.feature_threshold);
            }

            {
                const stage_t::weak_classifier_t::feature_t &feature = stage.weak_classifier.level2_true_node.feature;
                stage_data.add("level2_true_node.feature.channel_index", feature.channel_index);
                stage_data.add("level2_true_node.feature.box.min_corner.x", feature.box.min_corner().x());
                stage_data.add("level2_true_node.feature.box.min_corner.y", feature.box.min_corner().y());
                stage_data.add("level2_true_node.feature.box.max_corner.x", feature.box.max_corner().x());
                stage_data.add("level2_true_node.feature.box.max_corner.y", feature.box.max_corner().y());

                stage_data.add("level2_true_node.feature_threshold", stage.weak_classifier.level2_true_node.feature_threshold);
                stage_data.add("level2_true_node.weight_true_leaf", stage.weak_classifier.level2_true_node.weight_true_leaf);
                stage_data.add("level2_true_node.weight_false_leaf", stage.weak_classifier.level2_true_node.weight_false_leaf);
            }

            {
                const stage_t::weak_classifier_t::feature_t &feature = stage.weak_classifier.level2_false_node.feature;
                stage_data.add("level2_false_node.feature.channel_index", feature.channel_index);
                stage_data.add("level2_false_node.feature.box.min_corner.x", feature.box.min_corner().x());
                stage_data.add("level2_false_node.feature.box.min_corner.y", feature.box.min_corner().y());
                stage_data.add("level2_false_node.feature.box.max_corner.x", feature.box.max_corner().x());
                stage_data.add("level2_false_node.feature.box.max_corner.y", feature.box.max_corner().y());

                stage_data.add("level2_false_node.feature_threshold", stage.weak_classifier.level2_false_node.feature_threshold);
                stage_data.add("level2_false_node.weight_true_leaf", stage.weak_classifier.level2_false_node.weight_true_leaf);
                stage_data.add("level2_false_node.weight_false_leaf", stage.weak_classifier.level2_false_node.weight_false_leaf);
            }

            // this is how JSON arrays are created with boost::property_tree (not great)
            stages_data.push_back(make_pair("", stage_data));
        } // end of "for each stage in the detector"

        detector_data.add_child("stages", stages_data);

        detector_scales_data.push_back( make_pair("", detector_data));
    } // end of "for each scale"

    data.add_child("detector_scales", detector_scales_data);

    const string filepath = "very_fast_detector_data.json";
    property_tree::json_parser::write_json(filepath, data);

    throw runtime_error("The json content for integral_channels_detector_kernel.mustache has been created in " + filepath);

    return;
}

} // end of namespace doppia
