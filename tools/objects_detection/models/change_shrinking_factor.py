#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program will change the shrinking factor of a trained classifier
"""


from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")

from detections_pb2 import Box
from detector_model_pb2 import DetectorModel, MultiScalesDetectorModel

import os, os.path#, glob
from optparse import OptionParser

from plot_detector_model import read_model

def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program a trained classifier models and " \
        "creates a new model with a different shrinking factor"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string",
                       help="path to the input model file")

    parser.add_option("-s", "--shrinking_factor", dest="shrinking_factor",
                       metavar="INTEGER", type="int",
                       help="new shrinking factor value")

    parser.add_option("-o", "--output", dest="output_path",
                       metavar="PATH", type="string",
                       help="path to the model file to be created")

    (options, args) = parser.parse_args()
    #print (options, args)


    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file %s" % options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if not options.shrinking_factor:
        parser.error("'shrinking_factor' option is required to run this program")

    options.input_path = os.path.normpath(options.input_path)

    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to a non existing file")
    else:
        parser.error("'output' option is required to run this program")

    return options


def scale_feature(feature, scaling_factor):

    box = feature.box
    box.min_corner.x = int(box.min_corner.x*scaling_factor)
    box.min_corner.y = int(box.min_corner.y*scaling_factor)

    box.max_corner.x = int(box.max_corner.x*scaling_factor)
    box.max_corner.y = int(box.max_corner.y*scaling_factor)
    return


def scale_stump(stump, scaling_factor):

    scale_feature(stump.feature, scaling_factor)
    stump.feature_threshold *= (scaling_factor*scaling_factor)
    return


def change_shrinking_factor(new_shrinking_factor, detector):

    model = detector.soft_cascade_model

    if not model:
        raise Exception("Only SoftCascadeOverIntegralChannels models are supported, " \
                        "received {0}".format(detector.detector_type))

    #print("model.shrinking_factor", model.shrinking_factor)
    scaling_factor = model.shrinking_factor / float(new_shrinking_factor)
    print("scaling_factor ==", scaling_factor)

    if scaling_factor == 1.0:
        raise Exception("Input model already has shrinking factor {0}".format(new_shrinking_factor))

    for stage in model.stages:
        if stage.feature_type == stage.Stumps:
            scale_stump(stage.decision_stump, scaling_factor)
        elif stage.feature_type == stage.Level2DecisionTree:
            for node in stage.level2_decision_tree.nodes:
                scale_stump(node.decision_stump, scaling_factor)
            # end of "for all nodes"
        elif stage.feature_type == stage.LevelNDecisionTree:
            for node in stage.levelN_decision_tree.nodes:
                scale_stump(node.decision_stump, scaling_factor)
            # end of "for all nodes"
        else:
            raise Exception("Received an unhandled stage type")
    # end of "for all stages"

    model.shrinking_factor = new_shrinking_factor


    print("detector.model_window_size",
          (detector.model_window_size.x, detector.model_window_size.y))

    object_window = detector.object_window
    print("detector.object_window",
          ((object_window.min_corner.x, object_window.min_corner.y),
           (object_window.max_corner.x, object_window.max_corner.y)))

    return


def replace_scale_hack(model):
    """
    Small hack to replace a model of a given scale
    """
    assert isinstance(model, MultiScalesDetectorModel)

    model_path = "/users/visics/mmathias/devel/doppia/src/applications/boosted_learning/eccvWorkshop/2012_05_20_67022_trained_model_octave_-1.proto.bin.bootstrap2"
    replacement_model = read_model(model_path)
    assert isinstance(replacement_model, DetectorModel)
    replacement_scale = 0.5

    print("Replacing model of scale {0} " \
          "by the model read at {1}".format(replacement_scale, model_path))

    for detector in model.detectors:
        if detector.scale == replacement_scale:
            detector.CopyFrom(replacement_model)
            detector.scale = replacement_scale # we set the proper new scale
            break

    return


def create_new_shrinking_factor_model(input_path, new_shrinking_factor, output_path):


    input_model = read_model(input_path)

    model_class = input_model.__class__
    output_model = input_model

    if True:
        if model_class is MultiScalesDetectorModel:
            for detector in output_model.detectors:
                change_shrinking_factor(new_shrinking_factor, detector)

        elif model_class is DetectorModel:
            change_shrinking_factor(new_shrinking_factor, output_model)

        else:
            raise Exception("Received an unmanaged detector model class {0}".format(model_class) )

    #replace_scale_hack(output_model)

    output_content = output_model.SerializeToString()
    out_file = open(output_path, "wb")
    out_file.write(output_content)
    out_file.close()
    print("Created output model file", output_path)
    return


def main():
    options = parse_arguments()

    create_new_shrinking_factor_model(options.input_path,
                                      options.shrinking_factor,
                                      options.output_path)

    print("End of game, have a nice day!")
    return


if __name__ == "__main__":

    # Import Psyco if available
    try:
        import psyco
        psyco.full()
    except ImportError:
        #print("(psyco not found)")
        pass
    else:
        print("(using psyco)")

    main()



# end of file