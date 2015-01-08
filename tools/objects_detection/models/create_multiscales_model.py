#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This reads multiple trained models and merge them into a single multiscales model.
The program will do some check for consistency amongts the given models
"""

from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../helpers")

from detector_model_pb2 import DetectorModel, MultiScalesDetectorModel

import os, os.path
from optparse import OptionParser


def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes a set of trained single scales models and " \
        "creates a new multiscales model"

    parser.add_option("-i", "--input", dest="input_paths",
                       metavar="PATH", type="string", 
                       action="append", default=[],
                       help="path to one of the trained models. "
                       "This option should be repeated for each composing model. "
                       "Model scale will be deduced from the training data name. "
                       "To make everyone's life easier please provide models ordered from smaller to larger scale.")

    parser.add_option("-o", "--output", dest="output_path",
                       metavar="PATH", type="string",
                       help="path to the multiscales model file to be created")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)

    for path in options.input_paths:
        if not os.path.exists(path):
            parser.error("Could not find the input file %s" % path)        
    
    if not options.input_paths:
        parser.error("'input' option is required to run this program")

    # we normalize the paths
    options.input_paths = [os.path.normpath(x) for x in options.input_paths]

    if len(options.input_paths) < 2:
        parser.error("Should provide at least two models as input to make this program of any use")
            
    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to a non existing file")
    else:
        parser.error("'output' option is required to run this program")

    return options 


    

def create_multiscales_model(input_paths, output_path):


    multiscales_model = MultiScalesDetectorModel()
    
    detectors_names = []
    detectors_datasets = []
    scales = []
    for input_path in input_paths:
        model = DetectorModel()
        f = open(input_path, "rb")
        model.ParseFromString(f.read())
        f.close()

        if model.detector_name:
            detectors_names.append(model.detector_name)
            
        training_dataset_name = model.training_dataset_name
        split_dataset_name = training_dataset_name.split("_")
        
        if len(split_dataset_name) <= 2:
            print("WARNING: parsing training_dataset_name failed, using plan B")            
            print("training_dataset_name ==", training_dataset_name, 
                  "we expected something like DataSetName_octave_-0.5")        
            # input_path is expected to be of the kind
            # src/applications/boosted_learning/2011_10_13_model_octave_-1.proto.softcascade.bin
            # yes this is a hack !, that is why it is called plan B.
            split_dataset_name = os.path.basename(input_path).split(".")[0].split("_")
        
        assert len(split_dataset_name) > 2
        if split_dataset_name[-2] != "octave":
            raise RuntimeError("Input model should have a training dataset name of " \
                                "the type DataSetName_octave_-0.5. "\
                                "Instead received '%s'" % training_dataset_name)
        else:
            detectors_datasets.append(split_dataset_name[0])
        if split_dataset_name[-1] == "0.5":
            model_scale = 2**float(0.5849625)
        else:
            model_scale = 2**float(split_dataset_name[-1])
        print("Model %s is at scale %.3f (dataset name == %s)" % (
              input_path, model_scale, split_dataset_name))
        if model_scale in scales:
            raise RuntimeError("The input models define twice the scale %.2f, " \
                                "there should be only one model per scale" % model_scale)
        scales.append(model_scale)
        model_element = multiscales_model.detectors.add()
        model_element.scale = model_scale
        #model_element.detector_name # we leave the element model name non-initialized
        model_element.training_dataset_name = training_dataset_name # required
        
        model_element.model_window_size.CopyFrom(model.model_window_size)
        model_element.object_window.CopyFrom(model.object_window)
        model_element.detector_type = model.detector_type
        if model.detector_type == DetectorModel.LinearSvm:
            model_element.linear_svm_model.CopyFrom(model.linear_svm_model)
        elif model.detector_type == DetectorModel.SoftCascadeOverIntegralChannels:
            model_element.soft_cascade_model.CopyFrom(model.soft_cascade_model)
            print("Model %s has %i stages" % \
                    (input_path, len(model_element.soft_cascade_model.stages)))
        #elif model.detector_type == DetectorModel.HoughForest:
            #model_element.hough_forest_model.CopyFrom(model.hough_forest_model)

    # end of "for each input path"

    #print("scales ==", scales)    
    #print("detectors_names ==", detectors_names)    
    #print("detectors_datasets ==", detectors_datasets)    
    
    multiscales_model.detector_name = "Scales " + ", ".join([str(x) for x in set(scales)])
    multiscales_model.training_dataset_name = " and ".join([str(x) for x in set(detectors_datasets)])
    
    print("Multiscales detector name ==", multiscales_model.detector_name)
    print("Multiscales training dataset name ==", multiscales_model.training_dataset_name)

    out_file = open(output_path, "wb")
    out_file.write(multiscales_model.SerializeToString())
    out_file.close()
    print("Create output model file", output_path)
    return


def main():
    options = parse_arguments()    
    
    create_multiscales_model(options.input_paths, options.output_path)
        
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




        
