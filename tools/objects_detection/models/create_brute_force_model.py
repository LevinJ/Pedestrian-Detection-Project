#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper script to create the models bundle containing all the brute force models.
"""


from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../helpers")

from detector_model_pb2 import DetectorModel, SoftCascadeOverIntegralChannelsStage
from mirror_occluded_model import get_occlusion_level_and_type

def create_brute_force_model():
    

    bottom_path_pattern = "/users/visics/rbenenso/code/doppia/src/applications/boosted_learning/occluded_models/2012_10_16_2030_left_right_bottom_occlusions_0_to_half_brute_force/model_bottom_crop_%i"
    right_path_pattern = "/users/visics/rbenenso/code/doppia/src/applications/boosted_learning/occluded_models/2012_10_16_2030_left_right_bottom_occlusions_0_to_half_brute_force/model_right_crop_%i"

    bottom_models_paths = [bottom_path_pattern % (i*4) for i in range(1, 17)]
    right_models_paths = [right_path_pattern % (i*4) for i in range(1, 9)]
    
    all_models_paths = bottom_models_paths + right_models_paths
    
    model_window_size =  (64, 128)
    model_object_window =  (8, 16, 56, 112) # (min_x, min_y, max_x, max_y)
    shrinking_factor = 4

    for model_path in all_models_paths:    
        input_path = model_path
        input_model = DetectorModel()
        f = open(input_path, "rb")
        input_model.ParseFromString(f.read())
        f.close()
        
        print("Read model", input_model.detector_name or "(no name)", 
              "was trained on ", input_model.training_dataset_name)
    
        assert input_model.soft_cascade_model.shrinking_factor == shrinking_factor
        
        input_model.model_window_size.x = model_window_size[0]
        input_model.model_window_size.y = model_window_size[1]

        input_model.object_window.min_corner.x = model_object_window[0]
        input_model.object_window.min_corner.y = model_object_window[1]
        input_model.object_window.max_corner.x = model_object_window[2]
        input_model.object_window.max_corner.y = model_object_window[3]
        
        print("Updated model window size and object window")
        output_path = input_path
        output_model = input_model
        out_file = open(output_path, "wb")
        out_file.write(output_model.SerializeToString())
        out_file.close()
        print("Updated model file", output_path)
    # end of "for each model path"    
    
    return


import os, os.path
from optparse import OptionParser


def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes a left (or right) occluded model and creates its mirror right (or left) occluded one"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string", 
                       help="path to a trained model.")

    parser.add_option("-o", "--output", dest="output_path",
                       metavar="PATH", type="string",
                       help="path to the model fiel to be created")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)


    if not options.input_path:
        parser.error("'input' option is required to run this program")
    
    if not os.path.exists(options.input_path):
        parser.error("Could not find the input file %s" % path)        
    

    # we normalize the path
    options.input_path = os.path.normpath(options.input_path)
            
    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to a non existing file")
    else:
        parser.error("'output' option is required to run this program")

    return options 



def main():
    #options = parse_arguments()    
    
    create_brute_force_model()
        
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

