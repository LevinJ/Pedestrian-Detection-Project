#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transform a left occluded model into a right occluded model
"""


from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../helpers")

from detector_model_pb2 import DetectorModel, SoftCascadeOverIntegralChannelsStage

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


def get_occlusion_level_and_type(model, input_path):

    input_filename = os.path.basename(input_path)
    input_filename = input_path
    
    
    if model.HasField("occlusion_type"):            
        occlusion_type = model.occlusion_type
        print("File %s has occlusion type %i" % (input_filename, occlusion_type))
    else:
        #if occlusion_level == 0:
        if False:
            # no occlusion, setting a default occlusion type (that will be ignored)
            occlusion_type = DetectorModel.LeftOcclusion
        else:
            filename = input_filename.lower()
            if filename.find("left") != -1:
                print("Filename %s contains 'left', assuming LeftOcclusion" % input_filename)
                occlusion_type = DetectorModel.LeftOcclusion
            elif filename.find("right") != -1:
                print("Filename %s contains 'right', assuming RightOcclusion" % input_filename)
                occlusion_type = DetectorModel.RightOcclusion
            elif filename.find("bottom") != -1:
                print("Filename %s contains 'bottom', assuming BottomOcclusion" % input_filename)
                occlusion_type = DetectorModel.BottomOcclusion
            else:
                raise RuntimeError("Model contains no occlusion_type field and filename contains no clue, " 
                                    "failed to find the occlusion type")
                #print("Failed to deduce the occlusion type from the file name, assuming LeftOcclusion" )
                #occlusion_type = LeftOcclusion
                #print("No occlusion type found on the model assuming BottomOcclusion")
                #occlusion_type = DetectorModel.BottomOcclusion

    input_filename = os.path.basename(input_path)

    if model.HasField("occlusion_level"):
        occlusion_level = model.occlusion_level
    else:
        print("No occlusion level found on the model trying to parse the model file name")
        splits = input_filename.split("_")
        if splits[-2].endswith("crop"):
            crop_value = float(splits[-1].split(".")[0])
            
            if occlusion_type in [DetectorModel.BottomOcclusion, DetectorModel.TopOcclusion]:
                occlusion_level = float(crop_value)/128
            else: # left and right
                occlusion_level = float(crop_value)/64
        else:
            print("splits ==", splits)
            raise RuntimeError("Failed to deduce the occlusion level from the file name" )
            #print("Failed to deduce the occlusion level from the file name, assuming no occlusion" )
            #occlusion_level = 0

   
    return occlusion_level, occlusion_type   
  

def get_model_width_and_height(model):

    if model.model_window_size:
        model_width = model.model_window_size.x
        model_height = model.model_window_size.y        
    else:
        # we use the INRIAPerson as default
        model_width = 64 
        model_height = 128 
    print("Model size (width, height) == ", (model_width, model_height))    
    
    
    if model.object_window:
        b = model.object_window
        print("Model object window (min_x, min_y, max_x, max_y) == ",
              (b.min_corner.x, b.min_corner.y, b.max_corner.x, b.max_corner.y))
    
    shrinking_factor = 4 # best guess
    if model.soft_cascade_model:
        shrinking_factor = model.soft_cascade_model.shrinking_factor
        print("Model shrinking factor ==", shrinking_factor)
    
    # we take into account the shrinking factor    
    model_width /= shrinking_factor
    model_height /= shrinking_factor
 
    return model_width, model_height
    

def flip_stage(stage, model_width, model_height):
    
    if stage.feature_type == SoftCascadeOverIntegralChannelsStage.Level2DecisionTree:
        flip_binary_decision_tree(stage.level2_decision_tree, model_width, model_height)
    elif stage.feature_type == SoftCascadeOverIntegralChannelsStage.LevelNDecisionTree:
        flip_binary_decision_tree(stage.levelN_decision_tree, model_width, model_height)
    elif stage.feature_type == SoftCascadeOverIntegralChannelsStage.Stumps:
        flip_decision_stump(stage.decision_stump, model_width, model_height)
    elif stage.feature_type == SoftCascadeOverIntegralChannelsStage.StumpSet:
        flip_stump_set(stage.stump_set, model_width, model_height)
    else:
        raise RuntimeException("flip_stage received an unmanaged feature type")
    return


def flip_binary_decision_tree(decision_tree, model_width, model_height):
    for node in decision_tree.nodes:
        flip_decision_stump(node.decision_stump, model_width, model_height)
    return


def flip_stump_set(stump, model_width, model_height):
    for node in decision_tree.nodes:
        flip_decision_stump(node.decision_stump, model_width, model_height)
    return


def flip_decision_stump(stump, model_width, model_height):
    flip_hog6_luv_integral_channel_feature(stump.feature, model_width, model_height)
    return


def flip_hog6_luv_integral_channel_feature(feature, model_width, model_height):
    """
    Ten channels:
        6 HOG channels, 1 Magnitude, 3 LUV
    Channel 0: horizontal
    Channel 1: slightly up
    Channel 2: more up
    Channel 3: Vertical 
    Channel 4: more down
    Channel 5: slightly down
    """
    
    original_channel_index = feature.channel_index
    if original_channel_index == 1:
        feature.channel_index = 5        
    elif original_channel_index == 2:
        feature.channel_index = 4
    elif original_channel_index == 4:
        feature.channel_index = 2
    elif original_channel_index == 5:
        feature.channel_index = 1
    else: # channels 0, 3 and others, stay the same
        feature.channel_index = original_channel_index

    flip_box(feature.box, model_width, model_height)
    return

def flip_x(x, model_width):
    
    """
    half_width = model_width/2

    if x < half_width:
        #new_x = (half_width - x) + half_width
        new_x = width - x
    elif x > half_width:
        #new_x = half_width - (x - half_width)
        new_x = width - x
    else:
        new_x = x # == half_width
    """    
    # -1 to keep things in the range [0, model_width) (non-inclusive)
    return (model_width-1) - x


def flip_box(box, model_width, model_height):
    """
    We flip along the vertical axis
    """

    box.min_corner.x = flip_x(box.min_corner.x, model_width)
    box.max_corner.x = flip_x(box.max_corner.x, model_width)

    # swap if needed
    if box.max_corner.x < box.min_corner.x:
        box.min_corner.x, box.max_corner.x = box.max_corner.x, box.min_corner.x
    return
    

def create_mirrored_occluded_model(input_path, output_path):


    input_model = DetectorModel()
    f = open(input_path, "rb")
    input_model.ParseFromString(f.read())
    f.close()
        
    output_model = DetectorModel()
    
    print("Read model", input_model.detector_name or "(no name)", 
          "was trained on ", input_model.training_dataset_name)
    
    
    occlusion_level, occlusion_type = get_occlusion_level_and_type(input_model, input_path)
    

    if occlusion_level == 0 or (occlusion_type not in [DetectorModel.LeftOcclusion, DetectorModel.RightOcclusion]):
        print("occlusion_level, occlusion_type == %s, %s" % (occlusion_level, occlusion_type))
        raise RuntimeError("Input model has no left or right occlusion")
    
    
    output_model.CopyFrom(input_model)
    
    
    output_model.occlusion_level = occlusion_level
    
    if occlusion_type == DetectorModel.LeftOcclusion:
        output_model.occlusion_type = DetectorModel.RightOcclusion
        output_model.detector_name += " transformed to right model" 
    elif occlusion_type == DetectorModel.RightOcclusion:
        output_model.occlusion_type = DetectorModel.LeftOcclusion
        output_model.detector_name += " transformed to left model" 
    else:
        raise RuntimeError("Only left and right occlusions are currently handled")

    
    if not output_model.soft_cascade_model:
        raise RuntimeError("Only SoftCascadeOverIntegralChannels models are currently supported")
        
    
    model_width, model_height = get_model_width_and_height(input_model)
    
    if output_model.soft_cascade_model.channels_description != "hog6_luv":
        raise RuntimeError("Only hog6_luv channels are currently supported")
    
    
    for stage in output_model.soft_cascade_model.stages:
        flip_stage(stage, model_width, model_height)
    
    print("Flipped %i stages in the cascade" % len(output_model.soft_cascade_model.stages) )    
    
    out_file = open(output_path, "wb")
    out_file.write(output_model.SerializeToString())
    out_file.close()
    print("Create output model file", output_path)
    return


def main():
    options = parse_arguments()    
    
    create_mirrored_occluded_model(options.input_path, options.output_path)
        
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

