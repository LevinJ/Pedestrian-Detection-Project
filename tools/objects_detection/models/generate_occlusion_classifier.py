#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The program reads an existing model file and generates models for different amounts of occlusions
"""
from __future__ import print_function

from detector_model_pb2 import DetectorModel
import detector_model_pb2 as dm 
import detections_pb2 as det
import os, os.path#, glob
from optparse import OptionParser
from plot_detector_model import read_cascade, read_model



def add_feature_to_channels(channel_index, box, weight):
    #for y in range(box.min_corner.y, box.max_corner.y+1):
    #    for x in range(box.min_corner.x, box.max_corner.x+1):
    #        channels[channel_index, y, x] += weight
    slice_y = slice(box.min_corner.y, box.max_corner.y+1)
    slice_x = slice(box.min_corner.x, box.max_corner.x+1)
    channels[channel_index, slice_y, slice_x] += weight
    
    if print_the_features:
        print("box (min x,y) (max x,y) ==", 
              (box.min_corner.x, box.min_corner.y), 
              (box.max_corner.x, box.max_corner.y),
              "\tweight ==", weight)        
        
    return



def get_stump_box(stump):
    
    feature = stump.feature
    return feature.box
    
    
def get_node_boxes(node):
    if node.decision_stump:
        return get_stump_box(node.decision_stump)


def getMaxXY_tree(tree):
    nodes = []
    for node in tree.nodes:
        nodes.append(get_node_boxes(node))
    #check for the maximal y position
    maxy = -1
    maxx = -1
    for node in nodes:
        x = node.max_corner.x
        y = node.max_corner.y
        if x> maxx:
            maxx = x
        if y> maxy:
            maxy = y

    return [maxx, maxy]


    
def update_cascade(old_cascade, new_cascade, yThresh):
    new_cascade.Clear()
    
    for i, stage in enumerate(old_cascade.stages):
        tree = 0
        if stage.feature_type == stage.Level2DecisionTree:
            maxx, maxy = getMaxXY_tree(stage.level2_decision_tree)
        else:
            print("stage.feature_type ==", stage.feature_type)
            raise Exception("Received an unhandled stage.feature_type")
        if maxy< yThresh:
            new_stage = new_cascade.stages.add()
            new_stage.CopyFrom(stage)
    return

def update_cascade_left(old_cascade, new_cascade, xThresh):
    new_cascade.Clear()
    
    for i, stage in enumerate(old_cascade.stages):
        tree = 0
        if stage.feature_type == stage.Level2DecisionTree:
            maxx, maxy = getMaxXY_tree(stage.level2_decision_tree)
        else:
            print("stage.feature_type ==", stage.feature_type)
            raise Exception("Received an unhandled stage.feature_type")
        if maxx< xThresh:
            new_stage = new_cascade.stages.add()
            new_stage.CopyFrom(stage)
    return
    


def generate_occlusionClassifier(input_model):
    width = 32
    half_width = 16

    model=read_model(input_model)
    for i in range(1+half_width):
        yThresh = half_width-i
        new_model = DetectorModel()
        new_model.CopyFrom(model)
        if model.model_window_size:
            model_width = model.model_window_size.x
            model_height = model.model_window_size.y        
        print("model.detector_type", model.detector_type)

        if model.detector_type == model.SoftCascadeOverIntegralChannels:
            old_cascade = model.soft_cascade_model    
            new_cascade = new_model.soft_cascade_model    
            print("Model has %i stages" % len(old_cascade.stages))
            update_cascade(old_cascade, new_cascade, width-yThresh)



            output_path = input_model + "_artificial_crop_" + str(yThresh*4)
    

            out_file = open(output_path, "wb")
            out_file.write(new_model.SerializeToString())
            out_file.close()
            print("Create output model file", output_path)

def generate_occlusionClassifier_left(input_model):
    height = 16
    half_height = 8

    model=read_model(input_model)
    for i in range(1+half_height):
        xThresh = half_height-i
        new_model = DetectorModel()
        new_model.CopyFrom(model)
        if model.model_window_size:
            model_width = model.model_window_size.x
            model_height = model.model_window_size.y        
        print("model.detector_type", model.detector_type)

        if model.detector_type == model.SoftCascadeOverIntegralChannels:
            old_cascade = model.soft_cascade_model    
            new_cascade = new_model.soft_cascade_model    
            print("Model has %i stages" % len(old_cascade.stages))
            update_cascade_left(old_cascade, new_cascade, height-xThresh)



            output_path = input_model + "_artificial_crop_" + str(xThresh*4)
    

            out_file = open(output_path, "wb")
            out_file.write(new_model.SerializeToString())
            out_file.close()
            print("Create output model file", output_path)



def parse_arguments():
    parser = OptionParser()
    parser.description = \
    "The program reads an existing model file and generates models for different amounts of occlusions"

    parser.add_option("-i", "--input_model", dest="input_model",
                       type="string", 
                       help="path to the trained model.")

    parser.add_option("-c", "--classifier_type", dest="classifier_type",
                       type="string", 
                       help="this option is required and denotes the type of the classifier: \"up\" or \"left\"")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)

    if not options.classifier_type:
        parser.error("'classifier_type' has to be specified")
    if not options.input_model:
        parser.error("'input' option is required to run this program")
    else:
        if not os.path.exists(options.input_model):
            parser.error("Could not find the input file %s" % options.input_model)        

    return options 
        

def main():
    options = parse_arguments()    
    
    if options.classifier_type == "up":
        generate_occlusionClassifier(options.input_model)
    elif options.classifier_type =="left":
        generate_occlusionClassifier_left(options.input_model)
    else:
        raise Exception("classifier type must be eighter 'up or 'left'")
        return
        
    print("End of game, have a nice day!")
    return


if __name__ == "__main__":
    main()
