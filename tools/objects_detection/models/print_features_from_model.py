#!/usr/bin/env python
# -*- coding: utf-8 -*-


import plot_detector_model as pdm
import os, os.path
import detector_model_pb2


from optparse import OptionParser

def print_feature(feature):
    box = feature.box
    channel = feature.channel_index
    x = box.min_corner.x
    y = box.min_corner.y
    w = box.max_corner.x - x
    h = box.max_corner.y - y
    print("%d %d %d %d %d" %(x, y, w,h, channel));
    
    
def read_stump(stump, weight):
    
              
    feature = stump.feature
    print_feature(feature)    
    return 
    
    
def read_node(node, weight):
    if node.decision_stump:
        read_stump(node.decision_stump, weight)
    return
    

def read_tree(tree, weight):
    nodes = []
    for node in tree.nodes:
        nodes.append(read_node(node, weight))
    return


def read_cascade(cascade):
    
    for i, stage in enumerate(cascade.stages):
        #if i>1500:
		#	break

        if stage.feature_type == stage.Level2DecisionTree:
            read_tree(stage.level2_decision_tree, stage.weight)
        elif stage.feature_type == stage.Stumps:
            read_stump(stage.decision_stump, stage.weight)
        else:
            print("stage.feature_type ==", stage.feature_type)
            raise Exception("Received an unhandled stage.feature_type")
        #print("weight:", stage.weight)
        if False and stage.cascade_threshold > -1E5:
            # we only print "non trivial" values
            print("stage %i cascade threshold:" % i , stage.cascade_threshold)
            
    return

def read_model(model_filename):
        
    model = detector_model_pb2.DetectorModel()
    f = open(model_filename, "rb")
    model.ParseFromString(f.read())
    f.close()
    
    if not model.IsInitialized():
        print("Input file seems not to be a DetectorModel, " \
              "trying as MultiScalesDetectorModel")

        model = detector_model_pb2.MultiScalesDetectorModel()
        f = open(model_filename, "rb")
        model.ParseFromString(f.read())
        f.close()

    if not model.IsInitialized():
        print("Input file seems not to be "\
              "a DetectorModel nor a  MultiScalesDetectorModel")
        raise Exception("Unknown input file format")

    return model



def print_detector_model(model):

    cascade = model.soft_cascade_model    
    read_cascade(cascade)

def main():

    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and prints its content"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the model file")
    (options, args) = parser.parse_args()
    #print (options, args)
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    model_filename = options.input_path

    model = read_model(model_filename)
    
    if type(model) is detector_model_pb2.MultiScalesDetectorModel:

       raise ("only for non multiscale models")
    else: # assume single scale model
        print_detector_model(model)

    return
        

if __name__ == '__main__':
    main()
