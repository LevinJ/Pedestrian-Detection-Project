#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compares two models
"""

from __future__ import print_function

import detector_model_pb2

from itertools import izip

from plot_detector_model import read_model

def read_model_old(model_filepath):
    model = detector_model_pb2.DetectorModel()
    f = open(model_filepath, "rb")
    model.ParseFromString(f.read())
    f.close()
    assert model.IsInitialized()
    
    return model
    
def compare_weights(a, b):
    return a.weight < b.weight
    

def print_feature(feature, feature_threshold, weight):
    #feature.channel_index
    box = feature.box
    print("box (min x,y) (max x,y) ==", 
              (box.min_corner.x, box.min_corner.y), 
              (box.max_corner.x, box.max_corner.y),
              "threshold", feature_threshold,  
              "\tweight ==", weight)            
    return
    
    
def read_stump(stump, weight):
    feature = stump.feature
    print_feature(feature, stump.feature_threshold, weight)    
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

def compare_boxes(a, b):
    same = True
    same &= a.min_corner.x == b.min_corner.x
    same &= a.min_corner.y == b.min_corner.y
    same &= a.max_corner.x == b.max_corner.x
    same &= a.max_corner.y == b.max_corner.y
    return same
    
def compare_trees(a, b):
    
    same = True
    same &= len(a.nodes) == len(b.nodes)
    for na, nb in izip(a.nodes, b.nodes):
        stump_a = na.decision_stump
        stump_b = nb.decision_stump
        same &= stump_a.feature.channel_index == stump_b.feature.channel_index
        same &= stump_a.feature_threshold == stump_b.feature_threshold
        same &= compare_boxes(stump_a.feature.box, stump_b.feature.box)
    
    return same


def compare_detectors(model_a, model_b):
    
    print("Read the two models, starting comparison")
    stages_a = list(model_a.soft_cascade_model.stages)
    stages_b = list(model_b.soft_cascade_model.stages)
    
    assert len(stages_a) == len(stages_b)
    
    print(len(stages_a), "==", len(stages_b))
    
    weights_a = [stage.weight for stage in stages_a]
    weights_b = [stage.weight for stage in stages_b]
    sorted_weights_a = sorted(weights_a)
    sorted_weights_b = sorted(weights_b)
    assert sorted_weights_a == sorted_weights_b
    
    #sorted_stages_a = sorted(stages_a, compare_weights)
    #sorted_stages_b = sorted(stages_b, compare_weights)
    
    sorted_stages_a = [(stage.level2_decision_tree, stage.weight) for stage in stages_a ]
    sorted_stages_b = [(stage.level2_decision_tree, stage.weight) for stage in stages_b ]
    
    sorted_stages_a.sort(lambda x,y: x[1] < y[1])
    sorted_stages_b.sort(lambda x,y: x[1] < y[1])
    
    assert sorted_stages_a != stages_a
    assert sorted_stages_b != stages_b
    
    differences = 0
    counter = 0
    for a, b in izip(sorted_stages_a, sorted_stages_b):
        counter += 1
        
        is_the_same = compare_trees(a[0], b[0])    
        if not is_the_same:
            print("A"); read_tree(a[0], a[1])
            print("B"); read_tree(a[0], a[1])
            print("Found two different stages after %i comparisons" % counter)
            differences += 1
        if differences > 5:
            raise Exception("To many differences, the two models are not identical")

# end of compare_detectors

print("Reading the two models; please wait...")

#model_a_path = "../../src/applications/boosted_learning/2011_10_12_64789_trained_model_octave_0.proto.softcascade.bin"
#model_b_path = "../../src/applications/boosted_learning/2011_11_04_61124_2011_10_12_64789_trained_model_octave_0.proto.softcascade.softcascade.bin"

model_a_path = "/users/visics/rbenenso/code/europa/code/kul-rwth/objects_detection_lib/2012_04_04_1417_trained_model_multiscales_synthetic_softcascade.proto.bin"
model_b_path = "2012_04_04_1417_trained_model_multiscales_synthetic_softcascade.shrinking_factor_2_back_to_4.proto.bin"

model_a = read_model(model_a_path)
model_b = read_model(model_b_path)

if isinstance(model_a, detector_model_pb2.MultiScalesDetectorModel):
    assert isinstance(model_b, detector_model_pb2.MultiScalesDetectorModel)
    for i in range(len(model_a.detectors)):
        detector_a = model_a.detectors[i]
        detector_b = model_b.detectors[i]
        compare_detectors(detector_a, detector_b)
else:
    compare_detectors(model_a, model_b)

print("End of game, have a nice day !")