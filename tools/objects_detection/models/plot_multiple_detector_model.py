#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a trained model, this script will plot a visual representation of the channels features usage.
The plot should be comparable to figure 3 in Dollar et al. "Integral Channel Features" BMVC 2009,
but here we plot the model, not the specific activation for a single image 
"""

from __future__ import print_function

import detector_model_pb2
import os.path #, sys
from optparse import OptionParser

import numpy as np
#from PIL import Image

import pylab

colors_val =['r','g','b','c', 'y', 'k', 'm', (1,.5,0),(0.5,0.5,0.5)]


#print_the_features = True
print_the_features = False

def create_empty_channels(width, height, num_channels):
    return np.array(num_channels*[height*[np.zeros(width)]])

channels = None

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
    
    
def read_stump(stump, weight):
    
    if print_the_features:
        print("threshold ==", stump.feature_threshold,
              " larger_than_threshold ==", stump.larger_than_threshold,
              end="\t")
              
    feature = stump.feature
    add_feature_to_channels(feature.channel_index, feature.box, weight)    
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

        if print_the_features: 
            print("stage:" , i)
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

def plot_tau_vs_missing_classifiers():
    pylab.figure()
    pylab.clf() # clear the figure
    pylab.gcf().set_facecolor("w") # set white background            
    pylab.grid(True)
    pylab.axis([5000, 20000, 0, 0.2])
    pylab.title('Number of missing weak classifiers for different values of alpha')    
    missingLearners=[19213, 14779, 11745, 9812, 8958, 8307, 7564, 6858, 6673]
    meanMissrate=[0.1170, 0.1208, 0.1225, 0.1261, 0.1270, 0.1197, 0.1459, 0.1329, 0.1358]

    pylab.legend(loc ="upper left", fancybox=True)
    

    for i in range(len(missingLearners)):
        label_name = "alpha=0.0" +str(i)
        print(label_name)
        pylab.plot(missingLearners[i], meanMissrate[i], marker='o', label=label_name,linewidth=0,markerfacecolor=colors_val[i], markersize=8)#, label=label_name,linewidth=3, color=colors_val[i])
    pylab.xlabel("Sum of missing weak learners to reach 2000")
    pylab.ylabel("Average miss rate [0.1..1]")
    pylab.legend(loc='lower left',numpoints=1)

    pylab.draw()
    #pylab.show()

def plot_cascade(cascade, model):
    
    weights = []
    thresholds = []
    for i, stage in enumerate(cascade.stages):
        #print("stage:" , i)
        if stage.feature_type == stage.Level2DecisionTree:
            weights.append(stage.weight)
            thresholds.append(stage.cascade_threshold) 
        elif stage.feature_type == stage.Stumps:
            weights.append(stage.weight)
            thresholds.append(stage.cascade_threshold)       
        else:
            raise Exception("Received an unhandled stage.feature_type")            
    # end of "for each stage"

    for i, stage in enumerate(cascade.stages):
        #print("stage %i cascade threshold:" % i , stage.cascade_threshold)
        #print("stage %i weight:" % i , weights[i])
        pass
    
    if thresholds[0] < -1E5:
        print("The provided model seems not have a soft cascade, " \
              "skipping plot_cascade")
        return

    # create new figure    
    #fig = 
    pylab.figure()
    pylab.clf() # clear the figure
    pylab.gcf().set_facecolor("w") # set white background            
    pylab.grid(True)

    #pylab.spectral() # set the default color map    
        
    # draw the figure
    max_scores = pylab.cumsum(pylab.absolute(weights))
    pylab.plot(max_scores, label="maximum possible score")    
    pylab.plot(thresholds, label="cascade threshold")
    
    pylab.legend(loc ="upper left", fancybox=True)
    pylab.xlabel("Cascade stage")
    pylab.ylabel("Detection score")
  
    title = "Soft cascade"
    if model:
        title = "Soft cascade for model '%s' over '%s' dataset" \
                % (model.detector_name, model.training_dataset_name)
    pylab.title(title)    
    pylab.draw() 

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
    
    print(model.detector_name)
    if model.training_dataset_name:
        print("trained on dataset:", model.training_dataset_name)

    if type(model) is detector_model_pb2.DetectorModel \
       and model.soft_cascade_model:
        #print("Model shrinking factor ==", 
        #      model.soft_cascade_model.shrinking_factor)
        print("Model channels description ==", 
              model.soft_cascade_model.channels_description)

    return model

    
    
def plot_channels(channels, model=None):

    # convert 3d matrix into 2d matrix
    #print("channels.shape ==", channels.shape)
    num_channels, height, width = channels.shape
    channels_2d = channels.reshape(height, width*num_channels).copy()
    
    normalize_per_channel = True
    for c in range(num_channels):
        channel = channels[c, :, :].copy()
        if normalize_per_channel:
            channel -= channel.min()
            channel /= -channel.max() # minus just to make blue and red more logical
            
        channels_2d[:, width*c:width*(c + 1)] = channel
        
    if not normalize_per_channel:
        channels_2d -= channels_2d.min()
        channels_2d /= channels_2d.max() 
    
    # create new figure    
    #fig = 
    pylab.figure()
    pylab.clf() # clear the figure
    pylab.gcf().set_facecolor("w") # set white background            
    #pylab.grid(True)
    
    #colormap = pylab.cm.gray
    colormap = pylab.cm.RdBu
        
    # draw the figure
    pylab.imshow(channels_2d, cmap=colormap, interpolation="nearest")    
    
    title = "Learned model"
    if model:
        title = "Learned model '%s' over '%s' dataset" % (model.detector_name, model.training_dataset_name)
    pylab.title(title)    
    pylab.draw() 
    
    return


    return    
    
def plot_detector_model(model):
    
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
        
    # FIXME hardcoded value
    model_num_channels = 10
    global channels
    channels = create_empty_channels(model_width, model_height, model_num_channels)    
    print("channels.shape ==", channels.shape)
   
    #print("model.detector_type", model.detector_type)
    if model.detector_type == model.SoftCascadeOverIntegralChannels:
        cascade = model.soft_cascade_model    
        print("Model has %i stages" % len(cascade.stages))
        read_cascade(cascade)
        plot_cascade(cascade, model)

    plot_channels(channels, model)
   
    return


def compute_stumps_statistics(model):
    
    if model.model_window_size:
        #model_width = model.model_window_size.x
        model_height = model.model_window_size.y        
    else:
        # we use the INRIAPerson as default
        #model_width = 64 
        model_height = 128 

    shrinking_factor = 4 # best guess
    if model.soft_cascade_model:
        shrinking_factor = model.soft_cascade_model.shrinking_factor
        #print("Model shrinking factor ==", shrinking_factor)
    
    # we take into account the shrinking factor    
    #model_width /= shrinking_factor
    model_height /= shrinking_factor
 
    half_height = model_height * 0.5
    #half_height = model_height * 0.75
    weak_learners_counter = 0    
    stumps_counter = 0
    
    max_intra_tree_height_diff = 0    
    
    if model.detector_type == model.SoftCascadeOverIntegralChannels:
        cascade = model.soft_cascade_model 

        for i, stage in enumerate(cascade.stages):
            if stage.feature_type == stage.Level2DecisionTree:
                tree = stage.level2_decision_tree
                tree_is_ok = True                
                for node in tree.nodes:
                    bb = node.decision_stump.feature.box
                    stump_is_ok = (bb.max_corner.y <= half_height)
                    if stump_is_ok:
                        stumps_counter +=1
                    tree_is_ok = tree_is_ok and stump_is_ok
                if tree_is_ok:
                    weak_learners_counter += 1
                
                bb_ys = [node.decision_stump.feature.box.max_corner.y for node in tree.nodes ]
                max_intra_tree_height_diff = max(abs(bb_ys[0] - bb_ys[1]), 
                                             abs(bb_ys[1] - bb_ys[2]),
                                             abs(bb_ys[0] - bb_ys[2]), max_intra_tree_height_diff)
                
             
            else:
                raise Exception("Received an unhandled stage.feature_type")            
        # end of "for each stage"
        
        print("max_intra_tree_height_diff ==", max_intra_tree_height_diff,
              "[pixels] == %.3f %% of the model height" % (float(max_intra_tree_height_diff)*100 / model_height))
        print("Num weak classifiers accepted when cutting at 50% ==", weak_learners_counter)
        print("Num stumps accepted when cutting at 50% ==", stumps_counter)
        print("Total num weak classifiers ==", len(cascade.stages))
        
    return


def plot_weak_classifiers_versus_height(models, labels):
    global colors_val
    pylab.figure() # create new figure        
    pylab.clf() # clear the figure
    pylab.gcf().set_facecolor("w") # set white background            
    pylab.grid(True)
    #hold(True)
    pylab.xlabel("Height fraction")
    pylab.ylabel("Number of weak learners")
    
    for model_index,model in enumerate(models):
        if model.model_window_size:
            #model_width = model.model_window_size.x
            model_height = model.model_window_size.y        
        else:
            # we use the INRIAPerson as default
            #model_width = 64 
            model_height = 128 

        shrinking_factor = 4 # best guess
        if model.soft_cascade_model:
            shrinking_factor = model.soft_cascade_model.shrinking_factor
            #print("Model shrinking factor ==", shrinking_factor)
        
        # we take into account the shrinking factor    
        #model_width /= shrinking_factor
        model_height /= shrinking_factor
     
        half_height = model_height * 0.5
        #half_height = model_height * 0.75
        weak_learners_counter = 0    
        stumps_counter = 0
        
        max_intra_tree_height_diff = 0    
        
        num_bins = 128
        weak_learner_bins = [0]*num_bins
        stump_bins = [0]*num_bins
        bin_height = model_height / float(num_bins)    
        heights = [ i/float(num_bins) for i in range(num_bins)]
        
        index = 0
        if model.detector_type == model.SoftCascadeOverIntegralChannels:
            cascade = model.soft_cascade_model 

            for i, stage in enumerate(cascade.stages):
                if stage.feature_type == stage.Level2DecisionTree:
                    tree = stage.level2_decision_tree
                    index += 1
                    weak_learner_bin = 0
                    for node in tree.nodes:
                        bb = node.decision_stump.feature.box
                        #stump_bin_index = min(int(bb.max_corner.y/bin_height), num_bins - 1)
                        stump_bin_index = int(bb.max_corner.y/bin_height)
                        if int(bb.max_corner.y/bin_height) >= num_bins:
                                print("node.id ==", node.id)
                                print("node.parent_id ==", node.parent_id)
                                print("bb.max_corner.y ==", bb.max_corner.y)
                                print("weak classifier index ==", index)
                                raise Exception("Invalid bb.max_corner.y")
                        weak_learner_bin = max(weak_learner_bin, stump_bin_index)
                        
                        stump_bins[stump_bin_index] += 1     
                    # end of "for each node in the tree"
                    weak_learner_bins[weak_learner_bin] += 1
            # end of "for each stage"
            
            
            stumps_cumsum = np.cumsum(stump_bins)
            weak_learners_cumsum = np.cumsum(weak_learner_bins)
            print(model_index)
            #print(weak_learners_cumsum)
            weak_learners_missing = weak_learners_cumsum[64:]
            weak_learners_missing = 2000-weak_learners_missing
            #print(weak_learners_missing)
            weak_learners_missing = weak_learners_missing[0:64:4]
            #print(weak_learners_missing)
            print("total weak learners missing: ", sum(weak_learners_missing))
            #pylab.plot(heights, stumps_cumsum, label="stumps cumsum",linewidth=2, color=colors_val[model_index])
            
            pylab.plot(heights, weak_learners_cumsum, label=labels[model_index],linewidth=2, color=colors_val[model_index])
            pylab.legend(loc='upper left')
            pylab.title("Number of weak learners versus height")
            
        else:
            print("plot_weak_classifiers_versus_height received a model of unmanaged type")
            pass
    
    return

def main():

    plot_tau_vs_missing_classifiers()
    #return
    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and plot its content"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       action="append", default=[],
                       help="This option should be repeated for each model, give the model file")
    parser.add_option("-l", "--label", dest="label",
                       type="string",
                       action="append", default=[],
                       help="This option should be repeated for each model, give the label")
    (options, args) = parser.parse_args()
    
    #print (options, args)
    
    if options.input_path:
        for path in options.input_path:
            if not os.path.exists(path):
                parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    model_filename = options.input_path

    models = []
    for path in options.input_path:

        model = read_model(path)
        models.append(model)
    
    plot_weak_classifiers_versus_height(models, options.label)

    pylab.show() # blocking call
    return
        

if __name__ == '__main__':
    main()
