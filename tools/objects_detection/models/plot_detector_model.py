#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a trained model, this script will plot a visual representation of
the channels features usage.
The plot should be comparable to figure 3 in
Dollar et al. "Integral Channel Features" BMVC 2009,
but here we plot the model, not the specific activation for a single image
"""

from __future__ import print_function

import os.path
import sys
filedir = os.path.join(os.path.dirname(__file__))
sys.path.append(os.path.join(filedir, ".."))
sys.path.append(os.path.join(filedir, "..", "..", "helpers"))
import detector_model_pb2
from optparse import OptionParser

import numpy as np
import pylab
from matplotlib.ticker import MultipleLocator
import scipy.stats

from mirror_occluded_model import get_occlusion_level_and_type
#from hinton_diagram import hinton

#print_the_features = True
print_the_features = False

channels = None
min_box_xy = None
max_box_xy = None

channels_count = None
channels_weighted_count = None

level1_feature_centers_per_channel = None

channel_indices = []
level2_to_level1_feature_delta = []
features_area = []
features_ratio = []

# all the boxes, with no distinction on channel
features_box_and_weight = []


def box_center(box):
    """
    Returns the x,y coordinates of the center
    """

    center_x = (box.max_corner.x + box.min_corner.x) / 2.0
    center_y = (box.max_corner.y + box.min_corner.y) / 2.0

    return center_x, center_y


def add_feature_to_channels(channel_index, box, weight):

    if channel_index >= channels.shape[0]:
        increase_model_num_channels(channel_index + 1)

    #for y in range(box.min_corner.y, box.max_corner.y+1):
    #    for x in range(box.min_corner.x, box.max_corner.x+1):
    #        channels[channel_index, y, x] += weight
    slice_y = slice(box.min_corner.y, box.max_corner.y)
    slice_x = slice(box.min_corner.x, box.max_corner.x)
    channels[channel_index, slice_y, slice_x] += weight

    if print_the_features:
        print("channel ==", channel_index,
              "box (min x,y) (max x,y) ==",
              (box.min_corner.x, box.min_corner.y),
              (box.max_corner.x, box.max_corner.y),
              "\tweight ==", weight)

    # update stored statistics --
    min_box_xy[0] = min(min_box_xy[0], box.min_corner.x)
    min_box_xy[1] = min(min_box_xy[1], box.min_corner.y)
    max_box_xy[0] = max(max_box_xy[0], box.max_corner.x)
    max_box_xy[1] = max(max_box_xy[1], box.max_corner.y)

    channel_indices.append(channel_index)
    channels_count[channel_index] += 1
    channels_weighted_count[channel_index] += abs(weight)

    feature_width = box.max_corner.x - box.min_corner.x
    feature_height = box.max_corner.y - box.min_corner.y
    area = feature_width * feature_height
    features_area.append(area)

    ratio = float(feature_width) / feature_height
    features_ratio.append(ratio)

    center_x, center_y = box_center(box)

    level1_feature_centers_per_channel[channel_index].append(
        (center_x, center_y))

    features_box_and_weight.append((box, weight))
    return


def read_stump(stump, weight):

    if print_the_features:
        print("threshold ==", stump.feature_threshold,
              " larger_than_threshold ==", stump.larger_than_threshold,
              end="\t")

    feature = stump.feature
    add_feature_to_channels(feature.channel_index, feature.box, weight)
    return


def read_stump_set(stump_set, weight):

    for stump in stump_set.nodes:
        read_stump(stump, weight)

    i = 0
    for weight in stump_set.weights:
        print("weight %i= %.15f" % (i, weight))
        i = i + 1
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
            #break

        if print_the_features:
            print("stage:", i)
        if stage.feature_type == stage.Level2DecisionTree:
            read_tree(stage.level2_decision_tree, stage.weight)
        elif stage.feature_type == stage.Stumps:
            read_stump(stage.decision_stump, stage.weight)
        elif stage.feature_type == stage.StumpSet:
            read_stump_set(stage.stump_set, stage.weight)
        else:
            print("stage.feature_type ==", stage.feature_type)
            raise Exception("Received an unhandled stage.feature_type")
        #print("weight:", stage.weight)
        if False and stage.cascade_threshold > -1E5:
            # we only print "non trivial" values
            print("stage %i cascade threshold:" % i, stage.cascade_threshold)

    return


def create_new_figure():
    fig = pylab.figure()
    fig.clf()  # clear the figure
    fig.set_facecolor("w")  # set white background
    fig.gca().grid(True)
    return fig


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
        elif stage.feature_type == stage.StumpSet:
            weights.append(stage.weight)
            thresholds.append(stage.cascade_threshold)
        else:
            raise Exception("Received an unhandled stage.feature_type")
    # end of "for each stage"

    for i, stage in enumerate(cascade.stages):
        #print("stage %i cascade threshold:" % i , stage.cascade_threshold)
        #print("stage %i weight:" % i , weights[i])
        pass

    if False:
        if thresholds[0] < -1E5:
            print("The provided model seems not have a soft cascade, "
                   "skipping plot_cascade")
            return
    else:
         if thresholds[0] < -1E5:
            print("The provided model seems not have a soft cascade")
    

    create_new_figure()

    #pylab.spectral()  # set the default color map

    # draw the figure
    max_scores = pylab.cumsum(pylab.absolute(weights))
    pylab.subplot(2, 1, 1)
    pylab.plot(max_scores, label="maximum possible score")
    pylab.legend(loc="upper left", fancybox=True)
    pylab.xlabel("Cascade stage")
    pylab.ylabel("Detection score")

    pylab.subplot(2, 1, 2)
    pylab.plot(weights, label="weak classifier weights")
    pylab.plot(thresholds, label="cascade threshold")
    pylab.legend(loc="upper left", fancybox=True)
    pylab.xlabel("Cascade stage")
    pylab.ylabel("Detection score")

    title = "Soft cascade"
    if model:
        title = "Soft cascade for model '%s' over '%s' dataset" \
                % (model.detector_name, model.training_dataset_name)
    pylab.suptitle(title)
    #pylab.draw()

    pylab.tight_layout()
    return


def read_model(model_filename):

    model = detector_model_pb2.DetectorModel()
    f = open(model_filename, "rb")
    model.ParseFromString(f.read())
    f.close()

    if not model.IsInitialized():
        print("Input file seems not to be a DetectorModel, "
              "trying as MultiScalesDetectorModel")

        model = detector_model_pb2.MultiScalesDetectorModel()
        f = open(model_filename, "rb")
        model.ParseFromString(f.read())
        f.close()

    if not model.IsInitialized():
        print("Input file seems not to be "
              "a DetectorModel nor a  MultiScalesDetectorModel")
        raise Exception("Unknown input file format")

    print(model.detector_name)
    if model.training_dataset_name:
        print("trained on dataset:", model.training_dataset_name)

    if (type(model) is detector_model_pb2.DetectorModel) \
       and model.soft_cascade_model:
        print("Model shrinking factor ==",
              model.soft_cascade_model.shrinking_factor)
        #print("Model channels description ==",
        #      model.soft_cascade_model.channels_description)

    try:
        occlusion_level, occlusion_type = \
            get_occlusion_level_and_type(model, model_filename)
        if occlusion_level > 0:
            print("Model occlusion_level, occlusion_type == ",
                  (occlusion_level, occlusion_type))
    except Exception, e:
        print("Error when retrieving the occlusion type:", e)
        pass  # nothing to do

    return model


def plot_channels(channels, model=None,
                  input_channels_figure=None, input_model_figure=None):

    # convert 3d matrix into 2d matrix
    #print("channels.shape ==", channels.shape)
    num_channels, height, width = channels.shape
    channels_2d = channels.reshape(height, width * num_channels).copy()

    normalize_per_channel = True
    for c in range(num_channels):
        channel = channels[c, :, :].copy()
        if normalize_per_channel:
            channel -= channel.min()
            if channel.max() != 0:
                channel /= channel.max()

        channels_2d[:, (width * c):(width * (c + 1))] = channel

    if not normalize_per_channel:
        channels_2d -= channels_2d.min()
        channels_2d /= channels_2d.max()

    if not input_channels_figure:
        channels_figure = create_new_figure()
        channels_axes = channels_figure.gca()
    else:
        channels_figure = input_channels_figure
        channels_axes = channels_figure

    channels_axes.grid(False)

    # _r stands of "reversed"
    #colormap = pylab.cm.gray_r
    #colormap = pylab.cm.RdBu_r
    colormap = pylab.cm.RdYlBu_r

    # draw the figure
    channels_axes.imshow(channels_2d,
                         extent=(0, width*num_channels, height, 0),
                         cmap=colormap, interpolation="nearest")

    channels_axes.set_xticks(range(0, width*num_channels+1, width))

    if num_channels == 10:
        ticks = pylab.arange((width / 2), width*num_channels+1, width)
        ticks_labels = ("$90^{\\circ}$", "$60^{\\circ}$", "$30^{\\circ}$",
                        "$0^{\\circ}$", "$-30^{\\circ}$", "$-60^{\\circ}$",
                        #"$\\left\\Vert \\cdot \\right\\Vert$",
                        "$||\\cdot||$",
                        "L", "U", "V")

        #channels_axes.set_xticks(ticks, ticks_labels)
        #pylab.xticks(ticks, ticks_labels)
        channels_axes.set_xticks(range(0, width*num_channels+1, width))
        channels_axes.set_xticklabels(ticks_labels)
    else:
        channels_axes.set_xticks(range(0, width*num_channels+1, width))


    channels_axes.set_yticks(range(0, height+1, width))
    channels_axes.xaxis.set_minor_locator(MultipleLocator(min(8, width/4)))
    channels_axes.yaxis.set_minor_locator(MultipleLocator(8))
    channels_axes.xaxis.grid(color="white", linewidth=1.5, linestyle="--")
    channels_axes.set_aspect("equal")

    model_octave = 0
    title = "Learned model"
    if model:
        if model.training_dataset_name == "WindowDetection":
            model_octave = 0
        elif model.training_dataset_name == "TrafficSignDetection":
            model_octave = 0
        elif model.training_dataset_name == "INRIA":
            model_octave = 0
        else:
            try:
                model_octave = \
                    float(model.training_dataset_name.split("_")[-1])
            except:
                model_octave = 0  # if nothing works we assume octave 0

        title = "Learned model '%s' over '%s' dataset\n" \
                % (model.detector_name, model.training_dataset_name)
    # end of "if model"

    model_scale = 2**model_octave

    #pylab.xlabel("$x$ axis in pixels")
    #pylab.ylabel("$y$ axis in pixels")

    if not input_channels_figure:
        pylab.gca().yaxis.set_label_position("right")
        pylab.ylabel("Scale %.0f" % model_scale,
                     fontsize=20, labelpad=20, rotation=-90)

        pylab.title(title)
        pylab.gcf().canvas.set_window_title(
            "Octave %.1f channels" % model_octave)

    plot_all_in_one = True
    if plot_all_in_one:
        all_in_one = np.zeros(channels[0, :, :].shape)
        for c in range(channels.shape[0]):
            all_in_one += channels[c, :, :]  # minus to have the desired colors

        if not input_model_figure:
            model_figure = create_new_figure()
            model_axes = model_figure.gca()
        else:
            model_figure = input_model_figure
            model_axes = model_figure

        model_axes.grid(False)
        model_axes.imshow(all_in_one,
                          extent=(0, width, height, 0),
                         cmap=colormap, interpolation="nearest")
        model_axes.set_xticks(range(0, width+1, width/2))
        model_axes.set_yticks(range(0, height+1, width))
        model_axes.xaxis.set_minor_locator(MultipleLocator(min(8, width/4)))
        model_axes.yaxis.set_minor_locator(MultipleLocator(8))
        model_axes.xaxis.grid(False)
        model_axes.set_aspect("equal")

        pylab.xlabel("$x$ axis in pixels")
        pylab.ylabel("$y$ axis in pixels")

        #model_axes.yaxis.set_label_position("right")
        #twin_x = model_axes.twinx()
        #twin_x.yaxis.set_ticklabels([])
        #twin_x.set_aspect("equal")
        if not input_model_figure:
            model_axes.set_ylabel("Scale %.0f" % model_scale,
                  fontsize=45,
                  labelpad=20,
                  #rotation=-90
                )
            pylab.title(title)
            pylab.gcf().canvas.set_window_title("Octave %.1f" % model_octave)
        else:
            model_axes.set_ylabel("Scale %.0f" % model_scale,
                  fontsize=25,
                  labelpad=20,
                  #rotation=-90
                )
    # end of "plot all in one"

    pylab.draw()
    return


def occlusion_type_name(occlusion_type):

    m = detector_model_pb2.DetectorModel
    names = {
        m.LeftOcclusion: "left",
        m.RightOcclusion: "right",
        m.TopOcclusion: "top",
        m.BottomOcclusion: "bottom",
    }

    return names[occlusion_type]


def increase_model_num_channels(new_num_channels):

    global channels, channels_count, channels_weighted_count, \
        level1_feature_centers_per_channel

    assert channels.shape[0] < new_num_channels, \
        "Called increase_model_num_channels with " \
        "less num_channels than the already existing ones"

    additional_channels = new_num_channels - channels.shape[0]
    channels_count.extend([0]*additional_channels)
    channels_weighted_count.extend([0]*additional_channels)
    level1_feature_centers_per_channel.extend([[]]*additional_channels)

    new_empty_channels_shape = (additional_channels,
                                channels.shape[1], channels.shape[2])
    channels = np.append(channels,
                         np.zeros(new_empty_channels_shape), axis=0)
    return


def plot_channels_cooccurrence(cascade):

    model_num_channels = 0
    # square matrix matrix that counts the co-occurrences
    channels_cooccurrence = np.zeros((model_num_channels, model_num_channels))

    for i, stage in enumerate(cascade.stages):
        if stage.feature_type != stage.Level2DecisionTree:
            raise Exception("plot_channels_cooccurrence "
                            "only accepts level-2 trees")
        # stage.feature_type == stage.Level2DecisionTree

        tree = stage.level2_decision_tree
        root_channel = 0
        children_channels = []
        max_channel = 0
        for node in tree.nodes:
            channel = node.decision_stump.feature.channel_index
            max_channel = max(max_channel, channel)
            if node.id == node.parent_id:
                root_channel = channel
            else:
                children_channels.append(channel)
        # end  of "for all three nodes"

        if max_channel >= channels_cooccurrence.shape[0]:
            old_max_channel = channels_cooccurrence.shape[0]
            new_cooccurence = np.zeros((max_channel + 1, max_channel + 1))
            new_cooccurence[:old_max_channel, :old_max_channel] = \
                channels_cooccurrence  # copy current counts
            channels_cooccurrence = new_cooccurence  # replace


        for c in children_channels:
            channels_cooccurrence[root_channel, c] += 1
            channels_cooccurrence[c, root_channel] += 1

    # end of "for all cascade stages"

    if channels_cooccurrence.shape[0] == 43:
        # Special case for semantic context extension of base model
        channels_cooccurrence[10:16, 10:16] = channels_cooccurrence[-6:, -6:]
        channels_cooccurrence[10:16, :] = channels_cooccurrence[-6:, :]
        channels_cooccurrence[:, 10:16] = channels_cooccurrence[:, -6:]
        channels_cooccurrence = channels_cooccurrence[:16, :16]

        #channels_cooccurrence = pylab.log(channels_cooccurrence)
        channels_cooccurrence = pylab.sqrt(channels_cooccurrence)
        print("Showing sqrt of co-occurrences")


    create_new_figure()
    #hinton(channels_cooccurrence)

    pylab.pcolor(channels_cooccurrence, vmin=0)
    #pylab.imshow(channels_cooccurrence, vmin=0, origin="upper",
    #                interpolation="nearest")
    pylab.gca().invert_yaxis()

    ticks_labels = None
    if channels_cooccurrence.shape[0] == 10:
        ticks_labels = (u"|", u"╱", u"/", u"-", u"\\", u"╲", u"M",
                        u"L", u"U", u"V")
        ticks_labels = ("$90^{\\circ}$", "$60^{\\circ}$", "$30^{\\circ}$",
                        "$0^{\\circ}$", "$-30^{\\circ}$", "$-60^{\\circ}$",
                        #"$\\left\\Vert \\cdot \\right\\Vert$",
                        "$||\\cdot||$",
                        "L", "U", "V")
    elif channels_cooccurrence.shape[0] == 16:
        ticks_labels = ("$90^{\\circ}$", "$60^{\\circ}$", "$30^{\\circ}$",
                        "$0^{\\circ}$", "$-30^{\\circ}$", "$-60^{\\circ}$",
                        #"$\\left\\Vert \\cdot \\right\\Vert$",
                        "$||\\cdot||$",
                        "L", "U", "V",
                        "Ver.", "Hor.", "Veg.", "Sky", "Car", "Per.")
    else:
        pass

    if ticks_labels:
        ticks = pylab.arange(channels_cooccurrence.shape[0]) + 0.5
        pylab.xticks(ticks, ticks_labels)
        pylab.yticks(ticks, ticks_labels)

    #pylab.imshow(channels_cooccurrence)
    pylab.colorbar()
    pylab.title("Channels co-occurrence")
    pylab.gcf().canvas.set_window_title("Channels co-occurrence")
    pylab.tight_layout()

    return



def plot_pairwise_offsets(cascade):

    # for each pair of features, stores the offset between the features center
    pairwise_offsets = []

    for i, stage in enumerate(cascade.stages):
        if stage.feature_type != stage.Level2DecisionTree:
            raise Exception("plot_pairwise_offsets only accepts level-2 trees")
        # stage.feature_type == stage.Level2DecisionTree

        tree = stage.level2_decision_tree
        root_center = (0, 0)
        children_centers = []
        for node in tree.nodes:
            center = box_center(node.decision_stump.feature.box)
            if node.id == node.parent_id:
                root_center = center
            else:
                children_centers.append(center)
        # end  of "for all three nodes"

        for center in children_centers:
            pairwise_offsets.append((root_center, center))
    # end of "for all cascade stages"

    create_new_figure()

    pylab.subplot(1, 2, 1)
    for pair in pairwise_offsets:
        a_xy, b_xy = pair
        pylab.plot((a_xy[0], b_xy[0]), (a_xy[1], b_xy[1]),
                   color="black", alpha=0.05)


    pylab.gca().invert_yaxis()
    pylab.xlabel("Pixel $x$ coordinate")
    pylab.ylabel("Pixel $y$ coordinate")
    pylab.title("Feature pairs, start and end")


    pylab.subplot(1, 2, 2)

    #count_size = 41
    #count_size = (36 * 2) + 1
    count_size = (36 * 2 * 1.5) + 1
    count_size_half = count_size / 2
    count = np.zeros((count_size, count_size))
    delta_x, delta_y = [], []
    for pair in pairwise_offsets:
        a_xy, b_xy = pair
        d_x, d_y = a_xy[0] - b_xy[0], a_xy[1] - b_xy[1]
        delta_x.append(d_x)
        delta_y.append(-d_y)
        count[d_y + count_size_half, d_x + count_size_half] += 1

    if False:
        pylab.plot(delta_x, delta_y, "ko", markersize=5, alpha=0.05)
    else:
        pylab.pcolor(count, vmin=0)
        pylab.axis([0, count_size, 0, count_size])
        ticks = pylab.arange(5, count_size, 10)
        pylab.xticks(ticks, ticks - count_size_half)
        pylab.yticks(ticks, ticks - count_size_half)
        pylab.colorbar()

    pylab.xlabel("$\Delta x$ offset in pixels")
    pylab.ylabel("$\Delta y$ offset in pixels")
    pylab.title("Feature pairs offset")

    pylab.suptitle("Feature pairs")
    pylab.gcf().canvas.set_window_title("Feature pairs")
    pylab.tight_layout()
    return


def plot_detector_model(model, model_index=0, num_models=1):

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

    shrinking_factor = 4  # best guess
    if model.soft_cascade_model:
        shrinking_factor = model.soft_cascade_model.shrinking_factor
        print("Model shrinking factor ==", shrinking_factor)

    # we take into account the shrinking factor
    model_width /= shrinking_factor
    model_height /= shrinking_factor

    # We start by creating a 1 channel figure,
    # and extend it one demand
    model_num_channels = 1

    global channels, min_box_xy, max_box_xy, \
        channels_count, channels_weighted_count, \
        level1_feature_centers_per_channel

    min_box_xy = [float("inf"), float("inf")]
    max_box_xy = [0, 0]

    channels_count = [0] * model_num_channels
    channels_weighted_count = [0] * model_num_channels
    level1_feature_centers_per_channel = [[]] * model_num_channels

    # create empty channels
    channels = np.zeros((model_num_channels, model_height, model_width))
    print("channels.shape ==", channels.shape)

    #print("model.detector_type", model.detector_type)
    if model.detector_type == model.SoftCascadeOverIntegralChannels:
        cascade = model.soft_cascade_model
        print("Model has %i stages" % len(cascade.stages))
        read_cascade(cascade)
        plot_cascade(cascade, model)

    if model.HasField("occlusion_type"):
        if model.occlusion_level > 0:
            print("Model occlusion type '%s' and occlusion level %.3f" % (
                occlusion_type_name(model.occlusion_type),
                model.occlusion_level))

    print("Model min/max box xy == %s/%s" % (min_box_xy, max_box_xy))

    plot_cooccurrence = False
    if plot_cooccurrence:
        plot_channels_cooccurrence(cascade)

    plot_the_pairwise_offsets = False
    if plot_the_pairwise_offsets:
        plot_pairwise_offsets(cascade)

    force_separate_plots = False
    if num_models <= 1 or force_separate_plots:
        plot_channels(channels, model)
    else:
        if model_index == 0:
            create_new_figure()
        channels_figure = pylab.subplot(num_models, 2, 2*model_index + 0 + 1)
        model_figure = pylab.subplot(num_models, 2, 2*model_index + 1 + 1)
        plot_channels(channels, model,
                      input_channels_figure=channels_figure,
                      input_model_figure=model_figure)

        if model_index == (num_models - 1):
            channels_figure.set_xlabel("(Axes units in pixels)",
                                       fontsize="medium", labelpad=20)

    # end of "plot all channels in one figure"
    return


def compute_stumps_statistics(model):

    if model.model_window_size:
        #model_width = model.model_window_size.x
        model_height = model.model_window_size.y
    else:
        # we use the INRIAPerson as default
        #model_width = 64
        model_height = 128

    shrinking_factor = 4  # best guess
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
                        stumps_counter += 1
                    tree_is_ok = tree_is_ok and stump_is_ok
                if tree_is_ok:
                    weak_learners_counter += 1

                bb_ys = [node.decision_stump.feature.box.max_corner.y
                         for node in tree.nodes]
                max_intra_tree_height_diff = \
                    max(abs(bb_ys[0] - bb_ys[1]),
                        abs(bb_ys[1] - bb_ys[2]),
                        abs(bb_ys[0] - bb_ys[2]),
                        max_intra_tree_height_diff)
            elif stage.feature_type == stage.Stumps:
                bb = stage.decision_stump.feature.box
                stump_is_ok = (bb.max_corner.y <= half_height)
                if stump_is_ok:
                    stumps_counter += 1
                    weak_learners_counter += 1
                bb_ys = [stage.decision_stump.feature.box.max_corner.y]
                max_intra_tree_height_diff = 0
            else:
                raise Exception("Received an unhandled stage.feature_type")
        # end of "for each stage"

        print("max_intra_tree_height_diff ==", max_intra_tree_height_diff,
              "[pixels] == %.3f %% of the model height" % (
              (float(max_intra_tree_height_diff) * 100) / model_height))
        print("Num weak classifiers accepted when cutting at 50% ==",
              weak_learners_counter)
        print("Num stumps accepted when cutting at 50% ==", stumps_counter)
        print("Total num weak classifiers ==", len(cascade.stages))

    return


def plot_weak_classifiers_versus_width(model):

    if model.model_window_size:
        model_width = model.model_window_size.x
    else:
        # we use the INRIAPerson as default
        model_height = 128

    shrinking_factor = 4  # best guess
    if model.soft_cascade_model:
        shrinking_factor = model.soft_cascade_model.shrinking_factor
        #print("Model shrinking factor ==", shrinking_factor)

    # we take into account the shrinking factor
    model_width /= shrinking_factor

    half_width = model_width * 0.5
    weak_learners_counter = 0
    stumps_counter = 0

    max_intra_tree_height_diff = 0

    num_bins = 100
    weak_learner_bins = [0] * num_bins
    stump_bins = [0] * num_bins
    bin_width= model_width / float(num_bins)
    widths = [i / float(num_bins) for i in range(num_bins)]

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
                    stump_bin_index = int((bb.max_corner.x - 1) / bin_width)
                    if stump_bin_index >= num_bins:
                        print("node.id ==", node.id)
                        print("node.parent_id ==", node.parent_id)
                        print("bb.max_corner.x ==", bb.max_corner.x)
                        print("bin_width ==", bin_width)
                        print("num_bins ==", num_bins)
                        print("weak classifier index ==", index)
                        raise Exception("Invalid bb.max_corner.x")
                    weak_learner_bin = max(weak_learner_bin, stump_bin_index)

                    stump_bins[stump_bin_index] += 1
                # end of "for each node in the tree"
                weak_learner_bins[weak_learner_bin] += 1
        # end of "for each stage"

        create_new_figure()

        stumps_cumsum = np.cumsum(stump_bins)
        weak_learners_cumsum = np.cumsum(weak_learner_bins)
        pylab.plot(widths, stumps_cumsum, label="stumps cumsum")
        pylab.plot(widths, weak_learners_cumsum, label="weak learners cumsum")
        pylab.xlabel("Width fraction")
        pylab.ylabel("Number of elements")
        pylab.legend()
        pylab.title("Number of elements versus width")

    else:
        print("plot_weak_classifiers_versus_height "
              "received a model of unmanaged type")
        pass

    return


def plot_weak_classifiers_versus_height(model):

    if model.model_window_size:
        #model_width = model.model_window_size.x
        model_height = model.model_window_size.y
    else:
        # we use the INRIAPerson as default
        #model_width = 64
        model_height = 128

    shrinking_factor = 4  # best guess
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

    num_bins = 100
    weak_learner_bins = [0]*num_bins
    stump_bins = [0] * num_bins
    bin_height = model_height / float(num_bins)
    heights = [i / float(num_bins) for i in range(num_bins)]

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
                    stump_bin_index = int((bb.max_corner.y - 1) / bin_height)
                    if stump_bin_index >= num_bins:
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

        create_new_figure()

        stumps_cumsum = np.cumsum(stump_bins)
        weak_learners_cumsum = np.cumsum(weak_learner_bins)
        pylab.plot(heights, stumps_cumsum, label="stumps cumsum")
        pylab.plot(heights, weak_learners_cumsum, label="weak learners cumsum")
        pylab.xlabel("Height fraction")
        pylab.ylabel("Number of elements")
        pylab.legend()
        pylab.title("Number of elements versus height")

    else:
        print("plot_weak_classifiers_versus_height "
              "received a model of unmanaged type")
        pass

    return


def plot_features_statistics():

    num_channels = len(channels_count)
    _, model_height, model_width = channels.shape

    tango_sky_blue_1 = (114 / 255.0, 159 / 255.0, 207 / 255.0, 1)
    tango_sky_blue_2 = (52 / 255.0, 101 / 255.0, 164 / 255.0, 1)

    tango_plum_1 = (173 / 255.0, 127 / 255.0, 168 / 255.0, 1)
    tango_plum_2 = (117 / 255.0, 80 / 255.0, 123 / 255.0, 1)

    tango_chameleon_1 = (138 / 255.0, 226 / 255.0, 52 / 255.0, 1)
    tango_chameleon_2 = (115 / 255.0, 210 / 255.0, 22 / 255.0, 1)

    tango_scarlet_red_1 = (239 / 255.0, 41 / 255.0, 41 / 255.0, 1)
    tango_scarlet_red_2 = (204 / 255.0, 0, 0, 1)
    tango_scarlet_red_2_alpha_0_dot_05 = (204 / 255.0, 0, 0, 0.05)
    tango_scarlet_red_2_alpha_0_dot_1 = (204 / 255.0, 0, 0, 0.1)
    tango_scarlet_red_2_alpha_0_dot_3 = (204 / 255.0, 0, 0, 0.3)
    tango_scarlet_red_3 = (164 / 255.0, 0, 0, 1)

    create_new_figure()
    if False:
        pylab.bar(range(num_channels), channels_count)
        pylab.title("Channels count")
    else:
        hist, _ = np.histogram(channel_indices, bins=num_channels, normed=True)
        d = 0.2
        #pylab.hist(channel_indices, bins=num_channels,
                   #normed=True,
        #pylab.bar(range(num_channels), hist,
        pylab.bar(np.arange(num_channels) - ((1 - d) / 2), hist,
                  width=1 - d,
                  color=tango_sky_blue_1, edgecolor=tango_sky_blue_2)
        pylab.gca().set_xticks(range(num_channels))
        pylab.xlim(-0.5 - d, num_channels - 0.5 + d)

        if num_channels == 10:
            ticks_labels = ("$90^{\\circ}$", "$60^{\\circ}$", "$30^{\\circ}$",
                            "$0^{\\circ}$", "$-30^{\\circ}$", "$-60^{\\circ}$",
                            #"$\\left\\Vert \\cdot \\right\\Vert$",
                            "$||\\cdot||$",
                            "L", "U", "V")
            pylab.gca().set_xticklabels(ticks_labels)
        elif num_channels == 43:
            ticks_labels = ["$90^{\\circ}$", "$60^{\\circ}$", "$30^{\\circ}$",
                            "$0^{\\circ}$", "$-30^{\\circ}$", "$-60^{\\circ}$",
                            #"$\\left\\Vert \\cdot \\right\\Vert$",
                            "$||\\cdot||$",
                            "L", "U", "V"]
            ticks_labels.extend(range(10, 37))
            ticks_labels.extend(["vert.", #vertical structure,
                                 "horiz.", #horizontal surface,
                                 "veg.",
                                 "sky",
                                 "car",  #"vehicle",
                                 "person"])
            assert len(ticks_labels) == num_channels, \
                "num_labels %i is not equal to num_channels %i" \
                % (len(ticks_labels), num_channels)
            pylab.gca().set_xticklabels(ticks_labels)
        else:
            pylab.gca().set_xticklabels(range(num_channels))

        pylab.xlabel("Channel index")
        pylab.ylabel("Fraction of features")
        pylab.title("Fraction of features per channel")
        pylab.gcf().canvas.set_window_title("Features per channel")

    pylab.tight_layout()

    if False:
        # -=-=-=-=-=-=-=-=-=-=-=-=-
        create_new_figure()
        pylab.bar(range(num_channels), channels_weighted_count)
        pylab.title("Channels weighted count")
        pylab.tight_layout()

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    create_new_figure()
    if False:
        num_bins = 30
        pylab.hist(features_area, num_bins)
        pylab.title("Features area")

    if False:
        features_area2 = np.array(features_area)
        features_area2 = [float(i) for i in features_area]

        if False:
            fit_alpha,fit_loc,fit_beta = scipy.stats.gamma.fit(features_area)
            print ("gamma parameters: ", (fit_alpha,fit_loc,fit_beta))
            rv = scipy.stats.gamma(fit_alpha,fit_beta)
            x = np.linspace(0, np.minimum(rv.dist.b, num_bins), num_bins)
            h = pylab.plot(x, rv.pdf(x), 'r')

        x = np.linspace(0, num_bins, num_bins)
        scipy.stats.gaussian_kde.covariance_factor= lambda x: 0.05
        kernel = scipy.stats.gaussian_kde(features_area2)
        y = kernel.evaluate(x)
        print("First estimated elements of the histogram: ", y[:30])
        h = pylab.plot(x, y, 'r')
    else:
        areas = np.array(features_area)
        min_max_area = areas.min(), areas.max()
        num_bins = min_max_area[1] 
        hist, bins = pylab.histogram(features_area, num_bins, [1, num_bins], normed=1)
        cumsum = pylab.cumsum(hist)
        #bins_x_value = [((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)]
        bins_x_value = [bins[i] for i in range(len(bins) - 1)]
        pylab.plot(bins_x_value, cumsum, color=tango_plum_2, linewidth=4)

        if (min_max_area[1] - min_max_area[0]) == 0:
            xticks_step = min_max_area[1] / 10.0
        else:
            xticks_step = (min_max_area[1] - min_max_area[0]) / 10.0

        xticks_step = int(pylab.ceil(xticks_step / 10.0) * 10)
        xticks = [ min_max_area[0] ] \
                + range(0, min_max_area[1], xticks_step) + [min_max_area[1]]

        pylab.gca().set_xticks(xticks)
        #pylab.gca().set_xticklabels(range(num_channels))
        if (min_max_area[1] - min_max_area[0]) == 0:
            pylab.xlim(0, min_max_area[1])
        else:
            pylab.xlim(min_max_area)
        pylab.ylim(0, 1.1)
        pylab.xlabel("Feature area in pixels$^2$")
        pylab.ylabel("Fraction of features")
        pylab.title("Fraction of features below a given area value")
        pylab.gcf().canvas.set_window_title("Features per area")

    # end of if, elif, else chain

    pylab.tight_layout()
    # -=-=-=-=-=-=-=-=-=-=-=-=-
    create_new_figure()

    print("Max feature ratio %.3f (widest), min ratio %.3f (longest)" % (
          max(features_ratio), min(features_ratio)))

    if False:
        num_bins = 30
        pylab.hist(features_ratio, num_bins)
        pylab.title("Features ratio")

    elif False:
            features_ratio_separated = []
            for ratio in features_ratio:
                if ratio < 1:
                    features_ratio_separated.append(-1/ratio)
                elif ratio == 1:
                    features_ratio_separated.append(0)
                else:
                    features_ratio_separated.append(ratio)

            num_bins = 300
            pylab.hist(features_ratio_separated, num_bins)
            pylab.title("Features ratio")

            print("Max feature ratio %.3f (widest), min ratio %.3f (longest)"
                % (max(features_ratio), min(features_ratio)))

    elif False:
        signed_ratios = [box_signed_ratio(fbw[0])
                         for fbw in features_box_and_weight]
        max_unsigned_ratio = np.abs(signed_ratios).max()
        #num_bins = 2*max_unsigned_ratio
        num_bins = 1.5 * max_unsigned_ratio
        pylab.hist(signed_ratios, num_bins)
        pylab.title("Features ratio")

    else:
        signed_ratios = [box_signed_ratio(fbw[0])
                         for fbw in features_box_and_weight]
        positive_ratios = [x for x in signed_ratios if x > 0]
        negative_ratios = [-x for x in signed_ratios if x < 0]
        num_squares = len([x for x in signed_ratios if x == 0])
        num_ratios = float(len(signed_ratios))

        if positive_ratios:
            max_pos_ratio = max(positive_ratios)
        else:
            max_pos_ratio = 0.01
            positive_ratios = [0.01, 0.02]

        if negative_ratios:
            max_neg_ratio = max(negative_ratios)
        else:
            max_neg_ratio = 0.01
            negative_ratios = [0.01, 0.02]

        num_bins = 1000
        pos_hist, pos_bins = pylab.histogram(positive_ratios,
                                             bins=num_bins)
        pos_cumsum = pylab.cumsum(pos_hist) / num_ratios

        neg_hist, neg_bins = pylab.histogram(negative_ratios,
                                             bins=num_bins)
        neg_cumsum = pylab.cumsum(neg_hist) / num_ratios

        pylab.plot(pos_bins[:len(pos_cumsum)], pos_cumsum,
                   label="wide features",
                   color=tango_sky_blue_2, linewidth=4)
        pylab.plot(neg_bins[:len(neg_cumsum)], neg_cumsum,
                   label="long features",
                   color=tango_plum_2, linewidth=4)

        squares_x = [1, max(max_pos_ratio, max_neg_ratio)]
        squares_y = [num_squares / num_ratios, num_squares / num_ratios]
        pylab.plot(squares_x, squares_y,
                   label="square features",
                   linestyle="--", color=tango_chameleon_2, linewidth=4)

        xticks_step = squares_x[-1] / 10.0
        xticks_step = int(pylab.ceil(xticks_step / 5.0) * 5)
        xticks = range(0, int(squares_x[-1]), xticks_step) \
            + [squares_x[0], max_pos_ratio, max_neg_ratio]
        if xticks.count(15):
            xticks.remove(15)
        if xticks.count(30):
            xticks.remove(30)
        pylab.gca().set_xticks(xticks)

        pylab.xlim(squares_x)
        pylab.ylim(0, 1.1*max(pos_cumsum[-1], neg_cumsum[-1], squares_y[-1]))

        #pylab.legend(loc="best", fontsize="x-large")
        pylab.legend(loc="best")
        pylab.xlabel("Feature aspect ratio")
        pylab.ylabel("Fraction of features")
        pylab.title("Fraction of features below a given ratio value")
        pylab.gcf().canvas.set_window_title("Feature per ratio")

    # end of if, elif, else chain
    pylab.tight_layout()

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    create_new_figure()

    if False:
        level1_feature_centers = []
        for c in level1_feature_centers_per_channel:
            level1_feature_centers.extend(c)

        x_points = [x[0] for x in level1_feature_centers]
        y_points = [x[1] for x in level1_feature_centers]
        pylab.scatter(x_points, y_points)
        pylab.xlabel("X axis")
        pylab.ylabel("Y axis")
        pylab.title("Level 1 feature centers (all channels together)")

        #level2_to_level1_feature_delta = []
    else:

        x_points, y_points = [], []
        for x, y in [box_center(fb[0]) for fb in features_box_and_weight]:
                x_points.append(x)
                y_points.append(y)
        # end of "for each feature center"

        pylab.scatter(x_points, y_points,
                      s=200, marker="o",
                      linewidths=0.5,
                      edgecolors=tango_scarlet_red_2_alpha_0_dot_1,
                      facecolors=tango_scarlet_red_2_alpha_0_dot_05)

        pylab.xticks(range(0, model_width+1, 8))
        pylab.yticks(range(0, model_height+1, 8))
        d = 1.5
        pylab.xlim(0-d, model_width + d)
        pylab.ylim(model_height + d, 0-d)
        pylab.gca().set_aspect("equal")
        pylab.xlabel("$x$ axis in pixels")
        pylab.ylabel("$y$ axis in pixels")
        pylab.title("Features centres")
        pylab.gcf().canvas.set_window_title("Features centres")

    pylab.tight_layout()
    # -=-=-=-=-=-=-=-=-=-=-=-=-

    plot_top_features_by_weight()

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    plot_boxes_by_area_and_ratio()

    return

def plot_top_features_by_weight():
    create_new_figure()

    sorted_features_box_and_weight = list(features_box_and_weight)

    def compare_weights(x,y):
        if x[1] < y[1]:
            return 1
        elif x[1] > y[1]:
            return -1
        else: # x[1] == y[1]
            return 0
    sorted_features_box_and_weight.sort(cmp=compare_weights)

    print("top 5 features boxes and weights",
          sorted_features_box_and_weight[:5])

    #num_top_features = 18
    #num_top_features = 6
    num_top_features = 16
    top_features_boxes = [x[0]
        for x in sorted_features_box_and_weight[:num_top_features]]
    top_features_weights = [x[1]
        for x in sorted_features_box_and_weight[:num_top_features]]

    all_in_one = np.zeros(channels[0, :, :].shape)
    for c in range(channels.shape[0]):
        all_in_one += channels[c, :, :]  # minus to have the desired colors

    num_channels, height, width = channels.shape
    model_axes = pylab.gca()
    model_axes.grid(False)

    colormap = pylab.cm.gray_r  # _r stands for "reversed"
    background_alpha = 0.4
    model_axes.imshow(all_in_one,
                      extent=(0, width, height, 0),
                      cmap=colormap, interpolation="nearest",
                      alpha=background_alpha)
    model_axes.set_xticks(range(0, width+1, width/2))
    model_axes.set_yticks(range(0, height+1, width))
    model_axes.xaxis.set_minor_locator(MultipleLocator(min(8, width/4)))
    model_axes.yaxis.set_minor_locator(MultipleLocator(8))
    model_axes.xaxis.grid(False)
    model_axes.set_aspect("equal")

    min_weight = min(top_features_weights)
    max_weight = max(top_features_weights)
    #min_weight = min([x[1] for x in sorted_features_box_and_weight])
    #max_weight = max([x[1] for x in sorted_features_box_and_weight])

    delta_weight = max_weight - min_weight
    if delta_weight > 0:
        weight_scaling = 1.0 / delta_weight
    else:
        weight_scaling = 1.0

    #colormap = pylab.cm.RdYlBu_r  # _r stands for "reversed"
    #colormap = pylab.cm.hsv
    colormap = pylab.cm.spectral_r
    counter = 0
    for box, weight in sorted_features_box_and_weight[:num_top_features]:
        x, y = box.min_corner.x, box.min_corner.y
        box_width = box.max_corner.x - box.min_corner.x
        box_height = box.max_corner.y - box.min_corner.y
        scaled_weight = (weight - min_weight)*weight_scaling
        #color = colormap(scaled_weight)
        #color = colormap(np.random.rand())
        color = colormap(0.1 + float(counter)/(num_top_features*1.15))
        patch = pylab.Rectangle((x,y), box_width, box_height,
                                facecolor=color,
                                edgecolor=color,
                                linewidth=5,
                                alpha=0.5)
        model_axes.add_patch(patch)
        counter += 1
    # end of "for each top box"

    model_axes.set_xlabel("$x$ axis in pixels")
    model_axes.set_ylabel("$y$ axis in pixels")
    model_axes.set_title("Top %i features with largest weights\n" \
                         % num_top_features)
    pylab.gcf().canvas.set_window_title("Top features")

    pylab.tight_layout()

    return

def add_boxes_to_image(image, boxes_and_weights):

    for box, weight in boxes_and_weights:
        slice_y = slice(box.min_corner.y, box.max_corner.y)
        slice_x = slice(box.min_corner.x, box.max_corner.x)
        #image[slice_y, slice_x] += abs(weight)
        image[slice_y, slice_x] += 1

    return


def box_area(box):

    width = box.max_corner.x - box.min_corner.x
    height = box.max_corner.y - box.min_corner.y
    area = width * height
    return area


def box_signed_ratio(box):

    width = box.max_corner.x - box.min_corner.x
    height = box.max_corner.y - box.min_corner.y
    ratio = width / float(height)

    if ratio < 1:
        ratio = -1 / ratio
    elif width == height:
        ratio = 0
    # else ratio > 1: ratio = ratio

    return ratio


def filter_boxes_and_weights(boxes_and_weights,
                             areas, signed_ratios,
                             min_max_area, min_max_ratio):

    verbose = False

    if verbose:
        print("Filtering boxes by area in [%3.4f, %3.4f)\t"
              "and ratio in [%3.4f, %3.4f).\t" % (
              min_max_area[0], min_max_area[1],
              min_max_ratio[0], min_max_ratio[1]), end="")

    assert min_max_area[0] < min_max_area[1]
    #assert min_max_ratio[0] < min_max_ratio[1]

    filtered_boxes_and_weights = []
    for i, box_and_weight in enumerate(boxes_and_weights):
        area = areas[i]
        ratio = signed_ratios[i]

        area_is_fine = area >= min_max_area[0] and area < min_max_area[1]
        ratio_is_fine = ratio >= min_max_ratio[0] and ratio < min_max_ratio[1]

        if area_is_fine and ratio_is_fine:
            filtered_boxes_and_weights.append(box_and_weight)
        else:
            continue

    if verbose:
        print("Selected %i boxes" % len(filtered_boxes_and_weights))

        if len(filtered_boxes_and_weights) == 1:
            #print("boxes ==", filtered_boxes_and_weights)
            box = filtered_boxes_and_weights[0][0]
            print("box min corner == %i, %i; max corner == %i, %i" % (
                  box.min_corner.x, box.min_corner.y,
                  box.max_corner.x, box.max_corner.y))

    return filtered_boxes_and_weights


def plot_boxes_by_area_and_ratio():

    assert len(features_box_and_weight) > 0

    num_areas = 5
    num_ratios = 5

    assert (num_ratios % 2) == 1, "num_ratios should be an odd number"

    _, model_height, model_width = channels.shape

    #float_eps = np.finfo(float).eps
    float_eps = 1e-3  # enough for our purposes, easier to read when printed

    areas = [box_area(x[0]) for x in features_box_and_weight]
    #max_area = model_width*model_height
    max_area, min_area = max(areas) + float_eps, min(areas)
    delta_area = max_area - min_area
    area_step = float(delta_area) / num_areas

    area_ranges = []
    for area_index in range(num_areas):
        min_max_area = \
                (area_step * (area_index + 0)) + min_area, \
                (area_step * (area_index + 1)) + min_area
        area_ranges.append(min_max_area)
    # end of "for each area index"

    if False:
        new_area_ranges = []
        _, bins = np.histogram(areas, bins = num_areas)
        for i in range(len(bins) - 1):
            new_area_ranges.append( (bins[i], bins[i+1]) )

        print("Old area ranges ==", area_ranges)
        print("New area ranges ==", new_area_ranges)
        area_ranges = new_area_ranges

    signed_ratios = [box_signed_ratio(x[0]) for x in features_box_and_weight]
    max_ratio, min_ratio = max(signed_ratios), min(signed_ratios)
    unsigned_max_ratio = max(max_ratio, -min_ratio)
    print("unsigned_max_ratio ==", unsigned_max_ratio)
    mid_ratios_index = (num_ratios - 1) / 2
    ratio_step = unsigned_max_ratio / mid_ratios_index

    ratio_ranges = []
    for ratio_index in range(num_ratios):
        delta_ratio_index = (ratio_index - mid_ratios_index)
        if delta_ratio_index == 0:
            min_max_ratio = 0 - float_eps, 0 + float_eps
        elif delta_ratio_index > 0:
            min_max_ratio = \
                (ratio_step * (delta_ratio_index - 1 + 0)), \
                (ratio_step * (delta_ratio_index - 1 + 1))
        else:  # delta_ratio_index < 0:
            min_max_ratio = \
                (ratio_step * (delta_ratio_index + 1 - 1)), \
                (ratio_step * (delta_ratio_index + 1 - 0))

        # we cover an exception case
        if delta_ratio_index == 1:
            min_max_ratio = 0 + float_eps, min_max_ratio[1]
        elif delta_ratio_index == -1:
            min_max_ratio = min_max_ratio[0], 0 - float_eps

        if ratio_index == 0:
            min_max_ratio = min_max_ratio[0] - float_eps, min_max_ratio[1]
        elif ratio_index == (num_ratios - 1):
            min_max_ratio = min_max_ratio[0], min_max_ratio[1] + float_eps

        ratio_ranges.append(min_max_ratio)
    # end of "for each ratio index"

    if True:
        print("Old ratio ranges", ratio_ranges)
        #cut_a, cut_b = 10, max(model_height, model_width)
        cut_a, cut_b = unsigned_max_ratio/3.0, unsigned_max_ratio
        ratio_ranges =  [(-cut_b, -cut_a), (-cut_a, -float_eps),
                         (-float_eps, float_eps),
                         (float_eps, cut_a), (cut_a, cut_b)]

    print("Ratio ranges", ratio_ranges)
    pylab.tight_layout()

    figure = create_new_figure()

    full_colorbar_image = None

    for ratio_index in range(num_ratios):
        for area_index in range(num_areas):
            axes = pylab.subplot(num_ratios, num_areas,
                                 (num_areas * ratio_index) + area_index + 1)

            min_max_area = area_ranges[area_index]

            delta_ratio_index = (ratio_index - mid_ratios_index)
            min_max_ratio = ratio_ranges[ratio_index]

            filtered_boxes_and_weights = \
                filter_boxes_and_weights(features_box_and_weight,
                                         areas, signed_ratios,
                                         min_max_area, min_max_ratio)
            image = np.zeros((model_height, model_width))
            add_boxes_to_image(image, filtered_boxes_and_weights)

            num_boxes = len(filtered_boxes_and_weights)

            image_min_max = image.min(), image.max()
            #print("Min, max image value == %.3f, %.3f" % image_min_max)

            # draw the figure
            if num_boxes > 0:
                #colormap = pylab.cm.gray
                #colormap = pylab.cm.RdBu
                colormap = pylab.cm.RdYlBu_r  # _r stands for "reversed"

                shown_image = pylab.imshow(image, cmap=colormap,
                                           vmin = 0,
                                           interpolation="nearest")

                full_colorbar_image = shown_image
            else:
                colormap = pylab.cm.Greys
                shown_image = pylab.imshow(image, cmap=colormap,
                                           interpolation="nearest")

            pylab.setp(axes.get_xticklabels(), visible=False)
            pylab.setp(axes.get_yticklabels(), visible=False)


            area_text = "Area in range\n" \
                        "[%.0f, %.0f)\n" \
                        "pixels$^2$" % min_max_area
            if ratio_index == 0:
                pylab.title(area_text + "\n", fontsize=12)

            if num_boxes != 1:
                bottom_text = "%i features" % num_boxes
            else:
                bottom_text = "%i feature" % num_boxes
            if ratio_index == (num_ratios - 1):
                bottom_text += "\n\n" + area_text

            pylab.gca().set_xlabel(bottom_text)

            if area_index == 0:
                if delta_ratio_index != 0:
                    ratio_text = "Ratio in range\n" \
                                 "[%.1f, %.1f)\n" % min_max_ratio
                else:
                    ratio_text = "Square features\n\n"

                pylab.gca().set_ylabel(ratio_text)

            if False:
                if (num_boxes > 0):
                    bar = pylab.colorbar(shown_image)  # use_gridspec=True)
                    bar.set_ticks([image_min_max[0], image_min_max[1]])
            elif full_colorbar_image:
                if area_index == (num_areas - 1) \
                    and delta_ratio_index == 0:
                    #and ratio_index == (num_ratios - 1):
                        bar = pylab.colorbar(full_colorbar_image)  #, use_gridspec=True)
                        im = full_colorbar_image.get_array()
                        bar.set_ticks([0, im.max()])
                        bar.set_ticklabels([0, "maximum\noverlaps"])

        # end of "for each area"
    # end of "for each ratio"

    figure.suptitle("Features coverage for different area and ratio ranges",
                    fontsize="large")
    pylab.gcf().canvas.set_window_title("Features areas and ratios")

    #pylab.tight_layout()  # not good for this plot
    return


def main():

    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and plot its content"

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

        for i, detector_model in enumerate(model.detectors):
            plot_detector_model(detector_model,
                                model_index=i,
                                num_models=len(model.detectors))
            #compute_stumps_statistics(detector_model)
            if True:
                 plot_features_statistics()

    else:  # assume single scale model
        plot_detector_model(model)
		if True:
	        compute_stumps_statistics(model)
        if True:
            plot_weak_classifiers_versus_height(model)
            plot_weak_classifiers_versus_width(model)
        if True:
            plot_features_statistics()

    pylab.show()  # blocking call
    return


if __name__ == '__main__':
    main()

# end of file
