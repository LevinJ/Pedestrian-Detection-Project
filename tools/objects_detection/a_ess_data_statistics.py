#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test program that computes a few statistcs on Andreas Ess original data
"""

from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../stixels_evaluation")
sys.path.append("../helpers")
 
import pylab 
 
#from detections_pb2 import Detections, Detection
#from data_sequence import DataSequence
from idl_parsing import open_idl_file
from idl_to_detections_sequence import idl_data_to_detections

dalal_idl_path = "/home/rodrigob/code/ethz_svn/projects/data/gt_and_idl/bahnhof/dalal-raw-undist.idl"
ground_truth_idl_path = "/home/rodrigob/data/bahnhof/annotations/bahnhof-annot.idl"

print("Reading data...")
idl_data = open_idl_file(dalal_idl_path)
dalal_detections = idl_data_to_detections(idl_data)

idl_data = open_idl_file(ground_truth_idl_path)
ground_truth_detections = idl_data_to_detections(idl_data)

print("Computing ...")
def compute_heights(detections_sequence):       
    heights = []
    for detections in detections_sequence:
        for detection in detections.detections:
            bb = detection.bounding_box
            height = bb.max_corner.y - bb.min_corner.y
            heights.append(height)
    return heights
    
dalal_heights = compute_heights(dalal_detections)
ground_truth_heights = compute_heights(ground_truth_detections)

print("Min dalal idl height ==", min(dalal_heights))
print("Min ground truth idl height ==", min(ground_truth_heights))

pylab.figure()
pylab.hist(dalal_heights, bins=100, histtype='step', cumulative=True, label="HOG+Linear SVM")
pylab.hist(ground_truth_heights, bins=100, histtype='step', cumulative=True, label="Ground truth")

pylab.legend()
pylab.show() # blocking call