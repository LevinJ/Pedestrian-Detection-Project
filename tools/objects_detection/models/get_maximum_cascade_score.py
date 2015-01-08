#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a trained model, this script will sum up the weights of the individual stages to find the maximum possible detection score.
"""

#from __future__ import print_function

import detector_model_pb2
import os.path #, sys
from optparse import OptionParser

#import numpy as np


def parse_cascade(cascade):
    
	sum_score = 0
	for i, stage in enumerate(cascade.stages):
		if stage.feature_type == stage.Level2DecisionTree:
			sum_score += stage.weight
		else:
			raise "Received an unhandled stage.feature_type"            
	return sum_score

def read_model(model_filename):
        
    model = detector_model_pb2.DetectorModel()
    f = open(model_filename, "rb")
    model.ParseFromString(f.read())
    f.close()
    

    return model
    
    
    
def get_max_detector_model_score(model_filename):
    
    sum_score = 0
    model = read_model(model_filename)

    if model.detector_type == model.SoftCascadeOverIntegralChannels:
        cascade = model.soft_cascade_model    
        sum_score = parse_cascade(cascade)
    else:
		return -1
    
    return sum_score

def main():

    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and plot its content"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the model file")
    (options, args) = parser.parse_args()
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    model_filename = options.input_path

    print "The maximum score of the given model is: ", get_max_detector_model_score(model_filename)

    return
        

if __name__ == '__main__':
    main()
