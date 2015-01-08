#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a trained model from boosted_learning and a trained SVM from liblinear,
this script will modify the weak classifiers weights of the boosted model.
"""

from __future__ import print_function

import detector_model_pb2
import os.path #, sys
from optparse import OptionParser

import numpy as np
#from PIL import Image

import pylab


from plot_detector_model import read_model

def get_svm_model_w_vector(input_file):

    line = input_file.readline()
    while "w\n" not in line:
        line = input_file.readline()
    w_vector = []
    for line in input_file.readlines():
        w_vector.append(float(line))
    
    return w_vector

    
def adjust_weights(model, weights):
    
    assert model.detector_type == model.SoftCascadeOverIntegralChannels
    cascade = model.soft_cascade_model 

    assert len(weights) == len(cascade.stages), \
        "The number of stages in the boosted classifier " \
        "does not match the lenght of the SVM weights vector " \
        "(%i != %i)" % (len(weights), len(cascade.stages))
    
    for i, stage in enumerate(cascade.stages):
            stage.weight = weights[i]
    # end of "for each stage"
          
    return model

    
def cut_model(model, num_stages):
    
    assert model.detector_type == model.SoftCascadeOverIntegralChannels
    cascade = model.soft_cascade_model 

    if num_stages != len(cascade.stages):
        print ("We will cut the model from %i to %i stages" % (len(cascade.stages), num_stages) )
    
    del cascade.stages[-(len(cascade.stages) - num_stages):]
     
    assert len(cascade.stages) == num_stages
    return model


def parse_input_options():
    
    # TO BE DONE --    
    
    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and plot its content"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the boosted model file")
    (options, args) = parser.parse_args()
    #print (options, args)
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")
    
    return options

def main():

    #options = parse_input_options()
    #model_filename = options.input_path

    boosted_model_path = "../../src/applications/boosted_learning/2012_04_16_74669_full_size_model_non_recursive_seed_22222_push_up_0.05.bootstrap2crop_32"
    svm_model_path = "../../libs/liblinear-1.8/tmp.model"
    output_boosted_model_path = "test_model_c0.001_e0.001_dataBalanced.out.proto.bin"
    
    assert boosted_model_path != output_boosted_model_path
    
    print("Reading boosted model", boosted_model_path)
    model = read_model(boosted_model_path)
    print("Reading svm model", svm_model_path)    
    w_vector = get_svm_model_w_vector(open(svm_model_path, "r"))
    
    if type(model) is detector_model_pb2.DetectorModel:
	print("Cutting model...")
        cut_model(model, len(w_vector))

	print("Saving cut model at", output_boosted_model_path + ".cut")
        out_file = open(output_boosted_model_path + ".cut", "wb")
        out_file.write(model.SerializeToString())
        out_file.close()

        print("Adjusting weights model...")
        adjust_weights(model, w_vector)
        
	print("Saving result model at", output_boosted_model_path)
        out_file = open(output_boosted_model_path, "wb")
        out_file.write(model.SerializeToString())
        out_file.close()

	output_boosted_model_path
        print("Created output model file", output_boosted_model_path)

    else: 
        raise Exception("No model type other than DetectorModel is supported. "
                        "Received %s" % type(model) )

    pylab.show() # blocking call
    return
        

if __name__ == '__main__':
    main()
