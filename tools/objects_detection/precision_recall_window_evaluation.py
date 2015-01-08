#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mini VISICS specific script to plot the caltech evaluation results on the INRIA dataset
"""

from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")
 

import os, os.path, shutil, subprocess, glob
from optparse import OptionParser

from detections_to_caltech import detections_to_caltech

at_visics = True
#at_visics = False

def get_precision_recall(results_path, output_file):

    results_path = os.path.normpath(results_path)
    results_name = os.path.split(results_path)[-1]
    outfile = open(output_file, "w")
    all_detections = []

    data_sequence_path = os.path.join(results_path, "detections.data_sequence")
    v000_path  = os.path.join(results_path, "preRec")

    if os.path.exists(v000_path):
        print(v000_path, "already exists, skipping the generation step")
    else:
        # convert data sequence to caltech data format
        detections_to_caltech(data_sequence_path, v000_path)
	for f in glob.glob(os.path.join(v000_path, "*.txt")):
		all_detections.append(readDetection(f))
		all_detections = sorted(all_detections, key=lambda det: det[1], reverse=True)
		print (all_detections)

	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for det in all_detections:
		if det[0] =="pos":
			if det[2]:
				tp +=1
			else:
				fn +=1

		if det[0] =="neg":
			if not det[2]:
				tn +=1
			else:
				fp +=1
		correct = tp + tn
		incorrect = fp+fn
		allDet = tp + fn
		pr = 5#float(correct)/float(correct+incorrect)
		recall = 3#float(correct)/allDet
		outfile.writelines(str(pr) + " " + str(recall) + "\n")




		


def readDetection(det_file):
	f = open(det_file)
	lines = f.readlines()
	cl = ""
	if det_file.find("pos") >= 0:
		cl = "pos"
	elif det_file.find("neg")>=0:
		cl = "neg"
	else:
		#raise InputError("the filename does not encode a positive or negative class")
		print ("ERROR: unknown class")
				
	if len(lines) > 1:
		lines = lines[0].strip().split(",")
		lines = [k.strip() for k in lines]
		print(lines)
		det =  [float(k) for k in lines]
		print (det)
		print ("ERROR: more than one detection")
		return [cl, det[4], True]
	return [cl, 0, False]

def main():

    parser = OptionParser()
    parser.description = \
        """
        Reads the recordings of objects_detection over Caltech version of INRIA dataset, 
       	generates an output file containing the values for precision/recall 
        """

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the recording directory")
    parser.add_option("-o", "--output", dest="output_file",
                       metavar="FILE", type="string",
                       help="output file containing precision recall values")
    (options, args) = parser.parse_args()
    #print (options, args)
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input directory")
    else:
        parser.error("'input' option is required to run this program")

    if not os.path.isdir(options.input_path):
        parser.error("the 'input' option should point towards " \
                     "the recording directory of the objects_detection application")

    results_path = options.input_path
    output_file= options.output_file

    get_precision_recall(results_path, output_file)

    return
        

if __name__ == '__main__':
    main()
    
    
