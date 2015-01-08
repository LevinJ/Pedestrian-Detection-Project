#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import glob
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../helpers")
 
from detections_pb2 import Detections, Detection
from data_sequence import DataSequence

import os, os.path

from optparse import OptionParser


def open_data_sequence(data_filepath):
        
    assert os.path.exists(data_filepath)
    
    the_data_sequence = DataSequence(data_filepath, Detections)
    
    def data_sequence_reader(data_sequence):    
        while True:
            data = data_sequence.read()
            if data is None:
                raise StopIteration
            else:
                yield data
    
    return data_sequence_reader(the_data_sequence)    
    

def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes a detections.data_sequence created by ./objects_detection and converts it into the Caltech dataset evaluation format"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the folder containing the recordings")

    parser.add_option("-o", "--output", dest="output_path",
                       type="string",
					   help="path to a directory where the curves are saved")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    if options.output_path:
		pass
    else:
        parser.error("'output_path' option is required to run this program")

    return options 



def getDetections(detections_sequence):
	output_detections = []
	negCount = 0
	posCount = 0
	detectionsDict = {}

	for detections in detections_sequence:
		det_class =0

		if detections.image_name.startswith("neg"):
			det_class = -1
			negCount +=1
		else:
			det_class = 1
			posCount+=1

		if (len(detections.detections)) >1:
			raise Exception("more than one detection per image")
		for detection in detections.detections:
			if detection.object_class != Detection.Pedestrian:
				continue
			
			detection_data = [det_class, detection.score]
			detectionsDict[detections.image_name] = detection.score
			#print(detection_data)
			output_detections.append(detection_data)
	#for key in detectionsDict.iterkeys():
	#	print (key," ", detectionsDict[key])
			
		
#sort detections by score
	output_detections = sorted(output_detections, key=lambda det: det[1],reverse=True)
	return [output_detections, posCount, negCount]
def saveCurve(detections, posCount, negCount, output_file):
	f = open(output_file, "w")

	allDet = len(detections) 
	tp = 0
	fp = 0
	fn = 0
	for det in detections:
		if det[0] ==1:
			tp +=1
		else:
			fp +=1
		
		fn = posCount-tp

		pr = tp/float(tp+fp)
		rec = tp/float(tp + fn)
		fppw = float(fp)/allDet
		missrate = fn/float(posCount)
		line = str(missrate) + " " + str(fppw) + "\n"
		f.writelines(line)
		#print("missrate: ", fn/float(posCount))
		#print("fppw: ", float(fp)/allDet)

	f.close()



def detections_to_precision_recall(input_path, output_file):
    
    # get the input file
    #input_file = open(options.input_path, "r")
    detections_sequence = open_data_sequence(input_path)

    
    # convert data sequence to caltech data format
    [detections, posCount, negCount] = getDetections(detections_sequence)
    saveCurve(detections, posCount, negCount, output_file)
    
    return
def getFolders(path):
	directories = []
	for d in glob.glob(os.path.join(path,"*")):
		if os.path.isdir(d):
			directories.append(d)
	return sorted(directories)


def main():
    
	options = parse_arguments()    
	counter = -4
	for d in getFolders(options.input_path):
		counter = counter + 4
		data_sequence = os.path.join(d, "detections.data_sequence")
		output_file = os.path.join(options.output_path, "crop_%03d.txt" %(counter) )
			
		detections_to_precision_recall(data_sequence, output_file)
	return


if __name__ == "__main__":
        
    main()






