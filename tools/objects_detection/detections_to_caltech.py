#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os.path
import sys
local_dir = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(local_dir, ".."))
sys.path.append(os.path.join(local_dir, "../data_sequence"))
sys.path.append(os.path.join(local_dir, "../helpers"))
 
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
                       help="path to the .data_sequence file")

    parser.add_option("-o", "--output", dest="output_path",
                       metavar="DIRECTORY", type="string",
                       help="path to a non existing directory where the caltech .txt files will be created")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to a non existing directory")
    else:
        parser.error("'output' option is required to run this program")

    return options 



def create_caltech_detections(detections_sequence, output_path):
    """
    """
    
    for detections in detections_sequence:
        file_path = os.path.join(output_path,  
                                     os.path.splitext(detections.image_name)[0] + ".txt")
        text_file = open(file_path, "a") # append to the file
            
        for detection in detections.detections:
            if detection.object_class != Detection.Pedestrian:
                continue
            
            if False and detection.score < 0:
                # we skip negative scores
                continue
            
            box = detection.bounding_box
            min_x, min_y = box.min_corner.x, box.min_corner.y
            width = box.max_corner.x - box.min_corner.x
            height = box.max_corner.y - box.min_corner.y
            
            adjust_width = True
            if adjust_width:
                # in v3.0 they use 0.41 as the aspect ratio
                # before v3.0 they use 0.43 (measured in their result files)
                #aspect_ratio = 0.41
                aspect_ratio = 0.43
                center_x = (box.max_corner.x + box.min_corner.x) / 2.0
                width = height*aspect_ratio
                min_x = center_x - (width/2)

            # data is [x,y,w,h, score]
            detection_data = []            
            detection_data += [min_x, min_y]
            detection_data += [width, height]
            detection_data += [detection.score]
            detection_line = ", ".join([str(x) for x in detection_data]) + "\n"
            text_file.write(detection_line)
            
        text_file.close()
        print("Created file ", file_path)
        
    return

def detections_to_caltech(input_path, output_path):
    
    # get the input file
    #input_file = open(options.input_path, "r")
    detections_sequence = open_data_sequence(input_path)

    os.mkdir(output_path)    
    print("Created the directory ", output_path)
    
    # convert data sequence to caltech data format
    create_caltech_detections(detections_sequence, output_path)
    
    return

def main():
    
    options = parse_arguments()    
    detections_to_caltech(options.input_path, options.output_path)
    return


if __name__ == "__main__":
        
    # Import Psyco if available
    try:
        import psyco
        psyco.full()
    except ImportError:
        #print("(psyco not found)")
        pass
    else:
        print("(using psyco)")
      
    main()






