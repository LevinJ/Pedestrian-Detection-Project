#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script that translates an idl file into a protobuf data sequence.
Reading data sequences is much faster than parsing idl files.
"""

from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../helpers")
 
from detections_pb2 import Detections, Detection
from data_sequence import DataSequence
from idl_parsing import open_idl_file

import sys, os.path
from optparse import OptionParser

#import random
        
def idl_data_to_detections(idl_data_generator):
    
    detections_sequence = []

    for line in idl_data_generator:
        filename = line.filename
        detections = Detections()
        detections.image_name = filename
        for box in line.bounding_boxes:
            detection = detections.detections.add()
            detection.object_class = Detection.Pedestrian
            detection.bounding_box.min_corner.x = box[0]
            detection.bounding_box.min_corner.y = box[1]
            detection.bounding_box.max_corner.x = box[2]
            detection.bounding_box.max_corner.y = box[3]
            if len(box) >= 5:            
                detection.score = box[4]
            else:
                detection.score = float("-inf")
                #detection.score = float(random.randint(0,10)) # just for testing
        # end of "for each box in the image"
        
        detections_sequence.append(detections)
    
    return detections_sequence


class IdlToDetectionsSequenceApplication:
    
    def __init__(self):
        return
        
    def run(self, options=None):

        self.parse_arguments()
        self.read_idl_data()
        self.write_data_sequence()
        
        print("End of game. Have a nice day !")
        return

    def parse_arguments(self):
        
        # FIXME should migrate parser to argparse    
        parser = OptionParser()
        parser.description = \
            "This program takes as input " \
            "a idl file containing objects detections and, " \
            "saves a protobuf data sequence containing the same data"
    
        parser.add_option("-i", "--input", dest="input_path",
                           metavar="IDL_FILE", type="string",
                           help="path to the truth idl file")
                           
        parser.add_option("-o", "--ouput", dest="output_path", 
                          metavar="DETECTIONS_FILE", type="string",
                          help="path to the .data_sequence output file")
        
    
        (options, args) = parser.parse_args()
        #print (options, args)
        
        
        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")

        if options.output_path:
            path, extension = os.path.splitext(options.output_path)
            if extension != ".data_sequence":
                parser.error("Output filename should end with extension '.data_sequence'")
            if os.path.exists(options.output_path):
                parser.error("The indicated output file already exist")
        else:
            # create the output path
            options.output_path = options.input_path + ".data_sequence"
            print("Going to save the data into ", options.output_path)
            if os.path.exists(options.output_path):
                parser.error("The output file already exist")
        
        self.options = options    
    
        return # end of parse_arguments


    def read_idl_data(self):

        print("Parsing data from",  self.options.input_path, end=" ... ")
        sys.stdout.flush()
        
        idl_data = open_idl_file(self.options.input_path)       
        self.data_sequence = idl_data_to_detections(idl_data)

        print("DONE")    
        return
        
        
    def write_data_sequence(self):
        
        attributes = { "original_idl_file": self.options.input_path }
        output_data_sequence = \
            DataSequence(self.options.output_path, Detections, attributes)
        
        for detections in self.data_sequence:
            output_data_sequence.write(detections)
        
        del output_data_sequence # close the file
        
        print("Created output file", self.options.output_path)    
        return
# end of class IdlToDetectionsSequenceApplication

        
if __name__ == '__main__':
     # Import Psyco if available
    try:
        import psyco
        psyco.full()
    except ImportError:
        #print("(psyco not found)")
        pass
    else:
        print("(using psyco)")
   
    application = IdlToDetectionsSequenceApplication()
    application.run()

