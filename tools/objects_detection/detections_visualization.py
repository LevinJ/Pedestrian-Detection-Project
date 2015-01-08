#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small utility to visualize the detections output from objects_detection
"""


from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../helpers")
 
from detections_pb2 import Detections
from data_sequence import DataSequence

import os.path 
from optparse import OptionParser

from random import seed, random

try:
    import Image, ImageDraw
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this application")



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
             "This program takes a detections.data_sequence created by ./objects_detection and plots the results"
             
        parser.add_option("-i", "--input", dest="input_path",
                           metavar="FILE", type="string",
                           help="path to the .data_sequence file")
        parser.add_option("-n", "--frame_number", dest="frame_number", 
                          metavar="NUMBER", type="int", default=-200,
                          help="frame number to visualize. " \
                          "If a negative number is given a random number between 0 and NUMBER is chose.")        
        parser.add_option("-t", "--score_threshold", dest="score_threshold", 
                          metavar="NUMBER", type="float", 
                          help="score threshold used to prune the visualization")
        
        (options, args) = parser.parse_args()
        #print (options, args)
    
        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")
        else:
            parser.error("'input' option is required to run this program")
    
        
        if options.frame_number == 0:
                parser.error("'frame_number' number must be != 0")
        elif options.frame_number > 0:
            pass
        else: # options.frame_number < 0:
            seed() # provide a seed for random
            options.frame_number = int(random()*-options.frame_number)
            
        
        return options 

def get_images_base_path(input_path):
    
    program_options_path = os.path.join(os.path.dirname(input_path), "program_options.txt")
    if not os.path.exists(program_options_path):
        raise Exception("Could not find required file " + program_options_path)

    process_folder = None    
    for line in open(program_options_path):
        a = line.split("=")
        b = [x.strip() for x in [a[0], "=".join(a[1:])]]
        if b[0] == "process_folder":
            process_folder = b[1]
            break
    
    if not process_folder:
        raise Exception("Failed to find the 'process_folder' inside " + program_options_path)
    
    return process_folder
    
    
    
def plot_detections(image, detections, score_threshold):
    
    draw = ImageDraw.Draw(image)
        
    rectangles_to_plot = []
    min_score = float("inf")
    max_score = -float("inf")
    for detection in detections.detections:
        if type(score_threshold) is float and detection.score < score_threshold:
            pass
        else:
            min_score = min(min_score, detection.score)
            max_score = max(max_score, detection.score)
            box = detection.bounding_box
            rectangle = [ (box.max_corner.x, box.max_corner.y), (box.min_corner.x, box.min_corner.y)]
            rectangles_to_plot.append((rectangle, detection.score))
                    
    if max_score != min_score:
        scaling = 200.0 / (max_score - min_score)
    else:
        scaling = 200.0 / max_score 
        
    for rectangle, score in rectangles_to_plot:
        normalized_score = int( (score - min_score) * scaling ) + 55
        #color = "white"
        #color = "rgb(%i,%i,%i)" % (normalized_score, normalized_score, normalized_score)
        color = "rgb(%i,%i,%i)" % (normalized_score, 0, 0) # red results
        for i in [-1, 0, 1]:
            t_rectangle = [(rectangle[0][0] + i, rectangle[0][1] + i),
                           (rectangle[1][0] + i, rectangle[1][1] + i),]
            draw.rectangle(t_rectangle, outline=color)        
    
    
    print("score_threshold == ", score_threshold)
    print("max_score, min_score == (%.3f, %.3f)" % (max_score, min_score))
    image.show()
    
    return
    


def main():
    
    options =  parse_arguments()    
    
    # get the input data sequence
    detections_sequence = open_data_sequence(options.input_path)


    # find the correct frame
    detections = None
    i = 0
    for detections in detections_sequence:
        if i == options.frame_number:
            break
        else:
           i += 1
    
    assert detections and detections.image_name

    image_filename = detections.image_name
    if not os.path.exists(detections.image_name):
        # find the program_options.txt file
        images_base_path = get_images_base_path(options.input_path)
        image_filename = os.path.join(images_base_path, detections.image_name)
        if not os.path.exists(detections.image_name):
            raise Exception("Failed to find the input image")
        
        
    image = Image.open(image_filename)
    assert image
    
    # plot
    print("Plotting frame %i:" % i, detections.image_name, image.size, "pixels" )
    plot_detections(image, detections, options.score_threshold)
    
    print("Have a nice day")
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



