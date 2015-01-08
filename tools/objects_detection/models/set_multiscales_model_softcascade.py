#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a multiscales model and a cascade_threshold.txt file created from test_objects_detection,
will generate an updated multiscales model with the proper cascade threshold set.
"""

from __future__ import print_function

from detector_model_pb2 import MultiScalesDetectorModel

import os, os.path
from optparse import OptionParser

from pylab import loadtxt

class Point2d:
    """
    Helper class to define Box
    """
    def __init__(self, x,y):
        self.x = x
        self.y = y
        return
        
class Box:
    """
    This class replaces from detections_pb2 import Box
    since it does not support negative coordinates for the boxes
    """
    
    def __init__(self):
        self.min_corner = Point2d(0,0)
        self.max_corner = Point2d(0,0)
        return

def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes multiscale model and " \
        "a cascade_threshold.txt file created from test_objects_detection, and" \
        "will ouput a model with updated cascades thresholds"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string", default=None,
                       help="path to a multiscales model")

    parser.add_option("-t", "--thresholds", dest="cascades_thresholds_path",
                       metavar="PATH", type="string", default=None,
                       help="path to the 'cascade_threshold.txt' file created by  test_objects_detection")


    parser.add_option("-o", "--output", dest="output_path",
                       metavar="PATH", type="string",
                       help="path to multiscales model to be created")
                                                  
    (options, args) = parser.parse_args()
    #print (options, args)


    if not options.input_path:
        parser.error("'input' option is required to run this program")
    if not os.path.exists(options.input_path):
            parser.error("Could not find the input file %s" % options.input_path)        

    
    if not options.cascades_thresholds_path:
        parser.error("'thresholds' option is required to run this program")
    if not os.path.exists(options.cascades_thresholds_path):
        parser.error("Could not find the input file %s" % options.cascades_thresholds_path)        


    if not options.output_path:
        parser.error("'output' option is required to run this program")
    if os.path.exists(options.output_path):
        parser.error("Could the indicated output file %s already exists, " \
                      "please select another name" % options.output_path)        
    
    return options 


    
def create_updated_multiscales_model(input_path, cascades_thresholds_path, output_path):
    
    
    multiscales_model = MultiScalesDetectorModel()
    f = open(input_path, "rb")
    multiscales_model.ParseFromString(f.read())
    f.close()
    
    thresholds = loadtxt(cascades_thresholds_path)

    print("Read all data, updating the cascade thresholds...")    
    
    if len(multiscales_model.detectors) != thresholds.shape[0]:
        raise Exception("The number of detectors in the multiscales model, " \
                        "does not match the number of cascade thresholds")


    # update all the cascade thresholds     
    for detector_index, detector in enumerate(multiscales_model.detectors):

        # cascade thresholds where computed using normalized weights,
        # so we need to scale them back
        weights_sum = sum([float(stage.weight) for stage in detector.soft_cascade_model.stages])
            
        for stage_index, stage in enumerate(detector.soft_cascade_model.stages):
            threshold = thresholds[detector_index, stage_index]
            if abs(threshold) < 1E10:
                # non-maxfloat value
                threshold *= weights_sum             
            else:
                if threshold > 0:
                    # bug in test_evaluation code, this should be a very negative value
                    # (not a very positive one)
                    threshold = -threshold
                else:
                    # threshold = threshold
                    pass
            stage.cascade_threshold = threshold
        # end of "for each stage"                
    # end of "for each detector"
        

    out_file = open(output_path, "wb")
    out_file.write(multiscales_model.SerializeToString())
    out_file.close()
    print("Created output model file", output_path)
        
    return


def main():
    options = parse_arguments()    
    
    create_updated_multiscales_model(options.input_path, options.cascades_thresholds_path, options.output_path)
        
    print("End of game, have a nice day!")
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




        