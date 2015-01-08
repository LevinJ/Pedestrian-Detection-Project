#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for the detections evaluation.

Example usage

    
~/code/doppia/tools/objects_detection$ ./detections_evaluation.py -g ~/data/bahnhof/annotations/bahnhof-annot.idl -d ~/code/doppia/src/applications/objects_detection/2011_08_19_75791_recordings_bahnhof_chnftrs/detections.data_sequence -d ~/code/doppia/src/applications/objects_detection/2011_08_19_80853_recordings_stixels_evaluation_default_parameters/detections.data_sequence -d ~/code/doppia/src/applications/objects_detection/2011_08_19_86008_recordings_stixels_evaluation_using_residual/detections.data_sequence -d ~/code/ethz_svn/projects/data/gt_and_idl/bahnhof/dalal-raw-dist.idl
"""

from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../stixels_evaluation")
sys.path.append("../helpers")
 
from detections_pb2 import Detections, Detection
from data_sequence import DataSequence
from idl_parsing import open_idl_file
from idl_to_detections_sequence import idl_data_to_detections
from stereo_rectification import StereoRectification


import os, os.path
import re
from itertools import izip
from copy import copy#, deepcopy
from optparse import OptionParser

import pylab, numpy


class Bunch:
    """
    Helper class that stores a bunch of stuff and provide easy access
    
    >>> point = Bunch(datum=y, squared=y*y, coord=x)
    >>> if point.squared > threshold:
    ...     point.isok = 1

    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        

def area(rect):
    "rect is expected to be a bounding box"
    w = abs(rect.max_corner.x - rect.min_corner.x)
    h = abs(rect.max_corner.y - rect.min_corner.y)

    #print(dir(rect.max_corner))
    #print("rect.max_corner.x ==", rect.max_corner.x)
    #print("rect.min_corner.x ==", rect.min_corner.x)
    #print(rect.max_corner)
    #print(rect, "w ==", w, "h ==", h)
    return float(w*h)
        
def overlapping_area(a, b):
   """
   a and b are expected to be detection bounding boxes
   code adapted from http://visiongrader.sf.net
   """       
   w = min(a.max_corner.x, b.max_corner.x) - max(a.min_corner.x, b.min_corner.x)
   h = min(a.max_corner.y, b.max_corner.y) - max(a.min_corner.y, b.min_corner.y)
   if w < 0 or h < 0:
        return 0
   else:
        return float(w * h)
        
def do_overlap(a,b):
    """
    a and b are expected to be detection bounding boxes
    code adapted from http://visiongrader.sf.net
    """       
    return (a.min_corner.x < b.max_corner.x and a.max_corner.x > b.min_corner.x \
            and a.min_corner.y < b.max_corner.y and a.max_corner.y > b.min_corner.y)

def intersection_over_union(detection, ground_truth):
    """
    """
    #b1 = detection.bounding_box; b2 = ground_truth.bounding_box
    b1 = detection; b2 = ground_truth
    inter = overlapping_area(b1, b2)
    #assert inter > 0 # in our setup this should always be true
    union = area(b1) + area(b2) - inter
    assert union > 0
    intersection_over_union =  (inter / union)
    #print("IoU", intersection_over_union, b1, b2)
    return intersection_over_union


def intersection_over_union_criterion(detection, ground_truth, p):
    """
    p is the maximum ratio allowed between the intersection and the union.
    Set it to 0.5 to have 50%.
    """

    return intersection_over_union(detection, ground_truth) > p
   

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
    
    
def rectify_left_detection(stereo_rectification, detection):

    box = detection.bounding_box
    box_tuple = (box.min_corner.x,box.min_corner.y, box.max_corner.x,box.max_corner.y)
    rectified_box = stereo_rectification.rectify_left_bounding_box(box_tuple)

    detection.bounding_box.min_corner.x = max(int(rectified_box[0]),0)
    detection.bounding_box.min_corner.y = max(int(rectified_box[1]),0)
    detection.bounding_box.max_corner.x = max(int(rectified_box[2]),0)
    detection.bounding_box.max_corner.y = max(int(rectified_box[3]),0)
    return detection
    

def rectify_left_detections(stereo_rectification, detections_sequence):
    
    rectified_detections = copy(detections_sequence)

    for detections in rectified_detections:
        for detection in detections.detections:
            detection = rectify_left_detection(stereo_rectification, detection)

    return rectified_detections
    

def adjust_detection_width(detection):
    
    box = detection.bounding_box
        
    # in Caltech v3.0 they use 0.41 as the aspect ratio
    # before v3.0 they use 0.43 (measured in their result files)
    #aspect_ratio = 0.41
    aspect_ratio = 0.43
    center_x = (box.max_corner.x + box.min_corner.x) / 2.0
    height = box.max_corner.y - box.min_corner.y
         
    width = height*aspect_ratio
    box.min_corner.x = int(max(0, round(center_x - (width/2))))
    box.max_corner.x = int(max(0, round(center_x + (width/2))))
                    
    #detection.bounding_box = box
    return detection


def adjust_detections_width(detections_sequence):

    adjusted_detections = copy(detections_sequence)

    for detections in adjusted_detections:
        for detection in detections.detections:
            detection = adjust_detection_width(detection)

    return adjusted_detections
    

class DetectionsEvaluationApplication:
    
    def __init__(self):
        return
        
    def run(self, options=None):

        if not options:
            self.parse_arguments()
        else:
            self.options = options

        self.open_stereo_rectification()                
        self.read_ground_truth_annotations()
        self.read_data_sequences()
        
        if self.should_render_example_result:
            self.render_example_result()
        else:
            self.render_evaluation_graphs()



        print("End of game. Have a nice day !")
        return

    def parse_arguments(self):
        
        # FIXME should migrate parser to argparse    
        parser = OptionParser()
        parser.description = \
            "This program takes as input " \
            "a recording directories created by objects_detection and/or idl files, " \
            "a ground truth idl file and computes an evaluation graph comparing the recordings"
    
        parser.add_option("-g", "--ground_truth", dest="ground_truth_path",
                           metavar="DETECTIONS_FILE or IDL_FILE", type="string",
                           help="path to the ground truth annotations file")
                           
        parser.add_option("-r", "--recording", dest="recordings_paths", 
                          metavar="DIRECTORY", 
                          type="string", action="append", default=[],
                          help="path to the recording directory containing the detections data sequence")
        parser.add_option("-d", "--detections", dest="detections_paths", 
                          metavar="DETECTIONS_FILE or IDL_FILE", 
                          type="string", action="append", default=[],
                          help="path to the detections datasequence or idl file (with detection confidence)")
                          
        parser.add_option("-o", "--output", dest="output_filename", 
                          metavar="FILENAME", type="string",
                          help="define a specific output filename, should be an image type")
                          
        parser.add_option("--max_num_frames", dest="max_num_frames", 
                          metavar="NUMBER", type="int", default=-1,
                          help="when this number is positive it will limit the number of frames to be processed" \
                                "(usefull for debugging purposes) [default: %default]")

        parser.add_option("--minimum_box_height", dest="minimum_box_height", 
                          metavar="NUMBER", type="int", default=40,
                          help="minimum height required to consider a ground truth or detection box [default: %default]")                          
        # minimum_box_height
        # WRONG ! >= 60 leads to considering 4626 out of at total of 7698 pedestrians on Bahnhof         
        # WRONG ! >= 50 leads to considering 5783 out of at total of 7698 pedestrians on Bahnhof     
        # WRONG ! >= 40 leads to considering 6620 out of at total of 7698 pedestrians on Bahnhof   
        # >= 40 leads to considering 7391 out of at total of 7698 pedestrians on Bahnhof         
        # >= 60 leads to considering 5342 out of at total of 7698 pedestrians on Bahnhof         
        # A. Ess thesis considered 5193 pedestrians in the 999 frames of Bahnhof sequence        

        parser.add_option("-f", "--frame", dest="example_result_frame", 
                          metavar="NUMBER", type="int", default=-1,
                          help="if frame is bigger than -1 it will be used to render an example result frame [default: %default]")
        
        parser.add_option("--adjust_width", dest="adjust_width", 
                          metavar="BOOLEAN", type="int", default=1,
                          help="adjust the width of ground truth and detections so that they all have the same height to width ratio [default: %default]")                          
        
        (options, args) = parser.parse_args()
        #print (options, args)

        self.should_render_example_result = options.example_result_frame > -1
                
        if options.ground_truth_path:
            if not os.path.exists(options.ground_truth_path):
                parser.error("Could not find the ground_truth file")
		
        if (not options.recordings_paths) and (not options.detections_paths):
              parser.error("Not input recording/detections was indicated, nothing to do")
		
        for path in (options.recordings_paths + options.detections_paths):
                if not os.path.exists(path):
                        print("'%s' does not exit" % path)
                        parser.error("Could no find indicated file or directory")
                
        for directory in options.recordings_paths:                
                if not os.path.isdir(directory):
                        print("'%s' is not a directory" % directory)
                        parser.error("The --recordings option expects to receive a directory, not a file")

        self.options = options    
    
        return # end of parse_arguments


    def read_ground_truth_annotations(self):

        if True:
            path = self.options.ground_truth_path
            basepath, extension = os.path.splitext(path)
            if extension == ".idl":
                idl_data = open_idl_file(path)       
                self.ground_truth = idl_data_to_detections(idl_data)
            elif extension == ".data_sequence":
                self.ground_truth = list(open_data_sequence(path))
            else:
                raise Exception("Unknown filename extension '%s' in file %s" % \
                                    (extension, path))
    
            if False: # should be false, this offset is already compensated somewhere else
                # we skip the first frame, since doppia code skip the first frames too
                self.ground_truth = self.ground_truth[1:]
    
            if self.stereo_rectification:       
                self.ground_truth = rectify_left_detections(self.stereo_rectification, self.ground_truth)
               
            if self.options.adjust_width:   
               self.ground_truth = adjust_detections_width(self.ground_truth)
               
            print("Read ground truth from", self.options.ground_truth_path)    
        else:
            sys.path.append("../stixels_motion_evaluation")
            from stixels_motion_evaluation import read_ground_truth
            
            self.options.stereo_rectification = self.stereo_rectification
            self.ground_truth = read_ground_truth(self.options)
            # self.stereo_rectification = self.options.stereo_rectification
            
        return
        
        
    def open_recordings_options(self):
        for name, recording in self.recordings.items():
            recording.options = self.open_recording_options(recording)
        return
            
    def open_recording_options(self, recording):
        """
        All the options associated to a recording
        """        
        
        options_filename = os.path.join(recording.directory, "program_options.txt")
        
        if not os.path.exists(options_filename):
            raise ValueError("could not find the program options file at %s" % options_filename)
        recording_options = dict()

        with open(options_filename, "r") as options_file:           
         for line in options_file.readlines():
             splitted_line = line.split('=')
             key = splitted_line[0].strip()
             value = ('='.join(splitted_line[1:])).strip()
             recording_options[key] = value
             
        return recording_options
            
            
    def open_stereo_rectification(self):
        
        #a_recording = self.recordings.values()[0] # one of the recordings
        #calibration_filename = self._get_recordings_option("video_input.calibration_filename")
        
        #calibration_filename = os.path.join(a_recording.directory, "..", calibration_filename)
        # FIXME hardcoded value        
        calibration_filename = "../../src/video_input/calibration/stereo_calibration_bahnhof.proto.txt"
        
        print("Using calibration filename:", calibration_filename)
        self.stereo_rectification = StereoRectification(calibration_filename)

        return
        
    def read_data_sequences(self):

        self.data_sequences = dict()

        paths = []
        paths += [os.path.join(path, "detections.data_sequence") for path in self.options.recordings_paths]
        paths += self.options.detections_paths       

        for path in paths:
            basename, extension = os.path.splitext(path)
            path_pieces = os.path.normpath(os.path.abspath(basename)).split(os.sep)
            
            if extension == ".data_sequence":
                recording_name = path_pieces[-2] # the folder name
                data_sequence = list(open_data_sequence(path))
            elif extension == ".idl":
                recording_name = path_pieces[-1] # the filename, without extension
                idl_data = open_idl_file(path)
                data_sequence = idl_data_to_detections(idl_data)
                
                if True:
                    # we skip the first frame, since doppia code skip the first frames too
                    # if from idl, we assume it does not come from doppia code
                    data_sequence = data_sequence[1:]
        
                if self.stereo_rectification: # if idl, we assume, undistorted, non rectified data
                    data_sequence = rectify_left_detections(self.stereo_rectification, data_sequence)
            else:
                raise Exception("Unknown filename extension '%s' in file %s" % \
                                (extension, path))

            if self.options.adjust_width:   
                data_sequence = adjust_detections_width(data_sequence)
        
            print("Read data for", recording_name)
            self.data_sequences[recording_name] = data_sequence
                            
        return 
        
  
    def _get_recordings_option(self, key):
        """
        Retrieve a recording option and check it is the same in all recordings
        """
        
        a_recording = self.recordings.values()[0] # one of the recordings
        value = a_recording.options[key]
        for recording in self.recordings.values():
            # we check that all the recordings have the same value
            assert recording.options[key] == value
            
        return value

        
    def render_evaluation_graphs(self):
        """
        Compute the result graphs, plot them and then show them
        """

        self._plot_detection_rates()
        
        pylab.show()
        return

                
    
    def _get_num_frames(self):
        
        end_frame = self._get_recordings_option("video_input.end_frame")
        start_frame = self._get_recordings_option("video_input.start_frame")
        
        num_frames = int(end_frame) - \
                     int(start_frame)
        
        return num_frames
    
    def _get_max_num_frames(self):
        
        if self.options.max_num_frames > 0:
            max_num_frames = self.options.max_num_frames
        else:
            max_num_frames = float("inf")
            
        return max_num_frames
    
    
    def _get_dataset_name(self):
        
        ground_truth_filename = os.path.split(self.options.ground_truth_path)[-1]
        if ground_truth_filename.lower().find("bahnhof") != -1 \
            or ground_truth_filename.lower().find("part06") != -1:
            return "Bahnhof dataset"
        else:
            raise "_get_dataset_name not yet properly implemented"
            calibration_filename = self._get_recordings_option("video_input.calibration_filename")
            m = re.search(".*stereo_calibration_(.*)\.proto.*", calibration_filename)
            sequence_name = m.group(1)
            dataset_name = "%i frames of sequence %s" % (
            self.num_accounted_frames, sequence_name.capitalize())

        return dataset_name

     
    @staticmethod    
    def _filter_detections_by_height(detections, minimum_height, maximum_height = float("inf")):
        
        for detection in detections.detections:
            # should use the rectified boxes size instead ?
            #rectified_box = self.stereo_rectification.rectify_left_bounding_box(box)
            height = detection.bounding_box.max_corner.y - detection.bounding_box.min_corner.y
            
            if height == 0:
                print("Warning: found bounding box with height == 0")
            elif height < 0:
                print("Error: found bounding box with height ==", height)
                box = detection.bounding_box                
                print("box  min corner (x,y), max corner (x,y) == (%i ,%i), (%i ,%i)" % ( 
                      box.min_corner.x, box.min_corner.y,
                      box.max_corner.x, box.max_corner.y ))
                
            #assert height >= 0

            if height >= minimum_height and height <= maximum_height:
                yield detection
        
        return
    
    @staticmethod
    def  _update_score_cumulators(ground_truth, detections, \
                                  false_positives, true_positives, 
                                  min_score, max_score, score_step,
                                  boxes_filter):

        minimal_intersection_over_union = 0.5
        p = minimal_intersection_over_union        
        iou = intersection_over_union_criterion
                                        
        # we copy the input data                 
        detection_boxes = copy(list(boxes_filter(detections)))
        ground_truth_boxes = copy(list(boxes_filter(ground_truth)))
        #ground_truth_boxes_copy = copy(ground_truth_boxes)

        if detection_boxes:
            for ground_truth_box in ground_truth_boxes:                    
                g_bb = ground_truth_box.bounding_box                
                for detection_box in detection_boxes:
                    #print("Testing", ground_truth_box, "with", detection_box)                    
                    d_bb = detection_box.bounding_box
                    if do_overlap(g_bb, d_bb) and iou(g_bb, d_bb, p):
                        # found a match
                        score_index = int((detection_box.score - min_score) / score_step)
                        true_positives[score_index] += 1
                        #ground_truth_boxes_copy.remove(ground_truth_box)
                        detection_boxes.remove(detection_box)
                        #print("Match found")
                        break
                    else:
                        # no match yet
                        continue
                # end of "for each detection"                
            # end of "for each ground truth annotation"
        else:
            pass
        
        # count unmatched boxes
        false_positive_boxes = detection_boxes
        #false_negative_boxes = ground_truth_boxes_copy
        
        for false_positive in false_positive_boxes:
             score_index = int((false_positive.score - min_score) / score_step)
             false_positives[score_index] += 1
                       
        return len(ground_truth_boxes)

    
    @staticmethod
    def _check_ground_truth_detections_match(ground_truth, detections_sequence, name):
        """
        Check that the ground truth and detections are refering to the same dataset
        """

        print("len(ground_truth) == ", len(ground_truth))
        print("len(detections_sequence) == ", len(detections_sequence))

        #assert len(ground_truth) >= len(detections_sequence)
        
        max_index = min(len(ground_truth), len(detections_sequence)) - 1
        middle_index = max_index / 2

        def check_image_name(a,b):
            a = os.path.split(a.image_name)[-1] 
            b = os.path.split(b.image_name)[-1] 
            print("ground truth image name ==", a)
            print("detections image name ==", b)            
            #assert a == b
            return
        
        check_image_name(ground_truth[0], detections_sequence[0])
        check_image_name(ground_truth[middle_index], detections_sequence[middle_index])
        check_image_name(ground_truth[max_index], detections_sequence[max_index])
        
        print() # add an empty line
        return
    
    @staticmethod
    def _get_min_max_scores(detections_sequence):
        
        min_score = float("inf")
        max_score = float("-inf")
                
        for detections in detections_sequence:
            for detection in detections.detections:
                if detection.object_class != Detection.Pedestrian:
                    continue
                score = detection.score
                min_score = min(score, min_score)
                max_score = max(score, max_score)

        assert min_score != float("inf")
        assert max_score != float("-inf")
        
        
        return min_score, max_score
    
    def _compute_detection_rate(self, ground_truth, detections_sequence, name):

        self._check_ground_truth_detections_match(ground_truth, detections_sequence, name)

        min_score, max_score = self._get_min_max_scores(detections_sequence)
        print("%s min, max score == %.3f, %.3f" % (name, min_score, max_score))

        # the number of bins, defines the number of points in the plot        
        #num_score_bins = 20
        num_score_bins = 100
        score_step = (max_score - min_score) / (num_score_bins - 1) 

        minimum_box_height = self.options.minimum_box_height                         
        boxes_filter = lambda b: self._filter_detections_by_height(b, minimum_box_height)
 
        false_positives = [0]*num_score_bins
        true_positives = [0]*num_score_bins
                
        num_images = 0
        num_ground_truth_boxes = 0
        generator = izip(ground_truth, detections_sequence)
        for ground_truth, detections in generator:
            ground_truth_boxes_considered = self._update_score_cumulators(
            ground_truth, detections,
            false_positives, true_positives, 
            min_score, max_score, score_step,
            boxes_filter)
            num_images += 1
            num_ground_truth_boxes += ground_truth_boxes_considered
                    
        
        false_positives.reverse() # in-place reverse            
        true_positives.reverse() # in-place reverse            
                    
        # fppi: false positives per image
        fppi = numpy.cumsum(numpy.array(false_positives, float)/num_images)
        
        # recall in 0 to 1 fraction        
        recall = numpy.cumsum(numpy.array(true_positives, float)/num_ground_truth_boxes)
        
        print("num_images ==", num_images)        
        print("num_ground_truth_boxes ==", num_ground_truth_boxes)        
        
        result = { "fppi":fppi, "recall":recall }
        
        return result
    
    def _compute_plot_detection_rates(self):
        
        results = dict()
        for name, data in self.data_sequences.items():
            print("Computing results for", name)
            result = self._compute_detection_rate(self.ground_truth, data, name)            
            results[name] = result            
 
        return results
    
    def _plot_detection_rates(self):
        
        minimum_box_height = self.options.minimum_box_height
        print("Using minimum_box_height == ", minimum_box_height)
    
        results = self._compute_plot_detection_rates()
        
        print("Plotting %i results." % len(results))
        
        # create the actual plot ---
        
        # create figure
        pylab.figure()
        pylab.gcf().set_facecolor("w") # set white background            
        pylab.grid(True)
        

        colormap = pylab.cm.gist_rainbow
        #colormap = pylab.cm.rainbow    

        index = 0         
        for name, fppi_and_recall  in results.items():
            fppi = fppi_and_recall["fppi"]            
            recall = fppi_and_recall["recall"]
            
            #print("score_thresholds", score_thresholds)            
            #print("fppi", fppi)            
            #print("recall", recall)            

            label = name
            
            linewidth, linestyle = 1.5, "-"
            color_index = index/float(max(1, len(results)-1))
            index += 1
            color = colormap(color_index)

            #linewidth, linestyle = 2, "--"
            #linewidth, linestyle = 1.5, "-."

            #line = 
            pylab.plot(fppi, recall, linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        # end of "plot each curve"
        
        if True:        
            #pylab.xticks(numpy.arange(0.5, 1.5, 0.25))
            pylab.xticks(numpy.arange(0.0, 2, 0.25))
            #pylab.xscale("log")
            #pylab.xlim([1E-2, 2])  
            pylab.xlim([1E-2, 1.5])  
            
        if True:
            pylab.yticks(numpy.arange(0.1, 1.1, 0.1))
            #pylab.yscale("log")
            #pylab.ylim([1E-2, 1])
            pylab.ylim([1E-2, 0.75])

            
        pylab.legend(loc ="lower right", fancybox=True)
        pylab.xlabel("False positives per image")
        pylab.ylabel("Recall")

        dataset_name = self._get_dataset_name()
        
        # this measure also counts the occluded windows
        title = "Recall versus FPPI over " \
                "%s,\nconsidering all windows with height > %i [pixels]" % (dataset_name, minimum_box_height)
        pylab.title(title)
        
        pylab.draw()        
        return
        
        
    def render_example_result(self):

        try:
            import Image, ImageDraw
        except ImportError:
            raise SystemExit("PIL must be installed to render an example result frame")


        # ground truth and image for the example frame --
        #assert len(self.recordings) == 1        
        #recording = self.recordings.values()[0]
        #input_start_frame = int(recording.options["video_input.start_frame"])
        #image_filename_mask = recording.options["video_input.left_filename_mask"]
        #harcoded_image_filename_mask = "/home/rodrigob/data/bahnhof/left/image_%08i_0.png"        
        harcoded_image_filename_mask = "/users/visics/rbenenso/data/bertan_datasets/Zurich/bahnhof/left/image_%08i_0.png"
        
        input_start_frame = 0
        print("Using hardcoded images path", harcoded_image_filename_mask)
        image_filename_mask = harcoded_image_filename_mask
        
        example_result_frame = self.options.example_result_frame
        
        input_example_result_frame = example_result_frame + input_start_frame       
        
        image_filename = image_filename_mask % input_example_result_frame                
        if not os.path.exists(image_filename):
            raise ValueError("could not find the indicated image %s" % image_filename)
        
        image_basename = os.path.basename(image_filename)
        frame_name = "frame_%i" % (example_result_frame + 0)
        
        ground_truth_detections = None
        for detections in self.ground_truth:
            data_filename = os.path.basename(detections.image_name)
            if image_basename == data_filename:
                ground_truth_detections = detections
                break
                
        if not ground_truth_detections:
            raise Exception(
            "Failed to find the ground truth annotation for the recording frame '%s'. " \
            "Are they both refering to the same sequence ?" % image_basename )

        frame_detections = {}
        frame_detections["ground_truth"] = ground_truth_detections
        assert "ground_truth" not in self.data_sequences.keys()
        
        for name in self.data_sequences.keys():
            for detections in self.data_sequences[name]:
                data_filename = os.path.basename(detections.image_name)
                if image_basename == data_filename or frame_name == data_filename:
                    frame_detections[name] = detections
                    break
            if name not in frame_detections.keys():
                raise Exception(
                "Failed to find the recording frame %i in the detections %s. " \
                "Are they both refering to the same sequence ?" % (example_result_frame, name))
        # end of "for each data sequence"
        
        stereo_rectification = self.stereo_rectification
       
        print("example_result_frame", example_result_frame)
        print("video_input.start_frame", input_start_frame)
        
        # render example result image --
        
        #http://www.pythonware.com/library/pil/handbook/imagedraw.htm
        #http://matplotlib.sourceforge.net/examples/pylab_examples/image_demo3.html

        left_image_frame = Image.open(image_filename)
        image_frame = stereo_rectification.rectify_left_image(left_image_frame)
        #image_frame = left_image_frame
        draw = ImageDraw.Draw(image_frame)
        
        minimum_box_height = self.options.minimum_box_height
        boxes_filter = lambda b: self._filter_detections_by_height(b, minimum_box_height)        
        print("Using minimum_box_height == ", minimum_box_height)
        print() # add empty line
         
         
        minimal_intersection_over_union = 0.5
        p = minimal_intersection_over_union        
        iou = intersection_over_union_criterion
                
                
        colormap = pylab.cm.gist_rainbow
        colors = {}

        named_colors = ["red", "blue", "green", "white", "yellow"]
        
        for index in range(len(frame_detections)):
            name = frame_detections.keys()[index]
            if len(frame_detections) <= len(named_colors):         
                colors[name] = named_colors[index]
            else:                
                color_index = index/float(max(1, len(frame_detections)-1)) 
                color = colormap(color_index)
                c = color
                colors[name] = "rgb(%i,%i,%i)" % (int(255*c[0]), int(255*c[1]), int(255*c[2]))
                
            print("%s for %s" % (colors[name], name))
        
        print() # add empty line
        for name, detections in frame_detections.items():
            filtered_detections = list(self._filter_detections_by_height(detections, minimum_box_height))
            print("%s has %i boxes (%i before filtering by height)" % \
                    (name, len(filtered_detections), len(detections.detections)))
            for detection in filtered_detections:
                # all bounding boxes are already rectified (if needed)
                bb = detection.bounding_box
                box = [bb.min_corner.x, bb.min_corner.y, 
                       bb.max_corner.x, bb.max_corner.y ]
                if True and name == "ground_truth":
                    box = [bb.min_corner.x+1, bb.min_corner.y+1, 
                           bb.max_corner.x+1, bb.max_corner.y+1 ]
                draw.rectangle(box, outline=colors[name])
                print("score %.4f\n" % detection.score)
            if True: 
                # compute true and false positives for this frame
                true_positives = 0
                false_positives = 0
            
                # we copy the input data                 
                detection_boxes = copy(list(boxes_filter(detections)))                 
                ground_truth_boxes = copy(list(boxes_filter(ground_truth_detections)))
                ground_truth_boxes_copy = copy(ground_truth_boxes)
        
                num_detections = len(detection_boxes)
                if detection_boxes:
                    for ground_truth_box in ground_truth_boxes_copy:                    
                        g_bb = ground_truth_box.bounding_box                
                        found_match = False
                        for detection_box in detection_boxes:
                            #print("Testing", ground_truth_box, "with", detection_box)                    
                            d_bb = detection_box.bounding_box
                            if do_overlap(g_bb, d_bb) and iou(g_bb, d_bb, p):
                                # found a match
                                found_match = True
                                true_positives += 1
                                #ground_truth_boxes.remove(ground_truth_box)
                                detection_boxes.remove(detection_box)
                                #print("Match found")
                                break
                            else:
                                # no match yet
                                continue
                        # end of "for each detection"                  
                    # end of "for each ground truth annotation"
      
                    # count unmatched boxes
                    false_positive_boxes = detection_boxes
                    false_positives = len(false_positive_boxes)
        
                else:
                    pass
                print("%s found %i true positives, %i false positives " \
                     "(out of %i detections and %i ground truth annotations)"
                     % (name, true_positives, false_positives, num_detections, len(ground_truth_boxes_copy)) )

        output_filename = None # self.options.output_filename
        if output_filename: 
            print("Creating output image:", output_filename)
            assert not os.path.exists(output_filename)
            image_frame.save(output_filename)
        else:
             image_frame.show()

        return        
        
# end of class DetectionsEvaluationApplication

        
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
   
    application = DetectionsEvaluationApplication()
    application.run()

