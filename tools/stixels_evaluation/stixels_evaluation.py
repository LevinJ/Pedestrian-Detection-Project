#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for the stixel world evaluation.
"""

from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../helpers")
 
from stixels_pb2 import Stixels, Stixel
from data_sequence import DataSequence
from idl_parsing import open_idl_file, IdlLineData

import types
import os, os.path
import re
from itertools import izip

from optparse import OptionParser
import progressbar

from copy import copy, deepcopy

from stereo_rectification import StereoRectification

try:
    import Image, ImageDraw
except ImportError:
    raise SystemExit("PIL must be installed to run this application")

import numpy, pylab


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
        

#Recording = namedtuple("Recording", "directory frontmost_bboxes_only options")
Recording = Bunch



   
def area(rect):
    "rect is expected to be a tuple of the type (x1, y1, x2, y2) "
    w = abs(rect[2] - rect[0])
    h = abs(rect[3] - rect[1])

    return float(w*h)
        
def overlapping_area(a, b):
   """
   a and b are expected to be tuples of the type (x1, y1, x2, y2)
   code adapted from http://visiongrader.sf.net
   """       
   w = min(a[2], b[2]) - max(a[0], b[0])
   h = min(a[3], b[3]) - max(a[1], b[1])
   if w < 0 or h < 0:
        return 0
   else:
        return float(w * h)
        
def do_overlap(a,b):
    """
    a and b are expected to be tuples of the type (x1, y1, x2, y2)
    code adapted from http://visiongrader.sf.net
    """       
    return (a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])

   
def intersection_over_union_criterion(detection, ground_truth, p):
    """
    p is the maximum ratio allowed between the intersection and the union.
    Set it to 0.5 to have 50%.
    """
    b1 = detection; b2 = ground_truth
    inter = overlapping_area(b1, b2)
    #assert inter > 0 # in our setup this should always be true
    union = area(b1) + area(b2) - inter
    intersection_over_union =  (inter / union)
    #print("IoU", intersection_over_union, b1, b2)
    
    return intersection_over_union > p
       
        

    
def count_errors(ground_truth, detection_threshold, detection_boxes):
    """
    Count the errors in a frame
    """    
    false_positives = 0
    false_negatives = 0

    if type(detection_boxes) is types.GeneratorType:
        detection_boxes = list(detection_boxes)

    if not (type(detection_boxes) is list):
        detection_boxes = detection_boxes.bounding_boxes

    # we only consider detections with a score higher than the threshold
    detection_boxes = [box for box in detection_boxes if box[4] > detection_threshold] 
    
    minimal_intersection_over_union = 0.5
    p = minimal_intersection_over_union
    
    iou = intersection_over_union_criterion
        
    if type(ground_truth) is list:
        ground_truth_boxes = copy(ground_truth)
    else:
        ground_truth_boxes = copy(ground_truth.bounding_boxes)
 
    #print(len(ground_truth_boxes), "ground_truth_boxes", ground_truth_boxes)
    #print(len(detection_boxes), "detection_boxes", detection_boxes)

    ground_truth_boxes_copy = copy(ground_truth_boxes)
    
    for ground_truth_box in ground_truth_boxes_copy:
        for detection_box in detection_boxes:
            #print("Testing", ground_truth_box, "with", detection_box)
            if do_overlap(ground_truth_box, detection_box) and \
                iou(ground_truth_box, detection_box, p):
                    # found a match
                    ground_truth_boxes.remove(ground_truth_box)
                    detection_boxes.remove(detection_box)
                    #print("Match found")
                    break
            else:
                # no match
                continue
        
    #print(len(ground_truth_boxes), "unmatched ground_truth_boxes", ground_truth_boxes)
    #print(len(detection_boxes), "unmatched detection_boxes", detection_boxes)
    
    # count unmatched boxes
    false_positives = len(detection_boxes)
    false_negatives = len(ground_truth_boxes)
    
    return false_positives, false_negatives


    
def recordings_stixels_data_sequences_generator(recordings):
        stixel_data_dict = dict()
        while True:                    
            for name, value in recordings.items():
                stixel_data_dict[name] = next(value.stixels_data_sequence_generator)
            yield stixel_data_dict
            
            
def open_data_sequences(recordings):
    for name, recording in recordings.items():
        recording.stixels_data_sequence_generator = open_data_sequence(recording)
    return
    
def open_data_sequence(recording):
        
        # find the data sequence file --
        stixels_data_files = [ 
            x for x in os.listdir(recording.directory) 
            if x.endswith("sequence") and x.startswith("stixels") ]
                
        if len(stixels_data_files) > 1:
            print("Candidate stixel data sequence files are :", stixels_data_files)
            raise ValueError("recording_directory contains more than one candidate data sequence")
        elif len(stixels_data_files) == 0:
            raise ValueError("recording_directory contains no data sequence file")            

        stixels_data_file = os.path.join(recording.directory, 
                                         stixels_data_files[0])
     
        assert os.path.exists(stixels_data_file)
        
        the_data_sequence = DataSequence(stixels_data_file, Stixels)
        
        def data_sequence_reader(data_sequence):    
            while True:
                data = data_sequence.read()
                if data is None:
                    raise StopIteration
                else:
                    yield data
        
        return data_sequence_reader(the_data_sequence)    
    
class StixelsEvaluationApplication:
    
    
    def __init__(self):
        return
        
    def run(self, options=None):

        if not options:
            self.parse_arguments()
        else:
            self.options = options
            self.should_render_example_result = False
        self.recordings = self.options.recordings
        
        self.open_recordings_options()
        self.open_ground_truth_annotations()
        self.open_data_sequences()
        self.open_stereo_rectification()

        if self.should_render_example_result:
            self.render_example_result()
        else:
            self.render_evaluation_graphs()

        print("End of game. Have a nice day !")
        return

    def parse_arguments(self):
        
            
        parser = OptionParser()
        parser.description = \
            "This program takes as input a recording directory created by stixels_world, " \
            "a ground truth file and computes an evaluation graph of the extimated stixels"
    
        parser.add_option("-g", "--ground_truth", dest="ground_truth_path",
                           metavar="IDL_FILE", type="string",
                           help="path to the ground truth idl file")
        parser.add_option("-r", "--recording", dest="recording_directory", 
                          metavar="DIRECTORY", type="string",
                          help="path to the recording directory containing the stixels data sequence")
        
        parser.add_option("-d", "--detections", dest="detections_path", 
                          metavar="DETECTIONS_FILE", type="string",
                          help="path to the detections idl file (with detection confidence)")
                          
        parser.add_option("--references_detections", dest="references_detections_path", 
                          metavar="REFERENCES_DETECTIONS_FILE", type="string",
                          help="path to the detections idl file (with detection confidence)")
                      
                      
        parser.add_option("-f", "--frame", dest="example_result_frame", 
                          metavar="NUMBER", type="int", default=-1,
                          help="if frame is bigger than -1 it will be used to render an example result frame [default: %default]")

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
        
    
        (options, args) = parser.parse_args()
        #print (options, args)
    
        self.should_render_example_result = options.example_result_frame > -1
    
        if not self.should_render_example_result:
            if not (options.ground_truth_path and options.recording_directory):
                parser.error("'ground_truth' and 'recording' options are required to run this program")
        elif not options.recording_directory:
            parser.error("'recording' option is required to run this program")
    
        if options.ground_truth_path:
            if not os.path.exists(options.ground_truth_path):
                parser.error("Could not find the ground_truth file")

        if not os.path.exists(options.recording_directory):
            parser.error("Could not find the recording directory")
                        
        if not os.path.isdir(options.recording_directory):
            parser.error("The --recording option expects to receive a directory, not a file")
    
        
    
        options.recordings = dict()
        options.recordings["the_recording"] = Recording(
            directory = options.recording_directory, 
            frontmost_bboxes_only = False)
    
        self.options = options    
    
        return # end of parse_arguments


    def open_ground_truth_annotations(self):
        
        if self.options.ground_truth_path:
            self.ground_truth_data_generator = open_idl_file(self.options.ground_truth_path)
        else:
            self.ground_truth_data_generator = None
            
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
            
    def open_data_sequences(self):
        return open_data_sequences(self.recordings)
        
  
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

    def open_stereo_rectification(self):
        
        a_recording = self.recordings.values()[0] # one of the recordings
        calibration_filename = self._get_recordings_option("video_input.calibration_filename")
        
        calibration_filename = os.path.join(a_recording.directory, "..", calibration_filename)
        self.stereo_rectification = StereoRectification(calibration_filename)

        return

    def render_example_result(self):

        assert len(self.recordings) == 1

        # get stixels, ground truth and image for the example frame --
        
        
        recording = self.recordings.values()[0]
        input_start_frame = int(recording.options["video_input.start_frame"])
        image_filename_mask = recording.options["video_input.left_filename_mask"]
        # /home/rodrigob/work/data/zurich/bahnhof/video/image_%08i_0.png        
        
        example_result_frame = self.options.example_result_frame
        
        input_example_result_frame = example_result_frame + input_start_frame       
        
        image_filename = image_filename_mask % input_example_result_frame                
        if not os.path.exists(image_filename):
            raise ValueError("could not find the indicated image %s" % image_filename)
        
        image_basename = os.path.basename(image_filename)
        
        ground_truth_idl_data = None
        if  self.ground_truth_data_generator:
            for idl_data in self.ground_truth_data_generator:
                data_filename = os.path.basename(idl_data.filename)
                if image_basename == data_filename:
                    ground_truth_idl_data = idl_data
                    break
        else:
            ground_truth_idl_data = IdlLineData("empty_idl_data", [])
            
        if not ground_truth_idl_data:
            raise Exception(
            "Failed to find the ground truth annotation for the recording frame. " \
            "Are they both refering to the same sequence ?")

        detections_idl_data = None
        if self.options.detections_path:
            # retrieve detections idl
            self.detections_data_generator = open_idl_file(self.options.detections_path)
    
            for idl_data in self.detections_data_generator:
                data_filename = os.path.basename(idl_data.filename)
                if image_basename == data_filename:
                    detections_idl_data = idl_data
                    break
                
            if not detections_idl_data:
                raise Exception(
                "Failed to find the detections for the recording frame. " \
                "Are they both refering to the same sequence ?")
        

        stereo_rectification = self.stereo_rectification

        stixels_data_offset = -1 # caused because the preprocessing 
        stixels_data = None
        reached_frame_index = 0        
        
        for data in recording.stixels_data_sequence_generator:
            if reached_frame_index == (example_result_frame + stixels_data_offset):
                stixels_data = data
                break
            reached_frame_index += 1
            
        if not stixels_data:
                   raise Exception(
            "Failed to find the stixel data corresponding to frame %i (only reached frame %i). " \
            "Was the program execution stopped before reaching that frame ?" \
            % (example_result_frame, reached_frame_index) )
       
        print("example_result_frame", example_result_frame)
        print("video_input.start_frame", input_start_frame)
        print("ground_truth_idl_data.filename", ground_truth_idl_data.filename)
        print("stixels_data.image_name", stixels_data.image_name)

        # render example result image --
        
        #http://www.pythonware.com/library/pil/handbook/imagedraw.htm
        #http://matplotlib.sourceforge.net/examples/pylab_examples/image_demo3.html

        left_image_frame = Image.open(image_filename)
        image_frame = stereo_rectification.rectify_left_image(left_image_frame)
        #image_frame = left_image_frame
        draw = ImageDraw.Draw(image_frame)
        
        stixels_vertical_margin = 30 # just for presentation
        minimum_box_height = self.options.minimum_box_height
        print("Using minimum_box_height == ", minimum_box_height)
        
        detections_boxes = None
        ground_truth_boxes = None
        
        if detections_idl_data:
            detections_boxes = list(self._filter_bounding_boxes_by_height(detections_idl_data.bounding_boxes, minimum_box_height))                                    
            for box in detections_boxes:                
                box_is_allowed = self._box_is_allowed(box, stixels_data, stixels_vertical_margin)                
                rectified_box = stereo_rectification.rectify_left_bounding_box(box)
                if box_is_allowed:
                    draw.rectangle(rectified_box, outline="white")        
                else:
                    draw.rectangle(rectified_box, outline="grey")       
                            
        if ground_truth_idl_data:
            ground_truth_boxes = list(self._filter_bounding_boxes_by_height(ground_truth_idl_data.bounding_boxes, minimum_box_height))            
            for box in ground_truth_boxes:
                rectified_box = stereo_rectification.rectify_left_bounding_box(box)
                #rectified_box = box
                draw.rectangle(rectified_box, outline="red")

        if detections_idl_data and  ground_truth_idl_data:     
            detection_threshold = 0
            filtered_detections = self._filter_bounding_boxes(detections_boxes, stixels_data, stixels_vertical_margin)

            false_positives, false_negatives = count_errors(ground_truth_boxes, detection_threshold, detections_idl_data)
            print("Non filtered detections: false_positives %i, false_negatives %i" %(false_positives, false_negatives))
            #print("\n")
            false_positives, false_negatives = count_errors(ground_truth_boxes, detection_threshold, filtered_detections)
            print("Filtered detections: false_positives %i, false_negatives %i" %(false_positives, false_negatives))
            
            
        for stixel in stixels_data.stixels:
            # stixels are estimated on the rectified left image,
            # so we can write then directly
            if stixel.type == Stixel.Occluded:
                draw.point([stixel.x, stixel.bottom_y], fill="darkblue")
                draw.point([stixel.x, stixel.top_y], fill="darkblue")                
            else:
                draw.point([stixel.x, stixel.bottom_y+2], fill="yellow")
                draw.point([stixel.x, stixel.bottom_y+1], fill="yellow")
                draw.point([stixel.x, stixel.bottom_y], fill="yellow")
                draw.point([stixel.x, stixel.top_y], fill="orange")
                draw.point([stixel.x, stixel.top_y-1], fill="orange")
                draw.point([stixel.x, stixel.top_y-2], fill="orange")
            
       
        
        if self.options.output_filename:
            print("Creating output image:", self.options.output_filename)
            assert not os.path.exists(self.options.output_filename)
            image_frame.save(self.options.output_filename)
        else:
             image_frame.show()

        return
        
    def render_evaluation_graphs(self):
        """
        Compute cumulative histograms and plot them
        """
        
        #should_compute_error_histograms = True   
        should_compute_error_histograms = (not hasattr(self.options, "detections_path")) or (not self.options.detections_path)
        
        if should_compute_error_histograms:
            print("Computing error histograms. Please wait...")
            self._compute_error_histograms()
            print("Plotting results")
            if len(self.recordings) <= 2 and False:
                self._plot_error_histograms()
            else:
                self._plot_cumulative_error_histograms()
        
        if hasattr(self.options, "detections_path") and self.options.detections_path:
            self._plot_detection_rates()
        
        pylab.show()
        return

                
        
    def _get_recordings_stixels_data_sequence_generator(self):
        
        return recordings_stixels_data_sequences_generator(self.recordings)
    

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
    
    def _compute_error_histograms(self):
        
        for recording in self.recordings.values():
            recording.stixel_top_error_histogram = dict()
            recording.stixel_bottom_error_histogram = dict()
            recording.num_occluded_stixel_bounding_boxes = 0 
            recording.num_non_occluded_stixel_bounding_boxes = 0                     
            
        self.num_out_of_image_bounding_boxes = 0
        self.num_accounted_frames = 0
           
        stereo_rectification = self.stereo_rectification
        
        max_num_frames = min(self._get_max_num_frames(), self._get_num_frames())
        
        progress_bar_message = progressbar.SimpleMessage()
        progress_bar_widgets = [progressbar.Percentage(), progress_bar_message, progressbar.Bar(), progressbar.ETA()]
        progress_bar = progressbar.ProgressBar(widgets = progress_bar_widgets, maxval=(max_num_frames +1))        
        
        minimum_box_height = self.options.minimum_box_height        
        print("Using minimum_box_height == ", minimum_box_height)

        # we skip the first ground truth data, 
        # since the first frame is skip from the processing due to
        # preprocessing needs
        #self.ground_truth_data_generator.next()
        
        generator = izip(self.ground_truth_data_generator, 
                        self._get_recordings_stixels_data_sequence_generator())
        for ground_truth_idl_data, recordings_stixels_data in generator:

            if self.num_accounted_frames > max_num_frames:
                print("Reached indicated max number of frames (%i)" % max_num_frames )
                break

            self.num_accounted_frames += 1
            #print(ground_truth_idl_data.filename, stixels_data.image_name)
            progress_bar_message.message = " " + ground_truth_idl_data.filename
            progress_bar.update(self.num_accounted_frames)            
            
            
            for recording_name, stixels_data in recordings_stixels_data.items():
                
                recording = self.recordings[recording_name]

                ground_truth_boxes = ground_truth_idl_data.bounding_boxes             
                ground_truth_boxes = self._filter_bounding_boxes_by_height(ground_truth_boxes, minimum_box_height)
                
                for box in ground_truth_boxes:
                    if len(box) != 4:
                        print(ground_truth_idl_data.filename, stixels_data.image_name)
                        print("ERROR: received a weird bounding box at frame %i, box == %s" % (self.num_accounted_frames, box))
                        continue
                    rectified_box = stereo_rectification.rectify_left_bounding_box(box)
                    box_top = rectified_box[1]
                    box_bottom = rectified_box[3]
                    box_center = int((rectified_box[2] - rectified_box[0]) / 2 + rectified_box[0])
    
                    found_stixel = False               
                    for stixel in stixels_data.stixels:
                        if stixel.x == box_center:
                            found_stixel = True
                            if stixel.type == Stixel.Occluded:
                                recording.num_occluded_stixel_bounding_boxes += 1
                            else:
                                recording.num_non_occluded_stixel_bounding_boxes += 1
                                
                            top_error = int(box_top - stixel.top_y)
                            bottom_error = int(box_bottom - stixel.bottom_y)
                            
                            if True:
                                # FIXME THERE IS A NASTY BUG, this is a work around
                                if top_error > 1000: top_error = 1000
                                if top_error < -1000: top_error = -1000
                                if bottom_error > 1000: bottom_error = 1000
                                if bottom_error < -1000: bottom_error = -1000
                                
                            
                            if recording.stixel_top_error_histogram.has_key(top_error):
                                recording.stixel_top_error_histogram[top_error] += 1
                            else:
                                recording.stixel_top_error_histogram[top_error] = 1
                            
                            if recording.stixel_bottom_error_histogram.has_key(bottom_error):
                                recording.stixel_bottom_error_histogram[bottom_error] += 1
                            else:
                                recording.stixel_bottom_error_histogram[bottom_error] = 1        
                                
                            break
                        # end of "found stixel"
                    # end of "for each stixel"
                    if not found_stixel:
                       
                        stixels_x = [s.x for s in stixels_data.stixels]
                        max_stixels_x = max(stixels_x)
                        if box_center > max_stixels_x:
                            #print("box_center > max_stixels_x => %i > %i " %(box_center, max_stixels_x))
                            #print("Could not find stixel covering a given bounding box. "
                            #            "This should never happen")
                            self.num_out_of_image_bounding_boxes += 1
                            continue
                        else:
                            print(ground_truth_idl_data.filename, stixels_data.image_name)
                            print("box_center == ", box_center)
                            print("stixels.x == ", stixels_x)
                            
                            raise Exception("Could not find stixel covering a given bounding box. "
                                            "This should never happen")
                    #end of "if not found_stixel"
                        
                # end of "for each box in the image"
            # end of "for each item in recordings_stixels_data"
        # end of "for each processed image"
        progress_bar.finish()
                    
        for recording in self.recordings.values():
            assert recording.num_non_occluded_stixel_bounding_boxes > 0
            
        return

    def _dictionary_histogram_to_list_histogram(self, dict_histogram, max_bin, min_bin):
        
        #print("max_bin, min_bin == ", max_bin, min_bin)
        num_bins = (max_bin - min_bin) + 1
        print("num_bins == ", num_bins)
        histogram = [0]*num_bins
        
        for key, value in dict_histogram.items():
            bin_index = key - min_bin
            histogram[bin_index] = value
        
        return histogram

    def _get_dataset_name(self):
                
        calibration_filename = self._get_recordings_option("video_input.calibration_filename")
        m = re.search(".*stereo_calibration_(.*)\.proto.*", calibration_filename)
        sequence_name = m.group(1)
        dataset_name = "%i frames of sequence %s" % (
        self.num_accounted_frames, sequence_name.capitalize())

        return dataset_name


    def _print_recordings_info(self):
        
        for name, recording in self.recordings.items():
            self._print_recording_info(name, recording)
        
        return


    def _print_recording_info(self, name, recording):
        
        print("Recording '%s' informations:" % name)
            
        total_bounding_boxes = float(recording.num_occluded_stixel_bounding_boxes + recording.num_non_occluded_stixel_bounding_boxes) 
        
        percent_occluded_stixel_bounding_boxes = \
        recording.num_occluded_stixel_bounding_boxes / total_bounding_boxes
        
        print("\ttotal_considered_bounding_boxes == ", total_bounding_boxes)   
        print("\tnum_out_of_image_bounding_boxes (box_center > max_stixels_x) == ", self.num_out_of_image_bounding_boxes)
        print("\tnum_occluded_stixel_bounding_boxes == ", 
              recording.num_occluded_stixel_bounding_boxes)
        print("\tnum_non_occluded_stixel_bounding_boxes == ", 
              recording.num_non_occluded_stixel_bounding_boxes)            
        print("\tpercent_occluded_stixel_bounding_boxes == %.2f%%" % 
              (percent_occluded_stixel_bounding_boxes*100))
        

        return

    def _plot_error_histograms(self):

        name, recording = self.recordings.items()[0]
        self._print_recording_info(name, recording)
        
        
        min_error = float("inf")
        max_error = -float("inf")
        for recording in self.recordings.values():
            min_error = min(min_error,
                            min(recording.stixel_top_error_histogram.keys()),
                            min(recording.stixel_bottom_error_histogram.keys()))
           
            max_error = max(max_error,
                            max(recording.stixel_top_error_histogram.keys()), 
                            max(recording.stixel_bottom_error_histogram.keys()))
        #end of "for each recording"

        for recording in self.recordings.values():
            # convert dictionaries to list
            recording.stixel_top_error_histogram = \
                 self._dictionary_histogram_to_list_histogram(recording.stixel_top_error_histogram,
                                                        max_error, min_error)
            recording.stixel_bottom_error_histogram = \
                 self._dictionary_histogram_to_list_histogram(recording.stixel_bottom_error_histogram,
                                                        max_error, min_error)
        #end of "for each recording"

        bins = range(min_error, max_error + 1)
        
        normal_plot = False or self.recordings.items() > 1
        
        # create figure
        pylab.figure(0)
        pylab.gcf().set_facecolor("w") # set white background
        pylab.grid(True)
        #pylab.ylim(0, 1.05)

        if normal_plot:
            for name, recording in self.recordings.items():
                line = pylab.plot(bins, recording.stixel_top_error_histogram, 
                                  "-.", linewidth=1.5, label= name + " top error")
                pylab.plot(bins, recording.stixel_bottom_error_histogram, 
                                  "--", color = line[0].get_color(),
                                         linewidth=1.5, label= name + " bottom error")
            #end of "for each recording"
        else:
            # not normal plotting
            name0, recording0 = self.recordings.items()[0]
            name1, recording1 = self.recordings.items()[1]
            pylab.plot(bins, recording0.stixel_bottom_error_histogram, 
                                  "b-.",
                                         linewidth=1.5, label= "bottom error")
            pylab.plot(bins, recording0.stixel_top_error_histogram, 
                                  "r--", linewidth=1.5, label= name0 + " top error")
            pylab.plot(bins, recording1.stixel_top_error_histogram, 
                                  "g--", linewidth=1.5, label= name1 + " top error")
                                         
                                     
        xticks_steps = 10        
        #pylab.xticks(numpy.arange(min_error, max_error, xticks_steps))    
        pylab.xticks(numpy.arange(-100, 100, xticks_steps))    
        pylab.xlim([-100, 100])                             
        
        #yticks_steps = 0.1
        #pylab.yticks(numpy.arange(0, 1, yticks_steps))  
        
        pylab.legend(loc ="upper right", fancybox=True)
        pylab.xlabel("Vertical error in pixels")
        pylab.ylabel("Number of bounding boxes")
        
        dataset_name = self._get_dataset_name()
        
        title = "Stixels vertical error over over\n" \
                "%s, considering windows with height > %i [pixels]" % \
                (dataset_name, self.options.minimum_box_height)
        pylab.title(title)

        pylab.draw()
        
        return        
        
        
    def _convert_error_histogram_to_absolute_error_histogram(self, histogram):
        
        absolute_histogram = histogram.copy()

        for key, value in histogram.items():
            if key < 0:
                if absolute_histogram.has_key(-key):
                    absolute_histogram[-key] += value
                else:
                    absolute_histogram[-key] = value
                absolute_histogram.pop(key) # remove the negative keys
            else:
                # key >= 0
                continue
        
        return absolute_histogram
        
    def _plot_cumulative_error_histograms(self):

        min_error = 0
        max_error = 0
        for recording in self.recordings.values():
            recording.absolute_top_error = \
                self._convert_error_histogram_to_absolute_error_histogram(
                    recording.stixel_top_error_histogram)      
            recording.absolute_bottom_error = \
                self._convert_error_histogram_to_absolute_error_histogram(
                    recording.stixel_bottom_error_histogram)
            
            max_error = max(max_error,
                            max(max(recording.absolute_top_error.keys()), 
                            max(recording.absolute_bottom_error.keys())))
        # end of "for each recording"

        for recording in self.recordings.values():
            # convert dictionaries to list
            recording.stixel_top_error_histogram = \
                 self._dictionary_histogram_to_list_histogram(recording.absolute_top_error,
                                                        max_error, min_error)
            del recording.absolute_top_error # free some memory
            recording.stixel_bottom_error_histogram = \
                 self._dictionary_histogram_to_list_histogram(recording.absolute_bottom_error,
                                                        max_error, min_error)
            del recording.absolute_bottom_error # free some memory
                                            
            recording.stixel_top_error_cumsum = numpy.cumsum(recording.stixel_top_error_histogram, dtype="float")
            del recording.stixel_top_error_histogram  # free some memory

            recording.stixel_bottom_error_cumsum = numpy.cumsum(recording.stixel_bottom_error_histogram, dtype="float")
            del recording.stixel_bottom_error_histogram  # free some memory
            
            assert recording.stixel_top_error_cumsum[-1] == recording.stixel_bottom_error_cumsum[-1]
            
            num_bounding_boxes = recording.num_occluded_stixel_bounding_boxes + \
                                 recording.num_non_occluded_stixel_bounding_boxes
            recording.stixel_top_error_cumsum /= num_bounding_boxes
            recording.stixel_bottom_error_cumsum /= num_bounding_boxes
        # end of "for each recording"
        
        # create figure
        pylab.figure()
        pylab.gcf().set_facecolor("w") # set white background            
        
        
        bins = range(min_error, max_error + 1)
        
        num_recordings = len(self.recordings.keys())
        
        # for more colors
        # http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html        
        if num_recordings > 2:
            #colormap = pylab.cm.Accent
            colormap = pylab.cm.gist_rainbow
            color_index = [ i/float(max(1, num_recordings-1)) for i in xrange(num_recordings) ]
            #top_postfix = " top error"
        else:
            #colormap = pylab.cm.Spectral
            #colormap = pylab.cm.cool
            #colormap = pylab.cm.jet
            colormap = pylab.cm.hsv
            #top_postfix = " error"
            color_index = [ 0.0, 0.5 ] 
            #color_index = [ 0.25, 0.75 ] 
            #color_index = [ 0.33, 0.66 ] 
        
        defined_colors = {}
        defined_colors["ours"] = "#2a00ff" # blue       
        defined_colors["simple sad"] = "#0fff0f" # green       
        defined_colors["csbp"] = "#ff163b" # red       
        

        estimated_height_name = "estimated height "
        fixed_height_name = "fixed height "
        
        def get_cropped_name(name):
            "Helper to get the recordings in the proper order and naming"
            if name.startswith(fixed_height_name):
                cropped_name = name[len(fixed_height_name):] 
            elif name.startswith(estimated_height_name):
                cropped_name = name[len(estimated_height_name):]
            else:
                cropped_name = name
            return cropped_name
        
        recording_items = [(get_cropped_name(name), name, recording) for name, recording in self.recordings.items()]
        if self.recordings is dict:        
            recording_items.sort() # will sort by cropped_name
        else:
            # no need to sort if we use an OrderedDict
            pass
        print([name for cropped_name, name, recording in recording_items])
        
        #bottom_postfix = " bottom error"
        plotted_bottom_methods = []
        for index, (cropped_name, name, recording) in izip(xrange(num_recordings), recording_items):            
            top_error_line_type, bottom_error_line_type = "-", "-" # "-.", "--"
            color = colormap(color_index[index])   
            if name.startswith(fixed_height_name):
                bottom_name = cropped_name #+ bottom_postfix
                top_name = cropped_name + " fixed height"
                top_error_line_type = "--"
            elif name.startswith(estimated_height_name):
                bottom_name = cropped_name #+ top_postfix
                top_name = cropped_name + " estimated height"
                top_error_line_type = "-"
            else:
                bottom_name = name
                top_name = name
                
            if cropped_name in defined_colors.keys():
                color = defined_colors[cropped_name]                    
                
            pylab.subplot(122)
            line = pylab.plot(
                bins, recording.stixel_top_error_cumsum, 
                top_error_line_type, color = color,
                linewidth=1.5, 
                label= top_name )
            if cropped_name not in plotted_bottom_methods:
                pylab.subplot(121)
                pylab.plot(
                    bins, recording.stixel_bottom_error_cumsum, 
                    bottom_error_line_type, color = line[0].get_color(),
                    linewidth=1.5, 
                    label= bottom_name )
                plotted_bottom_methods.append(cropped_name)
        # end of "for each item in recordings"

        xticks_steps = 10
        #xticks = numpy.arange(min_error, max_error, xticks_steps)
        #xticks = numpy.arange(min_error, min(101, max_error), xticks_steps)
        xticks = numpy.arange(min_error, min(71, max_error), xticks_steps)
        #xticks = range(min_error, 60, 10) + range(60, max_error, 20)                                 


        #yticks_steps = 0.05
        yticks_steps = 0.1
        #yticks = numpy.arange(0.001, 1.001, yticks_steps)
        yticks = numpy.arange(0.00, 1.001, yticks_steps)
        #linthresh = 0.2; sub = range(0, 10)
        #pylab.yscale("log", linthreshy = linthresh, subsy=sub)        
        
        #x_limits = [0, 600]        
        #x_limits = [0, 100]        
        x_limits = [0, 50]        
        #x_limits = [0, 30]        

        subplots = [(121, "Stixels bottom error"), (122, "Stixels top error")]
        for subplot_id, subplot_title in subplots:
            pylab.subplot(subplot_id)
            pylab.grid(True)
            pylab.xticks(xticks)
            pylab.yticks(yticks)   
            pylab.xlim(x_limits)  
            pylab.legend(loc ="lower right", fancybox=True)
            pylab.xlabel("Absolute vertical error in pixels")
            pylab.ylabel("Fraction of bounding boxes")
            pylab.title(subplot_title)
        
        dataset_name = self._get_dataset_name()
                
        title = "Stixels vertical cumulative absolute error over\n%s" % dataset_name
        if self.options.minimum_box_height > 0:
            title += ", considering windows with height > %i [pixels]" % self.options.minimum_box_height
        pylab.suptitle(title)

        pylab.draw()
        
        self._print_recordings_info()
        return      
       
    def _box_is_allowed(self, box, stixels_data, stixels_vertical_margin):
        """
        Check if a box is allowed given the stixels, or not.
        box is a (x1, y1, x2, y2) tuple
        """
        
        margin = stixels_vertical_margin
    

        if margin == -2:
            raise "Support for ground_plane_corridor not yet implemented"
    
        # if the margin is negative, no filtering is done
        if margin < 0:
            return True
        
        rectified_box = self.stereo_rectification.rectify_left_bounding_box(box)
        box_center = int(((rectified_box[2] - rectified_box[0]) / 2) + rectified_box[0])
        
        for stixel in stixels_data.stixels:
            if stixel.x == box_center:
                top_margin = abs(stixel.top_y - rectified_box[1])
                bottom_margin = abs(stixel.bottom_y - rectified_box[3])
                #print("stixels_vertical_margin", margin)
                #print("top_margin", top_margin)
                #print("bottom_margin", bottom_margin)
                if top_margin > margin or bottom_margin > margin:
                    # non valid, we should skip this box
                    return False
                else:
                    # valid, we allow this box
                    return True
            else:
                # not the corresponding stixel
                continue
        # end of "for each stixel"
        
        #print("box_center", box_center, "not found in the stixels_data")
         
        # if no stixel covers the box center, then we assume it is not ok        
        return False
            
    def _adjust_box_score(self, box, top_margin, bottom_margin,  stixels_vertical_margin):
        
        # 0 if perfect match -> 2x original score
        # 1 would be "almost reject" -> 0.5x original score               
        margin_fraction = (top_margin + bottom_margin)*0.5 / stixels_vertical_margin

        old_score = box[4]
                        
        perfect_score_scaling = 2.0
        almost_reject_score_scaling = 0.5                
        score_scaling = (perfect_score_scaling - almost_reject_score_scaling)*max(0, 1 - margin_fraction) + almost_reject_score_scaling
        
        new_score = old_score * score_scaling
        adjusted_box = tuple(list(box[:4]) + [new_score])
        
        return adjusted_box
    
    def _adjust_score_using_stixels(self, box, stixels_data, stixels_vertical_margin):
        """
        Adapt the score of a box based on the fit with the stixels estimate
        box is a (x1, y1, x2, y2, score) tuple
        If the box is accepted will return a modified box
        If the box is rejected will return None
        """
        
        margin = stixels_vertical_margin
    
        assert margin > 0    
    
        rectified_box = self.stereo_rectification.rectify_left_bounding_box(box)
        box_center = int(((rectified_box[2] - rectified_box[0]) / 2) + rectified_box[0])
        
        for stixel in stixels_data.stixels:
            if stixel.x == box_center:
                top_margin = abs(stixel.top_y - rectified_box[1])
                bottom_margin = abs(stixel.bottom_y - rectified_box[3])
                
                if top_margin > margin or bottom_margin > margin:
                    # non valid, we should skip this box
                    return None
                    
                adjusted_box = self._adjust_box_score(box, top_margin, bottom_margin, margin)
                return adjusted_box
            else:
                # not the corresponding stixel
                continue
        # end of "for each stixel"
        
        #print("box_center", box_center, "not found in the stixels_data")
         
        # if no stixel covers the box center, then we return the original box
        #return box
        
        # if no stixel covers the box center, then we assume it is not ok        
        return None
        
    def _adjust_score_using_ground_plane(self, box, stixels_data, stixels_vertical_margin):
        """
        Adapt the score of a box based on the fit with the stixels estimate
        box is a (x1, y1, x2, y2, score) tuple
        If the box is accepted will return a modified box
        If the box is rejected will return None
        """
        
        margin = stixels_vertical_margin
    
        if not stixels_data.HasField("ground_top_and_bottom"):
            raise "Could not find the expected ground_top_and_bottom field in the stixels sequence data"
                    
        rectified_box = self.stereo_rectification.rectify_left_bounding_box(box)
        #box_x_center = int(((rectified_box[2] - rectified_box[0]) / 2) + rectified_box[0])
        #box_y_center = int(((rectified_box[3] - rectified_box[1]) / 2) + rectified_box[1])
        box_top_y = rectified_box[1]        
        box_bottom_y = rectified_box[3]
                
        #print("box_top_y", box_top_y, "box_bottom_y", box_bottom_y)        
        assert box_top_y <= box_bottom_y     
        adjusted_box = None                
                
        for top_and_bottom in stixels_data.ground_top_and_bottom.top_and_bottom:
            
              #print("top_y", top_and_bottom.top_y, "bottom_y", top_and_bottom.bottom_y)
              assert top_and_bottom.top_y < top_and_bottom.bottom_y     
              top_margin = abs(top_and_bottom.top_y - box_top_y)
              bottom_margin = abs(top_and_bottom.bottom_y - box_bottom_y)
              
              if top_margin <= margin and bottom_margin <= margin:
                    #print("top_margin", top_margin, "bottom_margin", bottom_margin)
              
                    new_box = self._adjust_box_score(box, top_margin, bottom_margin, margin)

                    if not adjusted_box:
                        adjusted_box = new_box
                    elif adjusted_box[4] < new_box[4]:
                        # new_box has better score
                        adjusted_box = new_box
                    else:
                        # keep previous adjusted_box
                        pass
              else:
                  # keep searching
                  pass
        # end of "for each top_and_bottom"
        
        # if the box does not fit into the ground plane corridor, then we assume it is not ok        
        return adjusted_box
        
    def _adjust_score(self, box, stixels_data, stixels_vertical_margin):
        """
        Adapt the score of a box based on the fit with the stixels estimate
        box is a (x1, y1, x2, y2, score) tuple
        If the box is accepted will return a modified box
        If the box is rejected will return None
        """
        
        margin = stixels_vertical_margin
    
        if margin > 0:
             return self._adjust_score_using_stixels(box, stixels_data, margin)
        elif margin == 0:
            # raw detections, no filtering is done
            return box
        elif margin < 0:
           return self._adjust_score_using_ground_plane(box, stixels_data, -margin)
        else:
            # margin < 0
            raise "Received an unmanaged negative margin"
    
        return None
            

    def _filter_bounding_boxes(self, bounding_boxes, stixels_data, stixels_vertical_margin):
        """
        Given a set of bounding_boxes, will return a pruned bounding_boxes generator
        """
        
        for box in bounding_boxes:
            
            box_is_valid = self._box_is_allowed(box, stixels_data, stixels_vertical_margin)
        
            if box_is_valid:
                yield box
            else:
                # non valid boxes are skipped
                continue
        # end of for each box
            
        return
        
    def _rescore_bounding_boxes(self, bounding_boxes, stixels_data, stixels_vertical_margin):
        """
        Given a set of bounding_boxes, will return a bounding_boxes generator
        """
        
        for box in bounding_boxes:            
            adjusted_box = self._adjust_score(box, stixels_data, stixels_vertical_margin)
            if adjusted_box:
                yield adjusted_box
            else:
                # non valid boxes are skipped
                continue
            
        # end of for each box 
            
        return
        
    def _filter_bounding_boxes_by_height(self, boxes, minimum_height, maximum_height = float("inf")):
        
        for box in boxes:
            # should use the rectified boxes size instead ?
            #rectified_box = self.stereo_rectification.rectify_left_bounding_box(box)
            height = box[3] - box[1]
            assert height >= 0
            if height >= minimum_height and height <= maximum_height:
                yield box
        
        return
    
    def _compute_point_on_detection_rates_curves(self, recordings, margins, detection_threshold, progress_bar):


        minimum_box_height = self.options.minimum_box_height
        
        recording_name = self.recordings.keys()[0]
        num_accounted_frames = 0
             
        max_num_frames = self._get_max_num_frames()

        # retrieve detections idl
        detections_data_generator = open_idl_file(self.options.detections_path)
        # detections start at frame 0 but ground truth start at frame 1,
        # so we skip the first frame of detections
        detections_data_generator.next()
        
        # retrieve ground truth idl        
        ground_truth_data_generator = open_idl_file(self.options.ground_truth_path)    
                
        # re-open the recordings
        open_data_sequences(recordings)        
        recordings_stixels_data_sequence_generator = recordings_stixels_data_sequences_generator(recordings)
        
        num_ground_truth_boxes = 0
        num_considered_ground_truth_boxes = 0
        false_positives = dict()
        false_negatives = dict()
        for margin in margins:
            false_positives[margin] = 0
            false_negatives[margin] = 0
        
        generator = izip(ground_truth_data_generator, 
                         detections_data_generator,
                         recordings_stixels_data_sequence_generator)
                         
        # for each frame in the data
        for ground_truth_idl_data, detections_idl_data, recordings_stixels_data in generator:

            if num_accounted_frames > max_num_frames:
                #print("Reached indicated max number of frames (%i)" % max_num_frames )
                break

            num_accounted_frames += 1
            #print(idl_data.filename, stixels_data.image_name)
            progress_bar.widgets[0].message = " " + ground_truth_idl_data.filename
            progress_bar.update(progress_bar.currval + 1)            
            
            stixels_data = recordings_stixels_data[recording_name]
            
            num_ground_truth_boxes += len(ground_truth_idl_data.bounding_boxes)

            ground_truth_boxes = list(self._filter_bounding_boxes_by_height(ground_truth_idl_data.bounding_boxes, minimum_box_height))            
            detections_boxes = list(self._filter_bounding_boxes_by_height(detections_idl_data.bounding_boxes, minimum_box_height))            
                                                
            # all ground truth boxes of tolerable height are considered
            # important: ground truth boxes in occluded areas are still counted
            # this is not the case in _compute_error_histograms
            
            num_considered_ground_truth_boxes += len(ground_truth_boxes)
            
            for margin in margins:
                
                #filtered_detections = self._filter_bounding_boxes(detections_boxes, stixels_data, margin)
                filtered_detections = self._rescore_bounding_boxes(detections_boxes, stixels_data, margin)
                
                num_false_positives, num_false_negatives = count_errors(ground_truth_boxes, detection_threshold, filtered_detections)                
                false_positives[margin] += num_false_positives
                false_negatives[margin] += num_false_negatives
                
        # end of "for each frame in sequence"
        
        assert num_accounted_frames > 0
        
        # store num_accounted_frames for future reference
        self.num_accounted_frames = num_accounted_frames
        results_dict = dict()
        
        for margin in margins:

            print("For margin %i, with score threshold %f, we found %i false_positives, %i false_negatives, considering %i out of at total of %i pedestrians" % \
            (margin, detection_threshold, 
             false_positives[margin], false_negatives[margin], num_considered_ground_truth_boxes, num_ground_truth_boxes))            
            
            fppi = false_positives[margin] / float(num_accounted_frames)
            recall = (num_ground_truth_boxes - false_negatives[margin]) / float(num_ground_truth_boxes)
            
            results_dict[margin] = dict()
            results_dict[margin]["fppi"] = [fppi]
            results_dict[margin]["recall"] = [recall]
                
        return results_dict
    
    def _compute_plot_detection_rates(self, results, score_thresholds, progress_bar):
        
        margins = results.keys()
        
        def add_results_points(points_dict): 
            for margin in margins:
                results[margin]["fppi"] += points_dict[margin]["fppi"]
                results[margin]["recall"] +=  points_dict[margin]["recall"]   
            return

        def compute_point(threshold):
            
            # remove the datasequence generator since it cannot be deepcopied
            for recording in self.recordings.values():
                if hasattr(recording, "stixels_data_sequence_generator"):
                    del recording.stixels_data_sequence_generator
                
            recordings = deepcopy(self.recordings) # we deep copy to avoid interference between threads
            return self._compute_point_on_detection_rates_curves(recordings, margins, threshold, progress_bar)
        
        try:
            
            raise ImportError("disable futures")
            
            # try multithreaded version
            # based on http://www.python.org/dev/peps/pep-3148/
            # using http://code.google.com/p/pythonfutures
            # http://pypi.python.org/pypi/futures
            import futures, multiprocessing
            num_threads = multiprocessing.cpu_count() + 1
            print("Launching %i processing threads" % num_threads)
            with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            #with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                for points_dict in executor.map(compute_point, score_thresholds):
                    add_results_points(points_dict)
                                                                       
        except ImportError:
            # no futures available           
            print("Launching one processing thread")
            for points_dict in map(compute_point, score_thresholds):
                add_results_points(points_dict)
            
        return
    
    def _plot_detection_rates(self):
        
        if len(self.recordings) != 1:
            print("_plot_detection_rates currently only manages a single recording, ",
                  "skipping this step since more than one recording is being evaluated")
            return
        
     
        # the detection score is assumed to be between in [0, 1]
        threshold_steps = 0.05
        #threshold_steps = 0.25
        #threshold_steps = 0.125
        score_thresholds = numpy.arange(0,1,threshold_steps)        
        #stixels_margins = [10, 20, 30, 50]
        stixels_margins = [20, 30, 50]
        #stixels_margins = [30]
        
            
        minimum_box_height = self.options.minimum_box_height
        print("Using minimum_box_height == ", minimum_box_height)
            
        results = dict()
        for margin in stixels_margins:
            results[margin] = { "fppi":[], "recall":[] }
            # negative margins will be used to indicate the "ground plane corridor" results
            results[-margin] = { "fppi":[], "recall":[] }
            
        # 0 will be used to indicate the "raw detector" results
        results[0] = { "fppi":[], "recall":[] }
        
        progress_bar_message = progressbar.SimpleMessage()
        progress_bar_widgets = [progressbar.Percentage(), progress_bar_message, progressbar.Bar(), progressbar.ETA()]
        max_progress = self._get_num_frames() * len(score_thresholds)
        progress_bar = progressbar.ProgressBar(widgets = progress_bar_widgets, maxval=max_progress)        
           
        self._compute_plot_detection_rates(results, score_thresholds, progress_bar)
        
        progress_bar.finish()
        
        # create the actual plot ---
        
        # create figure
        pylab.figure()
        pylab.gcf().set_facecolor("w") # set white background            
        pylab.grid(True)
        
        margins = results.keys()
        
        # set the lines order
        margins.sort()
            
        
        colormap = pylab.cm.gist_rainbow
        #colormap = pylab.cm.rainbow    
         
        for index, margin in izip(xrange(len(margins)), margins):
            fppi_and_recall = results[margin]
            fppi = fppi_and_recall["fppi"]            
            recall = fppi_and_recall["recall"]

            #print("score_thresholds", score_thresholds)            
            #print("fppi", fppi)            
            #print("recall", recall)            
            
            linewidth = 1.5
            linestyle = "-"
            color_index = index/float(max(1, len(margins)-1))
            color = colormap(color_index)
            
            if margin > 0:
                label = "stixels margin %i [pixels]" % margin
            elif margin == 0:
                label = "raw detector"
                linewidth = 2
                linestyle = "--"
                color = "black"
            elif margin < 0:
                label = "ground corridor margin %i [pixels]" % -margin
                linestyle = "-."
                print("corridor fppi", fppi)
                print("corridor recall", recall)                
            else:
                raise "Cannot plot unknown negative margin value"
            
              
            #line = 
            pylab.plot(fppi, recall, linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        # end of "plot each curve for each margin"
        
        #xticks_steps = 10
        #xticks = numpy.arange(min_error, max_error, xticks_steps)
        #xticks = numpy.arange(min_error, min(101, max_error), xticks_steps)
        #xticks = numpy.arange(min_error, min(71, max_error), xticks_steps)
        #xticks = range(min_error, 60, 10) + range(60, max_error, 20)                                 
        #pylab.xticks(xticks)

        #yticks_steps = 0.05
        #yticks_steps = 0.1
        #yticks = numpy.arange(0.001, 1.001, yticks_steps)
        #yticks = numpy.arange(0.00, 1.001, yticks_steps)
        #linthresh = 0.2; sub = range(0, 10)
        #pylab.yscale("log", linthreshy = linthresh, subsy=sub)        
        #pylab.yticks(yticks)   
        
        #pylab.xlim([0, 100])  
        #pylab.xlim([0, 70])  
        
        pylab.legend(loc ="lower right", fancybox=True)
        pylab.xlabel("False positives per image")
        pylab.ylabel("Recall")
        
        # this measure also counts the occluded windows
        title = "Recall versus FPPI over " \
                "%s,\nconsidering all windows with height > %i [pixels]" % (self._get_dataset_name(), minimum_box_height)
        pylab.title(title)
        
        pylab.draw()
        
        return
        
# end of class StixelsEvaluationApplication

        

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
   
    application = StixelsEvaluationApplication()
    application.run()

