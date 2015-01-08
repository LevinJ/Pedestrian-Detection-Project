#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program parses the INRIAPerson dataset and creates
negative samples, sampled from the positive data
"""

from __future__ import print_function

import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")
sys.path.append("/users/visics/rbenenso/no_backup/"
                "usr/local/lib/python2.7/site-packages")
import random

#from detections_pb2 import Box
#from data_sequence import DataSequence

import os, os.path, glob
from optparse import OptionParser
from multiprocessing import Pool, Lock, cpu_count
import itertools
import cv2

import create_multiscales_training_dataset as cmtd

import math, pylab
box_counter =0
failures_count =0

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

    def copy(self):
        b = Box()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b
    def intersect(self,anotherBox):
        x0 = max(self.min_corner.x, anotherBox.min_corner.x)
        x1 = min(self.max_corner.x, anotherBox.max_corner.x)
        y0 = max(self.min_corner.y, anotherBox.min_corner.y)
        y1 = min(self.max_corner.y, anotherBox.max_corner.y)
        if x1 <=x0 or y1 <= y0:
            return Box()
        box = Box()
        box.min_corner = Point2d(x0,y0)
        box.max_corner = Point2d(x1,y1)
        return box
    def area(self):
        w = self.max_corner.x - self.min_corner.x
        h = self.max_corner.y - self.min_corner.y
        return w*h
    def intersectOverMinArea(self, anotherBox):
        intersectBox= self.intersect(anotherBox)
        minarea = min(self.area(), anotherBox.area())
        return float(intersectBox.area())/float(minarea)

    def __eq__(self, other):
        return \
        self.min_corner.x == other.min_corner.x and \
        self.min_corner.y == other.min_corner.y and \
        self.max_corner.x == other.max_corner.x and \
        self.max_corner.y == other.max_corner.y

    def width(self):
        return self.max_corner.x - self.min_corner.x

    def height(self):
        return self.max_corner.y - self.min_corner.y
    def print(self):
        print ('(x1:%d, y1:%d), (x2:%d, y2:%d)' %(self.min_corner.x, self.min_corner.y, self.max_corner.x, self.max_corner.y))

def adjust_box_ratio(box, model_width, model_height):

    box_height = box.max_corner.y - box.min_corner.y
    box_width = box_height*(model_width/model_height)
    box_x_center = (box.max_corner.x + box.min_corner.x) / 2
    box.min_corner.x = box_x_center - box_width/2
    box.max_corner.x = box_x_center + box_width/2
    return box


def adjust_box_border(box, top_bottom_border, left_right_border):

    assert top_bottom_border > 0
    assert left_right_border > 0
    box = box.copy() # to be sure we are not messing the data
    box.min_corner.x -= left_right_border
    box.min_corner.y -= top_bottom_border
    box.max_corner.x += left_right_border
    box.max_corner.y += top_bottom_border

    # image boxes only make sense with integer values
    box.min_corner.x = int(round(box.min_corner.x))
    box.min_corner.y = int(round(box.min_corner.y))
    box.max_corner.x = int(round(box.max_corner.x))
    box.max_corner.y = int(round(box.max_corner.y))

    return box




def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates a new training dataset for multiscale objects detection"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string",
                       help="path to the INRIAPerson dataset Test or Train folder")

    parser.add_option("-o", "--output", dest="output_path",
                       metavar="DIRECTORY", type="string",
                       help="path to a non existing directory where the new training dataset will be created")

    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            # we normalize the path
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to a non existing directory")
    else:
        parser.error("'output' option is required to run this program")

    return options




def compute_jittered_positive_sample(input_image, jitter):
    image_height, image_width, image_depth = input_image.shape
    top_border = abs(min(jitter[1], 0 ))
    bottom_border = abs(max(jitter[1], 0))
    left_border = abs(min(jitter[0], 0))
    right_border = abs(max(jitter[0], 0))
    border_type = cv2.BORDER_REPLICATE
    bigger_image = cv2.copyMakeBorder(input_image,
                                       top_border, bottom_border,
                                       left_border, right_border,
                                       border_type)
    x1=max(jitter[0], 0)
    y1=max(jitter[1], 0)
    x2=x1 + image_width
    y2=y1 + image_height
    cropped_image=bigger_image[y1:y2,x1:x2, :]
    assert (input_image.shape == cropped_image.shape)
    return cropped_image



def compute_resized_positive_example(input_image, input_image_path,
                                     box, box_scale, model_octave,
                                     cropping_border, model_width, model_height):
    """
    The input box is expected to be a tight box around the pedestrian
    """

    assert input_image != None
    assert type(model_width) is int
    assert type(model_height) is int
    assert type(cropping_border) is int

    image_height, image_width, image_depth = input_image.shape

    # adjust the ratio, add the border --

    # adjusting the ratio is a bad idea, pedestrians look "fat"
    # instead, we adjust the left/right border
    #box = adjust_box_ratio(box, model_width, model_height)

    example_width = model_width+(2*cropping_border)
    example_height = model_height+(2*cropping_border)

    model_scale = 2**model_octave
    box_relative_scale = box_scale/model_scale
    desired_input_box_width = example_width * box_relative_scale
    desired_input_box_height = example_height * box_relative_scale

    input_box_width, input_box_height = box.width(), box.height()

    top_bottom_border = (desired_input_box_height - input_box_height)/2.0
    left_right_border = (desired_input_box_width - input_box_width)/2.0

    desired_input_box_shape = (desired_input_box_height, desired_input_box_width)

    if (left_right_border <= 0) or False:
        print("model width, height ==", (model_width, model_height))
        print("example_width ==", example_width)
        print("box_scale ==", box_scale)
        print("model_scale ==", model_scale)
        print("box_relative_scale ==", box_relative_scale)
        print("desired_input_box_width ==", desired_input_box_width)
        print("input_box_width ==", input_box_width)
        print("input_box_height ==", input_box_height)
        print("input_box ratio ==", input_box_height/float(input_box_width))
        p1 = (int(box.min_corner.x), int(box.min_corner.y))
        p2 = (int(box.max_corner.x), int(box.max_corner.y))
        print("input_box min_corner, max_corner == ", p1, ",", p2)
        color = (255, 10, 50)
        failure_image = input_image.copy()
        cv2.rectangle(failure_image, p1, p2, color)
        global failures_count
        failure_filename = "failure_case_%i_%s" % (failures_count, os.path.basename(input_image_path))
        cv2.imwrite(failure_filename, failure_image)
        print("Created %s skipping one bounding_box in picture %s" % (failure_filename, input_image_path))
        failures_count += 1
        return None # indicate that something went wrong

    if (top_bottom_border < 0):
        print("model width, height ==", (model_width, model_height))
        print("example_width ==", example_width)
        print("box_scale ==", box_scale)
        print("model_scale ==", model_scale)
        print("box_relative_scale ==", box_relative_scale)
        print("desired_input_box_width ==", desired_input_box_width)
        print("input_box_width ==", input_box_width)
        print("input_box_height ==", input_box_height)
        print("input_box ratio ==", input_box_height/float(input_box_width))
        print("desired width ==", desired_input_box_width)
        print("desired height==", desired_input_box_height)
        p1 = (int(box.min_corner.x), int(box.min_corner.y))
        p2 = (int(box.max_corner.x), int(box.max_corner.y))
        print("input_box min_corner, max_corner == ", p1, ",", p2)
        color = (255, 10, 50)
        failure_image = input_image.copy()
        cv2.rectangle(failure_image, p1, p2, color)
        global failures_count
        failure_filename = "failure_case_%i_%s" % (failures_count, os.path.basename(input_image_path))
        cv2.imwrite(failure_filename, failure_image)
        print("Created %s skipping one bounding_box in picture %s" % (failure_filename, input_image_path))
        failures_count += 1
        raise "adflkj"
        return None # indicate that something went wrong



    box = adjust_box_border(box, top_bottom_border, left_right_border)
    # the box now has the desired_input_box_width and _height

    box_shape = (box.height(), box.width())
    #print(box_shape, "=?=", desired_input_box_shape)
    assert abs(box_shape[0] - desired_input_box_shape[0]) <= 1.0
    assert abs(box_shape[1] - desired_input_box_shape[1]) <= 1.0


    # crop the box --
    # which part of the box is outside the image ?
    left_border = int(max(0, 0 - box.min_corner.x))
    top_border = int(max(0, 0 - box.min_corner.y))
    right_border = int(max(0, box.max_corner.x - image_width))
    bottom_border = int(max(0, box.max_corner.y - image_height))

    # define the part of the box that is inside the image
    inner_box = box.copy()
    inner_box.min_corner.x = max(0, box.min_corner.x)
    inner_box.min_corner.y = max(0, box.min_corner.y)
    inner_box.max_corner.x = min(box.max_corner.x, image_width)
    inner_box.max_corner.y = min(box.max_corner.y, image_height)

    # crop
    cropped_image = input_image[inner_box.min_corner.y:inner_box.max_corner.y,
                                inner_box.min_corner.x:inner_box.max_corner.x,
                               :]
    # extrapolate --

    #border_type = cv2.BORDER_CONSTANT
    border_type = cv2.BORDER_REPLICATE
    #border_type = cv2.BORDER_WRAP
    #border_type = cv2.BORDER_REFLECT
    #border_type = cv2.BORDER_REFLECT_101
    cropped_image = cv2.copyMakeBorder(cropped_image,
                                       top_border, bottom_border,
                                       left_border, right_border,
                                       border_type)


    #print(cropped_image.shape[:2], "=?=", box_shape)
    #assert cropped_image.shape[:2] == box_shape


    # rescale the box so it fits the desired dimensions --
    resized_positive_example = cmtd.resized_image(cropped_image, example_width, example_height)
    assert resized_positive_example.shape == (example_height, example_width, image_depth)

    return resized_positive_example


print_counter = 0

def sample_new_boxes(all_boxes, annotation_box, image_height, image_width, scale):
    min_scale = 0.79
    max_scale = 8.6
    object_height =96
    boxes = []
    count =0
    while len(boxes) < 2 and count < 50000:
        sampled_scale = random.randrange(0,9) + random.random()
        count = count +1;
        if (sampled_scale >= min_scale) and (sampled_scale<= max_scale) and (sampled_scale < 2*scale):
            height = round(sampled_scale * object_height)
            sampled_scale =  float(height)/float(object_height)
            width  = round(height /2.0)
            if height >= image_height or width >= image_width:
                continue
            minxpos = max(0,annotation_box.min_corner.x-width)
            maxxpos = min(image_width-width, annotation_box.max_corner.x+width)

            minypos = max(0,annotation_box.min_corner.y-height)
            maxypos = min(image_height-height, annotation_box.max_corner.y+height)

            x = random.randrange(minxpos, maxxpos)
            y = random.randrange(minypos, maxypos)
            sample_box = Box()
            sample_box.min_corner = Point2d(x,y)
            sample_box.max_corner = Point2d(x+width, y+height)
            overlaps = False
            annotation_overlap = sample_box.intersectOverMinArea(annotation_box)
            if annotation_overlap > 0.2 and annotation_overlap < 0.5:
                overlaps = True
            for box in all_boxes:
                box2 = Box()
                box2.min_corner = box.min_corner
                box2.max_corner = box.max_corner
                box = box2
                if box == sample_box:
                    continue
                intersectionOverMinArea = box.intersectOverMinArea(sample_box)
                if (intersectionOverMinArea > 0.5):
                    overlaps = False
            if overlaps:
                boxes.append([sample_box, sampled_scale, annotation_overlap])



    return boxes


def create_negatives_process_annotation_box(image_counter, image_and_annotations, positives_path,
                                            model_width, model_height, cropping_border, octave):

    global box_counter
    object_height = 96 # the expected height of the pedestrian inside the detection box
    # when create_blurred is true, will create
    # blurry scaled up images (ignoring the scales constraints)
    #create_blurred = False
    create_blurred = True

    image_path, all_boxes = image_and_annotations
    for box in all_boxes:
        box2 = Box()
        box2.min_corner = box.min_corner
        box2.max_corner = box.max_corner
        box = box2

        height = box.max_corner.y - box.min_corner.y
        scale = height / float(object_height)
        box_octave = round(math.log(scale)/math.log(2))

        is_bigger = (box_octave >= octave)
        if is_bigger or create_blurred:
            #print("Reading :", image_path)
            input_image = cv2.imread(image_path)
            #generate negatives with overlap < 0.5
            negative_boxes = sample_new_boxes(
                all_boxes, box,
                input_image.shape[0], input_image.shape[1],
                scale)

            for negative_box in negative_boxes:
                example = compute_resized_positive_example(
                    input_image, image_path,
                    negative_box[0], negative_box[1], octave,
                    cropping_border, model_width, model_height)
                filename_base = \
                    "hard_negative_sample_%i_%1.2f" % (box_counter,
                                                       negative_box[2])
                example_path = os.path.join(positives_path, filename_base)

                cmtd.save_positive_example(example, example_path,
                                           is_bigger, False)
                box_counter = box_counter + 1

                global print_counter
                print_counter += 1  # no need to be thread safe
                if print_counter % 10:
                    sys.stdout.flush()
        else:
            # we skip this sample
            pass
        # end of "if bigger or created blurred"

        #if create_blurred:
        #    print("These examples include blurry, upscaled, images")

    return
# end of "def create_negatives_process_annotation_box"


class Lambda:

    def __init__(self, f, args_tuple):
        self.args_tuple = args_tuple
        self.f = f
        return

    def __call__(self, args):
        #print("Lambda.__call__ with args", args)
        return self.f(*(args + self.args_tuple))


def create_hard_negatives(model_width, model_height,
                     cropping_border, octave,
                     annotations,
                     output_path):
    """
    Create the negative examples
    """

    # we expand all the anotations in the a long list of
    # (filename, box)
    # each box represent a pedestrian
    #annotation_boxes = itertools.chain.from_iterable(
    #            [itertools.product([x[0]], x[1]) for x in annotations])

    # create the output folder
    hard_negatives_path = os.path.join(output_path, "hard_negatives_octave_%.1f" % octave)
    if os.path.exists(hard_negatives_path):
        raise RuntimeError(hard_negatives_path + " should not exist")

    os.mkdir(hard_negatives_path)
    print("Created folder", hard_negatives_path)

    data = list(enumerate(annotations))
    #print("data[:10] ==", data[:10])
    g = Lambda(create_negatives_process_annotation_box,
                   (hard_negatives_path, model_width, model_height, cropping_border, octave))

    #use_multithreads = True
    use_multithreads = False
    if use_multithreads:
        # multithreaded processing of the files
        num_processes = cpu_count() + 1
        pool = Pool(processes = num_processes)
        #g = lambda z: f(z[0], z[1])
        chunk_size = len(data) / num_processes
        pool.map(g, data, chunk_size)

    else:
        for d in data:
            g(d)


    # end of "for each annotation file"
    print()
    print("Created the positives examples for octave %.1f" % (octave) )
    #if create_flipped:
    #    print("For octave %.1f created %i positive examples (and its mirror)" % (octave, ns.sample_counter) )
    #else:
    #    print("For octave %.1f created %i positive examples" % (octave, ns.sample_counter) )

    #if create_blurred:
    #    print("These examples include blurry, upscaled, images")

    return
# end of "def create_hard_negatives"



def create_negatives(data_path, output_path, annotations,
                                   model_width, model_height, cropping_border,
                                   octave):

    model_width, model_height = int(model_width*(2**octave)), int(model_height*(2**octave))
    octave_cropping_border = int(cropping_border*(2**octave))

    # make sure model size is pair, so we have a well defined center
    if (model_width % 2) == 1: model_width += 1
    if (model_height % 2) == 1: model_height += 1

    print("At octave %.1f, model (width, height) == %s and cropping_border == %i" %
            (octave, str((model_width, model_height)), octave_cropping_border))

    create_hard_negatives(model_width, model_height,
                     octave_cropping_border, octave,
                     annotations,
                     output_path)

    return
# end of "def create_positives_and_negatives"


def sample_negatives_from_inria_positive_set(data_path, output_path):


    # check thet data path is either Train or Test inside INRIAPerson
    inria_person_path, subfolder_name = os.path.split(data_path)
    assert os.path.isdir(data_path)
    assert os.path.basename(inria_person_path) == "INRIAPerson"
    assert subfolder_name in ["Train", "Test"]

    annotations = list(cmtd.annotations_to_filenames_and_boxes(data_path))

    # define the scales we care about
    min_scale, max_scale = 0.6094, 8.6 # this is the scales range used for INRIA
    delta_octave = 1.0
    #delta_octave = 0.5

    model_width, model_height = 64, 128
    cropping_border = 20 # how many pixels around the model box to define a test image ?


    min_octave = int(round(math.log(min_scale)/math.log(2)))
    max_octave = int(round(math.log(max_scale)/math.log(2)))
    #max_octave=-1
    octaves = list(pylab.frange(min_octave, max_octave, delta_octave))
    #octaves.reverse() # for debugging, it is easier to look at big pictures first

    if os.path.exists(output_path):
        raise RuntimeError(output_path + " should not exist")
    else:
        os.mkdir(output_path)
        print("Created folder", output_path)

    print("Will create data for octaves", octaves)

    for octave in octaves:
        create_negatives(data_path, output_path, annotations,
                                       model_width, model_height, cropping_border,
                                       octave)
    # end of "for each octave"

    return
# end of "def sample_negatives_from_inria_positive_set"



def main():
    options = parse_arguments()
    sample_negatives_from_inria_positive_set(options.input_path, options.output_path)

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





