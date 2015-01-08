#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a recording of detections, will extract the negative samples
"""

from __future__ import print_function
import os
from create_multiscales_training_dataset import create_positives, Box
from detections_to_precision_recall import open_data_sequence

from optparse import OptionParser


def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates occluded pedestrians"

    parser.add_option("-i", "--input", dest="input_file",
                      metavar="PATH", type="string",
                      help="path to the .data_sequence file")

    parser.add_option("-o", "--output", dest="output_path",
                      metavar="DIRECTORY", type="string",
                      help="path to a non existing directory where the "
                      "new training dataset will be created")

    parser.add_option("-n", "--number_of_samples", dest="number_of_samples",
                      type="int", default=-1,
                      help="Number of detections to sample. "
                      "If -1 will use all detections, "
                      "otherwise will use the top N detections")

    (options, args) = parser.parse_args()

    if options.input_file:
        if os.path.isdir(options.input_file):
            parser.error("'input_file' should point to a file, "
                         "not a directory")
    else:
        parser.error("'input_file' option is required to run this program")

    #if not options.output_path:
    #    parser.error("'output' option is required to run this program")
    #    if os.path.exists(options.output_path):
    #		parser.error("output_path already exists")

    return options


def get_annotations(detections_sequence, number):

    #inria_negatives_path = "/esat/kochab/mmathias/INRIAPerson/Train/neg"
    inria_negatives_path = "/home/rodrigob/data/INRIAPerson/Train/neg"

    output_detections = []
    negCount = 0
    posCount = 0
    all_detections = []

    for detections in detections_sequence:
        detections.image_name
        for detection in detections.detections:
            box = Box()
            box.min_corner.x = detection.bounding_box.min_corner.x
            box.min_corner.y = detection.bounding_box.min_corner.y
            box.max_corner.x = detection.bounding_box.max_corner.x
            box.max_corner.y = detection.bounding_box.max_corner.y

            image_path = os.path.join(inria_negatives_path,
                                      detections.image_name)

            if not os.path.exists(image_path):
                image_path_png = os.path.splitext(image_path)[0] + ".jpg"
                if os.path.exists(image_path_png):
                    image_path = image_path_png
                else:
                    raise Exception("Image %s "
                                    "(or its jpg version) does not exists"
                                    % image_path)
            detection_data = [
                image_path,
                detection.score,
                box]
            all_detections.append(detection_data)
        # end of "for all detections in this frame"
    # end of "for each frame"

    #sort detections by score
    all_detections = sorted(all_detections,
                            key=lambda det: det[1],
                            reverse=True)
    #print (all_detections[:100])
    annotations = []
    if number > 0:
        if len(all_detections) > number:
            all_detections = all_detections[:number]
        else:
            print("number of detections provided: ", len(all_detections))
            print("number of detections requested: ", number)
            raise Exception("Not enough detections provided")

    print("Going to sample %i detections" % len(all_detections))

    #sort detections by filename
    all_detections = sorted(all_detections,
                            key=lambda det: det[0],
                            reverse=True)
    #print (all_detections[:100])
    previous_filename = ""
    boxes = []
    for det in all_detections:
        fn = det[0]
        if fn == previous_filename:
            boxes.append(det[2])
        else:
            if (previous_filename != ""):
                annotation = (previous_filename, boxes)
                annotations.append(annotation)
            previous_filename = det[0]
            boxes = [det[2]]

    #print(annotations)
    return (annotations)

    return [output_detections, posCount, negCount]


def main():
    options = parse_arguments()
    os.mkdir(options.output_path)
    number_of_samples = options.number_of_samples

    #get annotations
    detections_sequence = open_data_sequence(options.input_file)

    annotations = get_annotations(detections_sequence, number_of_samples)

    #generate data
    model_width, model_height = 64, 128

    # how many pixels around the model box to define a test image ?
    cropping_border = 20  # on own multiscales models
    #cropping_border = 16  # on original INRIA dataset

    octave = 0

    create_positives(model_width, model_height, cropping_border,
                     octave, annotations, options.output_path)


if __name__ == "__main__":
    main()

# end of file
